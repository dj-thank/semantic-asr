import hashlib

import pytest

from semantic_asr.realtime_refine import (
    BoundedPcmHistory,
    GroupedRefineConfig,
    GroupedRefineRequest,
    GroupedRefineScheduler,
    RefineParentFinal,
)

SR = 16_000


def samples(milliseconds: int) -> int:
    return round(SR * milliseconds / 1000)


def pcm(milliseconds: int, *, value: int = 1) -> bytes:
    return int(value).to_bytes(2, "little", signed=True) * samples(milliseconds)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def parent(
    number: int,
    start_ms: int,
    end_ms: int,
    *,
    text: str | None = None,
) -> RefineParentFinal:
    return RefineParentFinal(
        utterance_id=f"u{number}",
        final_digest=digest(f"final:{number}"),
        audio_sha256=digest(f"audio:{number}"),
        start_sample=samples(start_ms),
        end_sample=samples(end_ms),
        observed_text=text if text is not None else f"text-{number}",
    )


def test_history_requires_contiguous_pcm_and_returns_exact_ranges():
    history = BoundedPcmHistory(sample_rate=SR, keep_samples=samples(2_000))
    a = pcm(1_000, value=1)
    b = pcm(1_000, value=2)
    history.append(a)
    history.append(b)

    assert history.start_sample == 0
    assert history.end_sample == samples(2_000)
    assert history.read(0, samples(2_000)) == a + b
    assert history.read(samples(1_000), samples(2_000)) == b

    with pytest.raises(ValueError, match="contiguous"):
        history.append(pcm(10), start_sample=history.end_sample + 1)


def test_history_trims_old_pcm_without_reindexing_absolute_time():
    history = BoundedPcmHistory(sample_rate=SR, keep_samples=samples(1_000))
    history.append(pcm(750, value=1))
    history.append(pcm(750, value=2))

    assert history.start_sample == samples(500)
    assert history.end_sample == samples(1_500)
    assert history.retained_samples == samples(1_000)
    expected = pcm(250, value=1) + pcm(750, value=2)
    assert history.read(samples(500), samples(1_500)) == expected
    with pytest.raises(ValueError, match="outside retained history"):
        history.read(0, samples(100))


def test_idle_gap_closes_once_at_two_seconds():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(1_000, value=1))
    assert scheduler.add_final(parent(1, 0, 1_000)) == ()

    assert scheduler.feed_pcm16(pcm(1_999, value=0)) == ()
    events = scheduler.feed_pcm16(pcm(1, value=0))
    assert len(events) == 1
    request = events[0]
    assert request.trigger == "idle-gap"
    assert [item.utterance_id for item in request.parents] == ["u1"]
    assert request.start_sample == 0
    assert request.end_sample == samples(1_000)
    assert request.pcm16le == pcm(1_000, value=1)
    assert scheduler.pending == ()
    assert scheduler.feed_pcm16(pcm(500, value=0)) == ()


def test_group_audio_comes_from_continuous_timeline_not_parent_buffer_concatenation():
    scheduler = GroupedRefineScheduler(session_id="s")
    first = pcm(1_000, value=1)
    gap = pcm(1_000, value=7)
    second = pcm(1_000, value=2)
    scheduler.feed_pcm16(first)
    scheduler.feed_pcm16(gap)
    scheduler.feed_pcm16(second)

    assert scheduler.add_final(parent(1, 0, 1_000)) == ()
    assert scheduler.add_final(parent(2, 2_000, 3_000)) == ()
    request = scheduler.force("manual")[0]

    assert request.pcm16le == first + gap + second
    assert len(request.pcm16le) == len(first + gap + second)
    assert request.group_audio_sha256 == hashlib.sha256(first + gap + second).hexdigest()
    assert [item.final_digest for item in request.parents] == [digest("final:1"), digest("final:2")]


def test_new_final_after_true_idle_closes_previous_group_before_append():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(4_000))
    scheduler.add_final(parent(1, 0, 1_000))

    events = scheduler.add_final(parent(2, 3_000, 4_000))
    assert len(events) == 1
    assert events[0].trigger == "idle-gap"
    assert [item.utterance_id for item in events[0].parents] == ["u1"]
    assert [item.utterance_id for item in scheduler.pending] == ["u2"]


def test_max_duration_splits_before_new_parent_would_exceed_25_seconds():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(27_000))
    scheduler.add_final(parent(1, 0, 12_000))
    scheduler.add_final(parent(2, 13_000, 24_000))

    events = scheduler.add_final(parent(3, 24_500, 27_000))
    assert len(events) == 1
    closed = events[0]
    assert closed.trigger == "max-duration"
    assert closed.duration_samples == samples(24_000)
    assert [item.utterance_id for item in closed.parents] == ["u1", "u2"]
    assert [item.utterance_id for item in scheduler.pending] == ["u3"]


def test_exact_max_duration_closes_current_group_immediately():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(25_000))
    scheduler.add_final(parent(1, 0, 12_000))
    events = scheduler.add_final(parent(2, 13_000, 25_000))
    assert len(events) == 1
    assert events[0].trigger == "max-duration"
    assert events[0].duration_samples == samples(25_000)
    assert scheduler.pending == ()


def test_parent_count_is_finitely_bounded():
    scheduler = GroupedRefineScheduler(
        config=GroupedRefineConfig(max_pending_finals=2),
        session_id="s",
    )
    scheduler.feed_pcm16(pcm(3_000))
    scheduler.add_final(parent(1, 0, 1_000))
    scheduler.add_final(parent(2, 1_000, 2_000))
    events = scheduler.add_final(parent(3, 2_000, 3_000))
    assert len(events) == 1
    assert events[0].trigger == "parent-limit"
    assert [item.utterance_id for item in events[0].parents] == ["u1", "u2"]
    assert [item.utterance_id for item in scheduler.pending] == ["u3"]


def test_short_group_is_preserved_as_explicit_skip_not_silent_loss():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(400))
    scheduler.add_final(parent(1, 0, 400))
    request = scheduler.force("eof")[0]
    assert request.trigger == "eof"
    assert request.eligible_for_decode is False
    assert request.skip_reason == "group-too-short"
    assert request.evidence_digest


def test_eof_force_flushes_pending_group_without_fabricating_silence():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(1_250, value=3))
    scheduler.add_final(parent(1, 100, 1_200))
    request = scheduler.force("eof")[0]
    assert request.trigger == "eof"
    assert request.start_sample == samples(100)
    assert request.end_sample == samples(1_200)
    assert request.eligible_for_decode is True
    assert scheduler.force("eof") == ()


def test_parent_must_still_be_bound_to_retained_continuous_audio():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(31_000))
    assert scheduler.history.start_sample == samples(1_000)
    with pytest.raises(ValueError, match="outside retained"):
        scheduler.add_final(parent(1, 0, 500))


def test_parent_timeline_must_strictly_advance():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(3_000))
    scheduler.add_final(parent(1, 500, 2_000))
    with pytest.raises(ValueError, match="strictly advancing"):
        scheduler.add_final(parent(2, 400, 1_900))


def test_request_rejects_tampered_group_audio_digest():
    data = pcm(1_000)
    p = parent(1, 0, 1_000)
    with pytest.raises(ValueError, match="group_audio_sha256"):
        GroupedRefineRequest(
            sequence=1,
            session_id="s",
            group_id="s:g1",
            trigger="eof",
            parents=(p,),
            pcm16le=data,
            sample_rate=SR,
            start_sample=0,
            end_sample=samples(1_000),
            group_audio_sha256="0" * 64,
            eligible_for_decode=True,
        )


def test_request_rejects_duplicate_or_reordered_parent_proof():
    data = pcm(3_000)
    p1 = parent(1, 0, 1_000)
    p2 = parent(2, 2_000, 3_000)
    common = dict(
        sequence=1,
        session_id="s",
        group_id="s:g1",
        trigger="eof",
        pcm16le=data,
        sample_rate=SR,
        start_sample=0,
        end_sample=samples(3_000),
        group_audio_sha256=hashlib.sha256(data).hexdigest(),
        eligible_for_decode=True,
    )
    with pytest.raises(ValueError, match="duplicate"):
        GroupedRefineRequest(parents=(p1, p1), **common)
    with pytest.raises(ValueError, match="ordered"):
        GroupedRefineRequest(parents=(p2, p1), **common)


def test_reset_requires_explicit_flush_or_discard_and_isolates_session():
    scheduler = GroupedRefineScheduler(session_id="old")
    scheduler.feed_pcm16(pcm(1_000))
    scheduler.add_final(parent(1, 0, 1_000))

    flushed, discarded = scheduler.reset(flush_pending=False, new_session_id="new")
    assert flushed == ()
    assert discarded is not None
    assert discarded.session_id == "old"
    assert discarded.parent_final_digests == (digest("final:1"),)
    assert discarded.reason == "reset-without-refine"
    assert scheduler.session_id == "new"
    assert scheduler.history.start_sample == 0
    assert scheduler.history.end_sample == 0
    assert scheduler.pending == ()

    scheduler.feed_pcm16(pcm(1_000, value=2))
    scheduler.add_final(parent(2, 0, 1_000))
    flushed, discarded = scheduler.reset(flush_pending=True, new_session_id="third")
    assert discarded is None
    assert len(flushed) == 1
    assert flushed[0].session_id == "new"
    assert flushed[0].trigger == "reset"
    assert scheduler.session_id == "third"
    assert scheduler.pending == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 8_000},
        {"idle_gap_ms": 0},
        {"max_group_ms": 0},
        {"min_group_ms": 0},
        {"min_group_ms": 26_000},
        {"history_keep_ms": 26_999},
        {"max_pending_finals": 0},
    ],
)
def test_invalid_scheduler_config_fails_closed(kwargs):
    with pytest.raises((TypeError, ValueError)):
        GroupedRefineConfig(**kwargs)


def test_invalid_parent_digests_fail_closed():
    with pytest.raises(ValueError, match="final_digest"):
        RefineParentFinal(
            utterance_id="u",
            final_digest="not-a-digest",
            audio_sha256=digest("audio"),
            start_sample=0,
            end_sample=1,
            observed_text="x",
        )
