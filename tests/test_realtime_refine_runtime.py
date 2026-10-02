"""Model-free integration and blocked-worker tests (not acoustic accuracy evidence)."""

import hashlib
import json
import math
import threading
from dataclasses import FrozenInstanceError, replace

import pytest

from semantic_asr.realtime_reazon import RealtimeDecodeInput, RealtimeEvent, RealtimeReazonConfig
from semantic_asr.realtime_refine import (
    GroupedRefineConfig,
    GroupedRefineScheduler,
    RefineParentFinal,
)
from semantic_asr.realtime_refine_runtime import (
    GroupedRealtimeReazon,
    GroupedRefineOutcome,
    _FifoRefiner,
)


def pcm(ms=100, value=1):
    return value.to_bytes(2, "little", signed=True) * (ms * 16)


def request(number=1, *, milliseconds=500):
    data = pcm(milliseconds, number)
    digest = hashlib.sha256(data).hexdigest()
    scheduler = GroupedRefineScheduler(session_id=f"s{number}")
    scheduler.feed_pcm16(data, speech=True)
    scheduler.add_final(
        RefineParentFinal(
            utterance_id=f"u{number}",
            final_digest="a" * 64,
            audio_sha256=digest,
            start_sample=0,
            end_sample=len(data) // 2,
            observed_text="ええ、買わない、買わない。",
        )
    )
    return scheduler.force("eof")[0]


def worker(factory, *, capacity=2, max_text_chars=100):
    return _FifoRefiner(
        factory, decoder_id="pinned-test-decoder", capacity=capacity, max_text_chars=max_text_chars
    )


def runtime(factory, **kwargs):
    return GroupedRealtimeReazon(
        kwargs.pop("decoder", lambda _: "ええ、買わない、買わない。"),
        refine_decoder_factory=factory,
        refine_decoder_id="pinned-test-decoder",
        config=RealtimeReazonConfig(
            partial_interval_ms=100,
            min_silence_ms=100,
            max_speech_ms=600,
            preroll_ms=100,
            max_chunk_ms=100,
        ),
        group_config=GroupedRefineConfig(
            idle_gap_ms=300, max_group_ms=2000, min_group_ms=200, history_keep_ms=2300
        ),
        session_id="test-session",
        **kwargs,
    )


def utterance(rt, value=1):
    events = []
    for _ in range(3):
        events.extend(rt.feed_pcm16(pcm(value=value), speech=True))
    events.extend(rt.feed_pcm16(pcm(value=0), speech=False))
    return events


def idle(rt):
    events = []
    for _ in range(3):
        events.extend(rt.feed_pcm16(pcm(value=0), speech=False))
    return events


def refinements(events):
    return [e for e in events if isinstance(e, GroupedRefineOutcome)]


def test_factory_is_lazy_and_audio_only_decoder_is_warm_and_fifo():
    calls, creations = [], []
    owner = threading.get_ident()

    def factory():
        creations.append(threading.get_ident())

        def decode(item):
            assert isinstance(item, RealtimeDecodeInput)
            assert not hasattr(item, "parents")
            calls.append(item)
            return f"候補:{item.session_id}"

        return decode

    w = worker(factory)
    assert creations == [] and not w.is_alive
    a, b = request(1), request(2)
    w.submit(a)
    w.submit(b)
    result = w.close(5)
    assert [e.request.group_id for e in result] == [a.group_id, b.group_id]
    assert [c.pcm16le for c in calls] == [a.pcm16le, b.pcm16le]
    assert [c.audio_sha256 for c in calls] == [a.group_audio_sha256, b.group_audio_sha256]
    assert [c.mode for c in calls] == ["final", "final"]
    assert len(creations) == 1 and creations[0] != owner
    assert not w.is_alive and w.pending_count == 0


def test_completed_but_unconsumed_work_also_uses_queue_capacity():
    entered = threading.Event()

    def decode(_):
        entered.set()
        return "候補"

    w = worker(lambda: decode, capacity=1)
    try:
        w.submit(request(1))
        assert entered.wait(5)
        assert w.submit(request(2)).reason == "queue-capacity"
        result = w.drain(5)
        assert len(result) == 1 and result[0].status == "candidate"
        assert w.submit(request(2)) is None
        assert len(w.close(5)) == 1
        assert w.peak_outstanding == 1
    finally:
        w.close(5)


def test_ineligible_short_group_never_initializes_model():
    def forbidden():
        raise AssertionError("must not load a model")

    w = worker(forbidden)
    result = w.submit(request(milliseconds=100))
    assert result.status == "skipped" and result.reason == "group-too-short"
    assert not w.is_alive and w.close(0) == ()


@pytest.mark.parametrize("bad", [None, 12, [], b"not text"])
def test_non_text_decoder_return_becomes_sanitized_error(bad):
    w = worker(lambda: lambda _: bad)
    w.submit(request())
    result = w.close(5)[0]
    assert result.status == "error" and result.text is None
    assert result.reason == "decode-failed:TypeError"


@pytest.mark.parametrize(
    "text,status,reason",
    [
        ("", "empty", "empty-decode"),
        (" \n", "empty", "empty-decode"),
        ("a" * 101, "error", "text-limit"),
        (" ええ、買わない。 ", "candidate", None),
    ],
)
def test_empty_large_and_verbatim_text(text, status, reason):
    w = worker(lambda: lambda _: text)
    w.submit(request())
    result = w.close(5)[0]
    assert result.status == status and result.reason == reason
    assert result.text == (text if status == "candidate" else None)


def test_exception_message_is_not_published_and_worker_keeps_running():
    count = 0

    def decode(_):
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("PRIVATE_PATH PRIVATE_SPEECH")
        return "二番目"

    w = worker(lambda: decode)
    w.submit(request(1))
    w.submit(request(2))
    result = w.close(5)
    assert [e.status for e in result] == ["error", "candidate"]
    assert "PRIVATE" not in json.dumps([e.as_dict() for e in result])


def test_failed_factory_runs_once_per_epoch_without_retry_storm():
    calls = []

    def factory():
        calls.append(1)
        raise FileNotFoundError("PRIVATE_PATH")

    w = worker(factory)
    w.submit(request(1))
    w.submit(request(2))
    result = w.close(5)
    assert calls == [1]
    assert all(e.reason == "decoder-initialization:FileNotFoundError" for e in result)
    assert "PRIVATE" not in json.dumps([e.as_dict() for e in result])


def test_blocked_refine_does_not_block_fast_finals_and_close_reports_timeout():
    entered, release = threading.Event(), threading.Event()

    def decode(_):
        entered.set()
        assert release.wait(5)
        return "遅い再認識"

    rt = runtime(lambda: decode, max_outstanding_groups=1)
    try:
        events = utterance(rt)
        events += idle(rt)
        assert entered.wait(5)
        first = rt.session.finals[0]
        events += utterance(rt, 2)
        assert len(rt.session.finals) == 2
        assert rt.session.finals[0] == first
        events += idle(rt)
        assert any(e.reason == "queue-capacity" for e in refinements(events))
        ended = rt.close(timeout_seconds=0)
        assert rt.closed and rt.worker_alive
        assert any(e.reason == "shutdown-timeout" for e in refinements(ended))
        assert rt.session.finals[0] == first
        assert rt.close(timeout_seconds=0) == ()
    finally:
        release.set()
        rt._worker.close(5)
    assert not rt.worker_alive and rt.poll() == ()


def test_grouped_audio_exactly_matches_continuous_stream_with_overlapping_preroll():
    calls = []

    def decode(item):
        calls.append(item)
        return "ええ、買わない。"

    rt = runtime(lambda: decode)
    raw = []
    events = []
    for value, speech in [
        (0, False),
        (1, True),
        (1, True),
        (1, True),
        (0, False),
        (2, True),
        (2, True),
        (0, False),
    ]:
        raw.append(pcm(value=value))
        events.extend(rt.feed_pcm16(raw[-1], speech=speech))
    parents_before = rt.session.finals
    events.extend(rt.close(timeout_seconds=5))
    result = refinements(events)
    assert len(result) == len(calls) == 1
    req = result[0].request
    assert len(req.parents) == 2
    assert req.parents[0].end_sample > req.parents[1].start_sample
    expected = b"".join(raw)[req.start_sample * 2 : req.end_sample * 2]
    assert calls[0].pcm16le == req.pcm16le == expected
    assert sum(len(p.pcm16le) for p in parents_before) > len(expected)
    assert rt.session.finals == parents_before
    assert [p.final_digest for p in req.parents] == [p.final_digest for p in parents_before]
    assert all(p.observed_text == "ええ、買わない、買わない。" for p in req.parents)
    assert result[0].text == "ええ、買わない。"  # A shorter proposal, NEVER an overwrite.
    payload = result[0].as_dict()
    assert payload["automaticallyApplied"] is False
    assert payload["independentEvidence"] is False
    assert "confidence" not in payload and "pcm16le" not in payload


def test_eof_publishes_final_before_wait_and_only_forces_group_once():
    rt = runtime(lambda: lambda _: "再認識")
    rt.feed_pcm16(pcm(100), speech=True)
    rt.feed_pcm16(pcm(100), speech=True)
    events = rt.flush()
    assert isinstance(events[0], RealtimeEvent) and events[0].kind == "final"
    events += rt.flush()
    events += rt.close(timeout_seconds=5)
    result = refinements(events)
    assert len(result) == 1 and result[0].request.trigger == "eof"
    assert result[0].request.pcm16le == pcm(200)
    with pytest.raises(RuntimeError):
        rt.feed_pcm16(pcm(), speech=True)


def test_no_idle_group_during_active_speech_and_long_stream_survives_history_rollover():
    rt = runtime(lambda: lambda _: "再認識")
    events = []
    for i in range(120):
        events.extend(rt.feed_pcm16(pcm(value=i), speech=True))
        assert rt.scheduler.history.retained_samples <= rt.scheduler.config.history_keep_samples
    events.extend(rt.close(timeout_seconds=5))
    groups = refinements(events)
    assert groups and all(e.request.trigger != "idle-gap" for e in groups)
    finals = [e for e in events if isinstance(e, RealtimeEvent) and e.kind == "final"]
    assert sum(len(e.request.parents) for e in groups) == len(finals)
    assert all(len(e.request.pcm16le) <= 2000 * 16 * 2 for e in groups)
    assert rt.peak_pending_groups <= 2


@pytest.mark.parametrize("flush", [False, True])
def test_reset_isolates_pending_groups_and_recreates_decoder_on_owner_thread(flush):
    creations = []

    def factory():
        creations.append(threading.get_ident())
        return lambda _: "候補"

    rt = runtime(factory)
    events = utterance(rt)
    old_id = rt.session.session_id
    events += list(rt.reset(flush_pending=flush, timeout_seconds=5))
    old = refinements(events)
    assert len(old) == 1
    assert old[0].status == ("candidate" if flush else "skipped")
    assert old[0].request.session_id == old_id
    assert rt.session.session_id != old_id
    assert rt.scheduler.history.end_sample == rt.session.sample_cursor == 0
    assert rt.session.finals == ()
    new_events = utterance(rt) + list(rt.close(timeout_seconds=5))
    assert all(e.request.session_id != old_id for e in refinements(new_events))
    assert len(creations) == (2 if flush else 1)
    assert len(set(creations)) == 1


def test_reset_drops_inflight_old_result_and_counts_it_toward_capacity():
    entered, release = threading.Event(), threading.Event()

    def decode(_):
        entered.set()
        assert release.wait(5)
        return "旧セッション"

    rt = runtime(lambda: decode, max_outstanding_groups=1)
    try:
        utterance(rt)
        idle(rt)
        assert entered.wait(5)
        old_id = rt.session.session_id
        receipts = rt.reset(flush_pending=True, timeout_seconds=0)
        assert any(e.reason == "reset-timeout" for e in refinements(receipts))
        assert rt.pending_groups == 1
        events = utterance(rt) + list(rt.flush())
        assert any(e.reason == "queue-capacity" for e in refinements(events))
        assert all(e.request.session_id != old_id for e in refinements(events))
        release.set()
        assert refinements(rt.close(timeout_seconds=5)) == []
        assert not rt.worker_alive
    finally:
        release.set()
        rt._worker.close(5)


@pytest.mark.parametrize(
    "pcm_value,speech,error",
    [
        (b"", True, ValueError),
        (b"x", True, ValueError),
        (bytearray(b"xx"), True, TypeError),
        (pcm(101), True, ValueError),
        (pcm(), 1, TypeError),
        (pcm(), None, TypeError),
    ],
)
def test_invalid_input_does_not_advance_either_timeline(pcm_value, speech, error):
    rt = runtime(lambda: lambda _: "候補")
    try:
        with pytest.raises(error):
            rt.feed_pcm16(pcm_value, speech=speech)
        assert rt.session.sample_cursor == rt.scheduler.history.end_sample == 0
    finally:
        rt.close(timeout_seconds=5)


@pytest.mark.parametrize("value", [-1, math.inf, math.nan, True, "5"])
def test_bad_timeout_does_not_close_or_reset_session(value):
    rt = runtime(lambda: lambda _: "候補")
    try:
        with pytest.raises((ValueError, TypeError)):
            rt.close(timeout_seconds=value)
        assert not rt.closed
        with pytest.raises((ValueError, TypeError)):
            rt.reset(flush_pending=True, timeout_seconds=value)
        assert rt.session.session_id == "test-session"
    finally:
        rt.close(timeout_seconds=5)


@pytest.mark.parametrize(
    "field,value", [("final_digest", "b" * 64), ("text", "改変"), ("audio_sha256", "c" * 64)]
)
def test_tampered_parent_proof_rejected_before_registration(field, value):
    rt = runtime(lambda: lambda _: "候補")
    data = pcm(300)
    try:
        rt.scheduler.feed_pcm16(data, speech=True)
        # Inject at the public binding seam; do not modify the persisted original event.
        for _ in range(3):
            rt.session.feed_pcm16(pcm(), speech=True)
        events = rt.session.flush()
        original = rt.session.finals[0]
        if field == "audio_sha256":
            tampered = replace(events[0], audio_sha256=value)
            with pytest.raises(ValueError, match="binding"):
                rt._register((tampered,))
        else:
            rt.session._history[-1] = replace(original, **{field: value})
            with pytest.raises(ValueError, match="binding"):
                rt._register(events)
        assert rt.scheduler.pending == ()
    finally:
        rt.abort()


def test_tampered_continuous_pcm_is_rejected_even_with_valid_parent_digest():
    rt = runtime(lambda: lambda _: "候補")
    try:
        rt.scheduler.feed_pcm16(pcm(300, 2), speech=True)
        for _ in range(3):
            rt.session.feed_pcm16(pcm(), speech=True)
        with pytest.raises(ValueError, match="continuous group PCM"):
            rt._register(rt.session.flush())
    finally:
        rt.abort()


def test_outcome_digest_excludes_latency_but_binds_candidate_and_request():
    result = GroupedRefineOutcome(request(), "decoder", "candidate", "候補", None, 1.0)
    assert (
        result.as_dict()["evidenceDigest"]
        == replace(result, decode_duration_ms=2.0).as_dict()["evidenceDigest"]
    )
    assert (
        result.as_dict()["evidenceDigest"]
        != replace(result, text="違う候補").as_dict()["evidenceDigest"]
    )
    with pytest.raises(FrozenInstanceError):
        result.text = "変更"


def test_abort_does_not_retry_failed_fast_decoder():
    calls = []

    def fail(_):
        calls.append(1)
        raise RuntimeError("decode")

    rt = runtime(lambda: lambda _: "候補", decoder=fail)
    with pytest.raises(RuntimeError):
        rt.feed_pcm16(pcm(100), speech=True)
        rt.feed_pcm16(pcm(100), speech=True)
    rt.abort()
    assert calls == [1] and rt.closed and not rt.worker_alive


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_queue_capacity(capacity):
    with pytest.raises((ValueError, TypeError)):
        worker(lambda: lambda _: "候補", capacity=capacity)
