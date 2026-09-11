import hashlib

import pytest

from semantic_asr.realtime_refine import GroupedRefineScheduler, RefineParentFinal

SR = 16_000


def pcm(milliseconds: int, *, value: int = 1) -> bytes:
    count = round(SR * milliseconds / 1000)
    return int(value).to_bytes(2, "little", signed=True) * count


def parent() -> RefineParentFinal:
    return RefineParentFinal(
        utterance_id="u1",
        final_digest=hashlib.sha256(b"final").hexdigest(),
        audio_sha256=hashlib.sha256(b"audio").hexdigest(),
        start_sample=0,
        end_sample=SR,
        observed_text="first",
    )


def test_active_speech_never_counts_toward_idle_gap():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(1_000), speech=True)
    scheduler.add_final(parent())

    for _ in range(3):
        assert scheduler.feed_pcm16(pcm(1_000), speech=True) == ()
    assert [item.utterance_id for item in scheduler.pending] == ["u1"]

    assert scheduler.feed_pcm16(pcm(1_999, value=0), speech=False) == ()
    events = scheduler.feed_pcm16(pcm(1, value=0), speech=False)
    assert len(events) == 1
    assert events[0].trigger == "idle-gap"


def test_new_speech_resets_accumulated_silence():
    scheduler = GroupedRefineScheduler(session_id="s")
    scheduler.feed_pcm16(pcm(1_000), speech=True)
    scheduler.add_final(parent())

    assert scheduler.feed_pcm16(pcm(1_500, value=0), speech=False) == ()
    assert scheduler.feed_pcm16(pcm(100), speech=True) == ()
    assert scheduler.feed_pcm16(pcm(1_999, value=0), speech=False) == ()
    events = scheduler.feed_pcm16(pcm(1, value=0), speech=False)
    assert len(events) == 1
    assert events[0].trigger == "idle-gap"


def test_activity_flag_fails_closed_on_non_bool():
    scheduler = GroupedRefineScheduler(session_id="s")
    with pytest.raises(TypeError, match="speech must be bool"):
        scheduler.feed_pcm16(pcm(10), speech=1)
    assert scheduler.history.end_sample == 0
