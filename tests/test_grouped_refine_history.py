import hashlib

import pytest

from semantic_asr.realtime_refine import (
    GroupedRefineConfig,
    GroupedRefineScheduler,
    RefineParentFinal,
)


def test_active_speech_closes_group_before_its_pcm_can_be_evicted():
    scheduler = GroupedRefineScheduler(
        config=GroupedRefineConfig(max_group_ms=3000, idle_gap_ms=1000, history_keep_ms=4000),
        session_id="s",
    )
    audio = b"\x01\x00" * 16000
    scheduler.feed_pcm16(audio, speech=True)
    scheduler.add_final(
        RefineParentFinal(
            utterance_id="s:u1",
            final_digest="a" * 64,
            audio_sha256=hashlib.sha256(audio).hexdigest(),
            start_sample=0,
            end_sample=16000,
            observed_text="えー、違います",
        )
    )
    closed = scheduler.feed_pcm16(audio * 3, speech=True)
    assert len(closed) == 1
    assert closed[0].trigger == "max-duration"
    assert closed[0].pcm16le == audio
    scheduler.feed_pcm16(audio * 2, speech=True)
    assert scheduler.force("eof") == ()


@pytest.mark.parametrize("invalid", [b"", b"x", bytearray(b"xx")])
def test_invalid_pcm_does_not_close_or_advance_pending_group(invalid):
    scheduler = GroupedRefineScheduler(session_id="s")
    audio = b"\x01\x00" * 16000
    scheduler.feed_pcm16(audio, speech=True)
    parent = RefineParentFinal("u1", "a" * 64, "b" * 64, 0, 16000, "first")
    scheduler.add_final(parent)
    with pytest.raises((TypeError, ValueError)):
        scheduler.feed_pcm16(invalid, speech=True)
    assert scheduler.pending == (parent,)
    assert scheduler.history.end_sample == 16000
