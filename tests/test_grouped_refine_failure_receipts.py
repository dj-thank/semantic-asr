"""Failed decode paths must preserve one audit receipt for every closed group."""

import hashlib
import json
import threading

import pytest

from semantic_asr.realtime_reazon import RealtimeReazonConfig
from semantic_asr.realtime_refine import (
    GroupedRefineConfig,
    GroupedRefineScheduler,
    RefineParentFinal,
)
from semantic_asr.realtime_refine_runtime import GroupedRealtimeReazon, _FifoRefiner


def pcm(milliseconds=100):
    return b"\x01\x00" * (milliseconds * 16)


def request(number):
    data = pcm(500)
    scheduler = GroupedRefineScheduler(session_id=f"failure-session-{number}")
    scheduler.feed_pcm16(data, speech=True)
    scheduler.add_final(
        RefineParentFinal(
            utterance_id=f"u{number}",
            final_digest=hashlib.sha256(f"parent-{number}".encode()).hexdigest(),
            audio_sha256=hashlib.sha256(data).hexdigest(),
            start_sample=0,
            end_sample=len(data) // 2,
            observed_text="ええ、買わない、買わない。",
        )
    )
    return scheduler.force("eof")[0]


@pytest.mark.parametrize("failure_mode,max_speech_ms", [("partial", 600), ("final", 500)])
@pytest.mark.parametrize("cleanup", ["abort", "close"])
def test_failed_fast_decode_keeps_already_closed_group_for_cleanup(
    failure_mode, max_speech_ms, cleanup
):
    fail = False
    primary = RuntimeError("primary fast decoder failure")
    decoder_calls = []

    def decode(item):
        decoder_calls.append(item.mode)
        if fail:
            raise primary
        return "ええ、買わない、買わない。"

    rt = GroupedRealtimeReazon(
        decode,
        refine_decoder_factory=lambda: lambda _: "候補",
        refine_decoder_id="pinned-test-decoder",
        config=RealtimeReazonConfig(
            partial_interval_ms=100,
            min_silence_ms=100,
            max_speech_ms=max_speech_ms,
            preroll_ms=100,
            max_chunk_ms=100,
        ),
        group_config=GroupedRefineConfig(
            idle_gap_ms=300,
            max_group_ms=700,
            min_group_ms=200,
            history_keep_ms=1000,
        ),
        session_id="fast-failure-session",
    )
    try:
        # The first final awaits grouping while the next utterance is active.
        for speech in [True, True, True, False, True, True, True]:
            rt.feed_pcm16(pcm(), speech=speech)
        original_finals = rt.session.finals
        assert len(original_finals) == len(rt.scheduler.pending) == 1
        assert rt.scheduler.history.end_sample == 700 * 16
        fail = True
        # This chunk closes old parents before invoking the failing fast decoder.
        with pytest.raises(RuntimeError) as caught:
            rt.feed_pcm16(pcm(), speech=True)
        assert caught.value is primary
        assert decoder_calls[-1] == failure_mode
        assert rt.session.finals == original_finals
        calls_after_failure = len(decoder_calls)
        for operation in (
            lambda: rt.feed_pcm16(pcm(), speech=False),
            rt.flush,
            lambda: rt.reset(flush_pending=False),
        ):
            with pytest.raises(RuntimeError):
                operation()
            assert len(decoder_calls) == calls_after_failure
        receipts = getattr(rt, cleanup)()
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.status == "skipped" and receipt.text is None
        assert receipt.request.trigger == "max-duration"
        assert [parent.final_digest for parent in receipt.request.parents] == [
            original_finals[0].final_digest
        ]
        assert receipt.request.pcm16le == original_finals[0].pcm16le
        assert rt.closed and rt.pending_groups == 0
        assert rt.abort() == ()
        assert rt.close() == ()
        assert len(decoder_calls) == calls_after_failure
    finally:
        rt.abort()
        rt._worker.close(5)
    assert not rt.worker_alive


@pytest.mark.parametrize("failure_site", ["factory", "decode"])
@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit])
def test_worker_interruption_emits_each_outstanding_receipt_once(
    failure_site, fatal_type, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    private_message = "PRIVATE_PATH PRIVATE_SPEECH"
    unhandled = []
    factory_calls = []
    # Baseline worker termination must not leak test exception messages or create
    # an unrelated unhandled-thread warning instead of the audit assertion below.
    monkeypatch.setattr(threading, "excepthook", unhandled.append)

    def interrupt():
        entered.set()
        assert release.wait(5)
        raise fatal_type(private_message)

    def factory():
        factory_calls.append(threading.get_ident())
        if failure_site == "factory":
            interrupt()

        def decode(_):
            interrupt()

        return decode

    w = _FifoRefiner(factory, decoder_id="pinned-test-decoder", capacity=2, max_text_chars=100)
    first, second = request(1), request(2)
    try:
        assert w.submit(first) is None
        assert entered.wait(5)
        assert w.submit(second) is None
        release.set()
        w._thread.join(5)
        assert not w.is_alive
        later = w.submit(request(3))
        assert later is not None and later.status == "error" and later.text is None
        assert factory_calls and len(factory_calls) == 1
        receipts = w.close(5)
        assert not w.is_alive
        assert [item.request.group_id for item in receipts] == [first.group_id, second.group_id]
        assert all(item.status != "candidate" and item.text is None for item in receipts)
        assert all(item.reason for item in receipts)
        assert private_message not in json.dumps([item.as_dict() for item in receipts])
        assert w.pending_count == 0
        assert w.poll() == w.close(5) == ()
        assert unhandled == []
    finally:
        release.set()
        w.discard("test-cleanup")
        w.close(5)


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit])
def test_worker_interruption_preserves_ready_candidate_before_terminal_receipts(
    fatal_type, monkeypatch
):
    ready, interrupted, queued = request(1), request(2), request(3)
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(threading, "excepthook", lambda _: None)

    def decode(item):
        if item.utterance_id == ready.group_id:
            return " 既に完了した候補 "
        entered.set()
        assert release.wait(5)
        raise fatal_type("PRIVATE_PATH PRIVATE_SPEECH")

    w = _FifoRefiner(
        lambda: decode, decoder_id="pinned-test-decoder", capacity=3, max_text_chars=100
    )
    try:
        assert w.submit(ready) is None
        with w._condition:
            assert w._condition.wait_for(lambda: ready.group_id in w._ready, timeout=5)
        assert w.submit(interrupted) is None
        assert entered.wait(5)
        assert w.submit(queued) is None
        release.set()
        outcomes = w.close(5)
        assert [item.request.group_id for item in outcomes] == [
            ready.group_id,
            interrupted.group_id,
            queued.group_id,
        ]
        assert outcomes[0].status == "candidate" and outcomes[0].text == " 既に完了した候補 "
        assert all(item.status == "error" and item.text is None for item in outcomes[1:])
        assert w.pending_count == 0 and not w.is_alive
        assert w.poll() == w.close(5) == ()
    finally:
        release.set()
        w.discard("test-cleanup")
        w.close(5)
