"""Worker timing is operational metadata, separate from speech evidence identity."""

import hashlib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from semantic_asr import realtime_refine_runtime as runtime
from semantic_asr.realtime_refine import GroupedRefineScheduler, RefineParentFinal
from semantic_asr.realtime_refine_runtime import GroupedRefineOutcome, _FifoRefiner

TIMING_FIELDS = (
    "decode_duration_ms",
    "queue_wait_ms",
    "decoder_factory_duration_ms",
    "completion_after_submit_ms",
)


def request(number):
    data = b"\x01\x00" * (500 * 16)
    scheduler = GroupedRefineScheduler(session_id=f"timing-session-{number}")
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


def test_operational_timing_preserves_preexisting_evidence_digest():
    original = GroupedRefineOutcome(request(1), "pinned-test-decoder", "candidate", text="候補")
    payload = original.as_dict()
    # Captured from ddb0cac before the additional timing metadata existed.
    expected_digest = "8ef178104004be39dd473dd12bc0725f09f02a1a7bc18966186138905ff31fc4"
    assert payload["evidenceDigest"] == expected_digest
    assert payload["queueWaitMs"] is None
    assert payload["decoderFactoryDurationMs"] is None
    assert payload["completionAfterSubmitMs"] is None
    timed = replace(
        original,
        decode_duration_ms=20.0,
        queue_wait_ms=30.0,
        decoder_factory_duration_ms=5.0,
        completion_after_submit_ms=50.0,
    ).as_dict()
    assert timed["decodeDurationMs"] == 20.0
    assert timed["queueWaitMs"] == 30.0
    assert timed["decoderFactoryDurationMs"] == 5.0
    assert timed["completionAfterSubmitMs"] == 50.0
    assert timed["evidenceDigest"] == payload["evidenceDigest"]
    assert timed["requestDigest"] == payload["requestDigest"]
    assert timed["request"] == payload["request"]
    # Candidate text remains evidence even though operational timings do not.
    assert replace(original, text="別の候補").as_dict()["evidenceDigest"] != expected_digest


@pytest.mark.parametrize("field", TIMING_FIELDS)
@pytest.mark.parametrize("value", [True, False, "1", -1.0, float("nan"), float("inf")])
def test_timing_metadata_rejects_non_numeric_or_non_finite_values(field, value):
    with pytest.raises((TypeError, ValueError)):
        GroupedRefineOutcome(
            request(1), "pinned-test-decoder", "candidate", text="候補", **{field: value}
        )


@pytest.mark.parametrize("field", TIMING_FIELDS)
def test_zero_timing_is_a_measured_value(field):
    outcome = GroupedRefineOutcome(
        request(1), "pinned-test-decoder", "candidate", text="候補", **{field: 0}
    )
    assert getattr(outcome, field) == 0


class ManualClock:
    def __init__(self, value):
        self._value = value
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            assert value >= self._value
            self._value = value


def test_worker_separates_queue_wait_cold_factory_and_warm_service(monkeypatch):
    clock = ManualClock(10.0)
    monkeypatch.setattr(
        runtime, "time", SimpleNamespace(perf_counter=clock, monotonic=time.monotonic)
    )
    factory_entered, release_factory = threading.Event(), threading.Event()
    decode_entered, release_decode = threading.Event(), threading.Event()
    first, second = request(1), request(2)
    factory_calls = []
    decoded = []

    def decode(item):
        decoded.append(item.utterance_id)
        if item.utterance_id == first.group_id:
            decode_entered.set()
            assert release_decode.wait(5)
        else:
            assert item.utterance_id == second.group_id
            clock.set(15.0)
        return "候補"

    def factory():
        factory_calls.append(threading.get_ident())
        factory_entered.set()
        assert release_factory.wait(5)
        return decode

    worker = _FifoRefiner(factory, decoder_id="pinned-test-decoder", capacity=2, max_text_chars=100)
    try:
        assert worker.submit(first) is None
        assert factory_entered.wait(5)
        clock.set(11.0)
        release_factory.set()
        assert decode_entered.wait(5)
        clock.set(12.0)
        assert worker.submit(second) is None
        # The second group stays queued until the first decoder is released.
        with worker._condition:
            assert len(worker._queue) == 1
            assert worker._active[1] == first.group_id
        clock.set(14.0)
        release_decode.set()
        outcomes = worker.close(5)
        assert not worker.is_alive
        assert [item.request.group_id for item in outcomes] == [first.group_id, second.group_id]
        assert decoded == [first.group_id, second.group_id]
        assert len(factory_calls) == 1
        cold, warm = outcomes
        assert cold.status == warm.status == "candidate"
        assert cold.queue_wait_ms == pytest.approx(0.0)
        assert cold.decoder_factory_duration_ms == pytest.approx(1000.0)
        # Preserve decodeDurationMs's former factory-plus-decode service meaning.
        assert cold.decode_duration_ms == pytest.approx(4000.0)
        assert cold.completion_after_submit_ms == pytest.approx(4000.0)
        assert warm.queue_wait_ms == pytest.approx(2000.0)
        assert warm.decoder_factory_duration_ms is None
        assert warm.decode_duration_ms == pytest.approx(1000.0)
        assert warm.completion_after_submit_ms == pytest.approx(3000.0)
        for item in outcomes:
            assert item.completion_after_submit_ms == pytest.approx(
                item.queue_wait_ms + item.decode_duration_ms
            )
    finally:
        release_factory.set()
        release_decode.set()
        worker.discard("test-cleanup")
        worker.close(5)


@pytest.mark.parametrize("failure_site", ["factory", "decode"])
@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit])
def test_terminal_worker_receipts_have_no_fabricated_timing(failure_site, fatal_type):
    entered, release = threading.Event(), threading.Event()
    factory_calls = []

    def interrupt():
        entered.set()
        assert release.wait(5)
        raise fatal_type("private speech and path")

    def factory():
        factory_calls.append(threading.get_ident())
        if failure_site == "factory":
            interrupt()

        def decode(_):
            interrupt()

        return decode

    worker = _FifoRefiner(factory, decoder_id="pinned-test-decoder", capacity=2, max_text_chars=100)
    first, second, later = request(1), request(2), request(3)
    try:
        assert worker.submit(first) is None
        assert entered.wait(5)
        assert worker.submit(second) is None
        release.set()
        worker._thread.join(5)
        assert not worker.is_alive
        late_receipt = worker.submit(later)
        assert late_receipt is not None
        outcomes = worker.close(5)
        assert [item.request.group_id for item in outcomes] == [first.group_id, second.group_id]
        assert len(factory_calls) == 1
        for item in (*outcomes, late_receipt):
            assert item.status == "error" and item.text is None
            assert item.reason == f"worker-terminated:{fatal_type.__name__}"
            assert all(getattr(item, field) is None for field in TIMING_FIELDS)
            assert all(
                item.as_dict()[field] is None
                for field in (
                    "decodeDurationMs",
                    "queueWaitMs",
                    "decoderFactoryDurationMs",
                    "completionAfterSubmitMs",
                )
            )
    finally:
        release.set()
        worker.discard("test-cleanup")
        worker.close(5)
