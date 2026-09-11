"""Bounded single-consumer worker for grouped realtime refinement.

The worker is deliberately specific to :mod:`semantic_asr.realtime_refine`.  It
keeps expensive second-pass decoding off the fast-final ingestion path without
introducing a generic executor/plugin framework.

Design goals, informed by Hayamimi's live Refiner lifecycle:

* exactly one consumer preserves chronological FIFO output order;
* close puts a stop marker at the back of the queue, so older accepted work
  drains before the thread exits;
* close is idempotent and bounded by a caller-supplied timeout;
* decoder exceptions become explicit failed results and never kill the worker;
* task and result queues are finite, so a stalled refiner cannot grow memory
  without bound;
* submit never blocks the realtime hot path: queue pressure is an explicit,
  evidence-producing rejection rather than hidden latency;
* wait_idle() is a reset/barrier primitive that prevents an old session's work
  from racing a new session.

This module is model-free.  A later integration supplies a warm Semantic ASR
transcriber callback; no model import, download, or profile promotion happens here.
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .realtime_refine import GroupedRefineRequest

SubmissionStatus = Literal["queued", "skipped", "rejected"]
ResultStatus = Literal["success", "failed"]

_STOP = object()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _finite_timeout(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class GroupedRefineDecodedText:
    """Stable text/provenance returned by the caller-owned second pass."""

    observed_text: str
    normalized_text: str
    decoder_id: str
    provenance_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observed_text, str):
            raise TypeError("observed_text must be str")
        if not isinstance(self.normalized_text, str):
            raise TypeError("normalized_text must be str")
        if not isinstance(self.decoder_id, str) or not self.decoder_id:
            raise ValueError("decoder_id is required")
        if self.provenance_digest is not None:
            if (
                not isinstance(self.provenance_digest, str)
                or len(self.provenance_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in self.provenance_digest)
            ):
                raise ValueError("provenance_digest must be lowercase 64-hex SHA-256")


@dataclass(frozen=True, slots=True)
class GroupedRefineSubmission:
    """Immediate, non-blocking receipt returned to the realtime hot path."""

    status: SubmissionStatus
    request_evidence_digest: str
    session_id: str
    group_id: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"queued", "skipped", "rejected"}:
            raise ValueError(f"unsupported submission status: {self.status!r}")
        if not self.request_evidence_digest:
            raise ValueError("request_evidence_digest is required")
        if not self.session_id or not self.group_id:
            raise ValueError("session_id and group_id are required")
        if self.status == "queued" and self.reason is not None:
            raise ValueError("queued submissions cannot have a reason")
        if self.status != "queued" and not self.reason:
            raise ValueError("skipped/rejected submissions require a reason")

    @property
    def evidence_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": "semantic-asr-grouped-refine-submission-v1",
                "status": self.status,
                "requestEvidenceDigest": self.request_evidence_digest,
                "sessionId": self.session_id,
                "groupId": self.group_id,
                "reason": self.reason,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "semantic-asr-grouped-refine-submission-v1",
            "status": self.status,
            "requestEvidenceDigest": self.request_evidence_digest,
            "sessionId": self.session_id,
            "groupId": self.group_id,
            "reason": self.reason,
            "evidenceDigest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class GroupedRefineWorkerResult:
    """One completed second-pass outcome bound to its immutable request."""

    status: ResultStatus
    request_sequence: int
    request_evidence_digest: str
    session_id: str
    group_id: str
    parent_final_digests: tuple[str, ...]
    group_audio_sha256: str
    observed_text: str | None = None
    normalized_text: str | None = None
    decoder_id: str | None = None
    provenance_digest: str | None = None
    error_code: str | None = None
    decode_duration_ms: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed"}:
            raise ValueError(f"unsupported result status: {self.status!r}")
        _strict_int(self.request_sequence, name="request_sequence", minimum=1)
        if not self.request_evidence_digest:
            raise ValueError("request_evidence_digest is required")
        if not self.session_id or not self.group_id:
            raise ValueError("session_id and group_id are required")
        if not self.parent_final_digests:
            raise ValueError("parent_final_digests cannot be empty")
        if not self.group_audio_sha256:
            raise ValueError("group_audio_sha256 is required")
        if self.status == "success":
            if self.observed_text is None or self.normalized_text is None or not self.decoder_id:
                raise ValueError("successful results require observed/normalized text and decoder_id")
            if self.error_code is not None:
                raise ValueError("successful results cannot have error_code")
        else:
            if not self.error_code:
                raise ValueError("failed results require error_code")
            if any(
                value is not None
                for value in (
                    self.observed_text,
                    self.normalized_text,
                    self.decoder_id,
                    self.provenance_digest,
                )
            ):
                raise ValueError("failed results cannot carry decoded text/provenance")
        if self.provenance_digest is not None:
            if (
                len(self.provenance_digest) != 64
                or any(ch not in "0123456789abcdef" for ch in self.provenance_digest)
            ):
                raise ValueError("provenance_digest must be lowercase 64-hex SHA-256")
        if self.decode_duration_ms is not None:
            if (
                isinstance(self.decode_duration_ms, bool)
                or not isinstance(self.decode_duration_ms, (int, float))
                or not math.isfinite(float(self.decode_duration_ms))
                or self.decode_duration_ms < 0
            ):
                raise ValueError("decode_duration_ms must be finite and non-negative")

    @property
    def evidence_digest(self) -> str:
        # Runtime duration is intentionally excluded from the immutable evidence hash.
        return _canonical_sha256(
            {
                "schema": "semantic-asr-grouped-refine-result-v1",
                "status": self.status,
                "requestSequence": self.request_sequence,
                "requestEvidenceDigest": self.request_evidence_digest,
                "sessionId": self.session_id,
                "groupId": self.group_id,
                "parentFinalDigests": list(self.parent_final_digests),
                "groupAudioSha256": self.group_audio_sha256,
                "observedText": self.observed_text,
                "normalizedText": self.normalized_text,
                "decoderId": self.decoder_id,
                "provenanceDigest": self.provenance_digest,
                "errorCode": self.error_code,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "semantic-asr-grouped-refine-result-v1",
            "status": self.status,
            "requestSequence": self.request_sequence,
            "requestEvidenceDigest": self.request_evidence_digest,
            "sessionId": self.session_id,
            "groupId": self.group_id,
            "parentFinalDigests": list(self.parent_final_digests),
            "groupAudioSha256": self.group_audio_sha256,
            "observedText": self.observed_text,
            "normalizedText": self.normalized_text,
            "decoderId": self.decoder_id,
            "provenanceDigest": self.provenance_digest,
            "errorCode": self.error_code,
            "decodeDurationMs": self.decode_duration_ms,
            "evidenceDigest": self.evidence_digest,
        }


Decoder = Callable[[GroupedRefineRequest], GroupedRefineDecodedText]


class GroupedRefineWorker:
    """Finite single-consumer FIFO for expensive grouped second-pass decoding."""

    def __init__(
        self,
        decoder: Decoder,
        *,
        max_pending: int = 4,
        max_results: int = 8,
        thread_name: str = "semantic-asr-grouped-refine",
    ) -> None:
        if not callable(decoder):
            raise TypeError("decoder must be callable")
        _strict_int(max_pending, name="max_pending", minimum=1)
        _strict_int(max_results, name="max_results", minimum=1)
        if not isinstance(thread_name, str) or not thread_name:
            raise ValueError("thread_name is required")

        self.decoder = decoder
        self.max_pending = max_pending
        self.max_results = max_results
        self._tasks: queue.Queue[GroupedRefineRequest | object] = queue.Queue(maxsize=max_pending)
        self._results: queue.Queue[GroupedRefineWorkerResult] = queue.Queue(maxsize=max_results)
        self._close_lock = threading.Lock()
        self._state = threading.Condition()
        self._closed = False
        self._stop_enqueued = False
        self._accepted = 0
        self._completed = 0
        self._thread = threading.Thread(target=self._worker_loop, name=thread_name, daemon=True)
        self._thread.start()

    @property
    def closed(self) -> bool:
        with self._close_lock:
            return self._closed

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def pending_count(self) -> int:
        return self._tasks.qsize()

    @property
    def result_count(self) -> int:
        return self._results.qsize()

    @property
    def accepted_count(self) -> int:
        with self._state:
            return self._accepted

    @property
    def completed_count(self) -> int:
        with self._state:
            return self._completed

    def submit(self, request: GroupedRefineRequest) -> GroupedRefineSubmission:
        """Queue one eligible group without ever blocking the hot path."""

        if not isinstance(request, GroupedRefineRequest):
            raise TypeError("request must be GroupedRefineRequest")
        digest = request.evidence_digest
        if not request.eligible_for_decode:
            return GroupedRefineSubmission(
                status="skipped",
                request_evidence_digest=digest,
                session_id=request.session_id,
                group_id=request.group_id,
                reason=request.skip_reason or "ineligible",
            )

        with self._close_lock:
            if self._closed:
                return GroupedRefineSubmission(
                    status="rejected",
                    request_evidence_digest=digest,
                    session_id=request.session_id,
                    group_id=request.group_id,
                    reason="worker-closed",
                )
            try:
                self._tasks.put_nowait(request)
            except queue.Full:
                return GroupedRefineSubmission(
                    status="rejected",
                    request_evidence_digest=digest,
                    session_id=request.session_id,
                    group_id=request.group_id,
                    reason="queue-full",
                )
            with self._state:
                self._accepted += 1
                self._state.notify_all()

        return GroupedRefineSubmission(
            status="queued",
            request_evidence_digest=digest,
            session_id=request.session_id,
            group_id=request.group_id,
        )

    def poll_result(self, *, timeout: float = 0.0) -> GroupedRefineWorkerResult | None:
        """Return the oldest completed result, or None when no result is ready."""

        timeout = _finite_timeout(timeout, name="timeout")
        try:
            if timeout == 0:
                return self._results.get_nowait()
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_results(self) -> tuple[GroupedRefineWorkerResult, ...]:
        output: list[GroupedRefineWorkerResult] = []
        while True:
            item = self.poll_result()
            if item is None:
                return tuple(output)
            output.append(item)

    def wait_idle(self, *, timeout: float) -> bool:
        """Wait until every accepted task has produced a result.

        This is the barrier callers should use before resetting scheduler/session
        state.  It returns False on timeout rather than hiding an unbounded wait.
        """

        timeout = _finite_timeout(timeout, name="timeout")
        deadline = time.monotonic() + timeout
        with self._state:
            while self._completed < self._accepted:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state.wait(remaining)
            return True

    def close(self, *, timeout: float = 30.0) -> None:
        """Drain accepted FIFO work, stop the worker and release its thread.

        The stop marker is enqueued after all already-accepted requests.  If result
        backpressure or a stuck decoder prevents shutdown within ``timeout``, a
        TimeoutError is raised instead of waiting forever.  A later close() call may
        be used to finish joining after the blockage is cleared.
        """

        timeout = _finite_timeout(timeout, name="timeout")
        deadline = time.monotonic() + timeout
        with self._close_lock:
            self._closed = True
            if not self._stop_enqueued:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    self._tasks.put(_STOP, timeout=remaining)
                except queue.Full as exc:
                    raise TimeoutError("timed out enqueueing grouped refine stop marker") from exc
                self._stop_enqueued = True

        remaining = max(0.0, deadline - time.monotonic())
        self._thread.join(remaining)
        if self._thread.is_alive():
            raise TimeoutError("timed out waiting for grouped refine worker to stop")

    def __enter__(self) -> "GroupedRefineWorker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _worker_loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is _STOP:
                self._tasks.task_done()
                return
            assert isinstance(task, GroupedRefineRequest)
            result = self._decode(task)
            self._results.put(result)
            with self._state:
                self._completed += 1
                self._state.notify_all()
            self._tasks.task_done()

    def _decode(self, request: GroupedRefineRequest) -> GroupedRefineWorkerResult:
        started = time.perf_counter_ns()
        try:
            decoded = self.decoder(request)
            if not isinstance(decoded, GroupedRefineDecodedText):
                raise TypeError("decoder must return GroupedRefineDecodedText")
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            return GroupedRefineWorkerResult(
                status="success",
                request_sequence=request.sequence,
                request_evidence_digest=request.evidence_digest,
                session_id=request.session_id,
                group_id=request.group_id,
                parent_final_digests=tuple(parent.final_digest for parent in request.parents),
                group_audio_sha256=request.group_audio_sha256,
                observed_text=decoded.observed_text,
                normalized_text=decoded.normalized_text,
                decoder_id=decoded.decoder_id,
                provenance_digest=decoded.provenance_digest,
                decode_duration_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            return GroupedRefineWorkerResult(
                status="failed",
                request_sequence=request.sequence,
                request_evidence_digest=request.evidence_digest,
                session_id=request.session_id,
                group_id=request.group_id,
                parent_final_digests=tuple(parent.final_digest for parent in request.parents),
                group_audio_sha256=request.group_audio_sha256,
                error_code=type(exc).__name__,
                decode_duration_ms=elapsed_ms,
            )
