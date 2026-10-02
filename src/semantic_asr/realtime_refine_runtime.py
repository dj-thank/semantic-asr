"""Opt-in Hayamimi-style grouped re-decode, separate from immutable fast finals.

One lazy FIFO worker owns its decoder. Pending, running and unconsumed completed
work share one finite capacity; overload produces receipts, never a blocking put.
This produces unscored audio-derived candidates, not verified transcript edits.
A thread cannot interrupt a hung native decoder: shutdown is deadline-bounded,
reports discarded work, and never claims that a timed-out call was terminated.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .realtime_reazon import (
    Decoder,
    RealtimeDecodeInput,
    RealtimeEvent,
    RealtimeReazonConfig,
    RealtimeReazonSession,
)
from .realtime_refine import (
    GroupedRefineConfig,
    GroupedRefineRequest,
    GroupedRefineScheduler,
    RefineParentFinal,
)


def _timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout_seconds must be a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError("timeout_seconds must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class GroupedRefineOutcome:
    """A separate revision or control receipt; never a new first-pass final."""

    request: GroupedRefineRequest
    decoder_id: str
    status: str
    text: str | None = None
    reason: str | None = None
    decode_duration_ms: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, GroupedRefineRequest):
            raise TypeError("request must be GroupedRefineRequest")
        if not isinstance(self.decoder_id, str) or not self.decoder_id:
            raise ValueError("decoder_id is required")
        if self.status not in {"candidate", "skipped", "error", "empty"}:
            raise ValueError("unsupported grouped outcome status")
        if self.status == "candidate":
            if not isinstance(self.text, str) or not self.text.strip() or self.reason is not None:
                raise ValueError("candidate requires non-empty text and no failure reason")
        elif self.text is not None or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("non-candidate requires a reason and no text")
        if self.decode_duration_ms is not None:
            _timeout(self.decode_duration_ms)

    def as_dict(self) -> dict[str, object]:
        payload = {
            "schema": "semantic-asr-grouped-refine-outcome-v1",
            "kind": "group_refine" if self.status == "candidate" else "group_refine_warning",
            "requestDigest": self.request.evidence_digest,
            "request": self.request.as_dict(),
            "decoderId": self.decoder_id,
            "status": self.status,
            "candidateText": self.text,
            "reason": self.reason,
            "automaticallyApplied": False,
            "independentEvidence": False,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {
            **payload,
            "evidenceDigest": hashlib.sha256(encoded).hexdigest(),
            "decodeDurationMs": self.decode_duration_ms,
        }


class _FifoRefiner:
    """Single-consumer worker. The owner serializes submit/poll/reset/close."""

    def __init__(
        self,
        factory: Callable[[], Decoder],
        *,
        decoder_id: str,
        capacity: int,
        max_text_chars: int,
    ) -> None:
        if not callable(factory):
            raise TypeError("refine_decoder_factory must be callable")
        if not isinstance(decoder_id, str) or not decoder_id:
            raise ValueError("refine_decoder_id is required")
        for name, value in (("capacity", capacity), ("max_text_chars", max_text_chars)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        self._factory = factory
        self.decoder_id = decoder_id
        self.capacity = capacity
        self.max_text_chars = max_text_chars
        self._condition = threading.Condition()
        self._epoch = 0
        self._queue: deque[tuple[int, GroupedRefineRequest]] = deque()
        self._outstanding: dict[str, GroupedRefineRequest] = {}
        self._ready: dict[str, GroupedRefineOutcome] = {}
        self._active: tuple[int, str] | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._terminal_error: str | None = None
        self.peak_outstanding = 0

    def receipt(self, request: GroupedRefineRequest, reason: str) -> GroupedRefineOutcome:
        return GroupedRefineOutcome(request, self.decoder_id, "skipped", reason=reason)

    def _count(self) -> int:
        stale_active = self._active is not None and self._active[0] != self._epoch
        return len(self._outstanding) + int(stale_active)

    @property
    def pending_count(self) -> int:
        with self._condition:
            return self._count()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, request: GroupedRefineRequest) -> GroupedRefineOutcome | None:
        with self._condition:
            if self._closed:
                raise RuntimeError("refine worker is closed")
            if self._terminal_error is not None:
                return GroupedRefineOutcome(
                    request, self.decoder_id, "error", reason=self._terminal_error
                )
            if not request.eligible_for_decode:
                return self.receipt(request, request.skip_reason or "ineligible")
            if self._count() >= self.capacity:
                return self.receipt(request, "queue-capacity")
            if request.group_id in self._outstanding:
                raise ValueError("duplicate outstanding group")
            self._outstanding[request.group_id] = request
            self._queue.append((self._epoch, request))
            self.peak_outstanding = max(self.peak_outstanding, self._count())
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="semantic-asr-group-refine", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
        return None

    def _run(self) -> None:
        decoder: Decoder | None = None
        decoder_epoch = -1
        initialization_error: str | None = None
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closed:
                        if decoder_epoch != self._epoch:
                            decoder = None
                            initialization_error = None
                        self._condition.wait()
                    if not self._queue:
                        return
                    epoch, request = self._queue.popleft()
                    self._active = (epoch, request.group_id)
                started = time.perf_counter()
                text = None
                reason = None
                status = "candidate"
                try:
                    if decoder_epoch != epoch:
                        # Release any previous session's model on its owner thread.
                        decoder = None
                        decoder_epoch = epoch
                        initialization_error = None
                        try:
                            decoder = self._factory()
                            if not callable(decoder):
                                raise TypeError("decoder factory must return a callable")
                        except Exception as exc:
                            initialization_error = f"decoder-initialization:{type(exc).__name__}"
                    if initialization_error is not None:
                        status, reason = "error", initialization_error
                    else:
                        assert decoder is not None
                        # Audio only: parent text, references and dictionaries are not prompts.
                        text = decoder(
                            RealtimeDecodeInput(
                                session_id=request.session_id,
                                utterance_id=request.group_id,
                                mode="final",
                                pcm16le=request.pcm16le,
                                sample_rate=request.sample_rate,
                                start_sample=request.start_sample,
                                end_sample=request.end_sample,
                                audio_sha256=request.group_audio_sha256,
                            )
                        )
                        if not isinstance(text, str):
                            raise TypeError("refine decoder must return str")
                        if len(text) > self.max_text_chars:
                            status, reason, text = "error", "text-limit", None
                        elif not text.strip():
                            status, reason, text = "empty", "empty-decode", None
                except Exception as exc:
                    # Exception messages may contain paths or speech: emit only the class.
                    status, reason, text = "error", f"decode-failed:{type(exc).__name__}", None
                elapsed = (time.perf_counter() - started) * 1000
                outcome = GroupedRefineOutcome(
                    request, self.decoder_id, status, text, reason, elapsed
                )
                with self._condition:
                    if epoch == self._epoch and request.group_id in self._outstanding:
                        self._ready[request.group_id] = outcome
                    self._active = None
                    self._condition.notify_all()
                # Do not retain the previous group's PCM while the worker is idle.
                del request, outcome
        except BaseException as exc:
            # A callback can terminate this thread without raising Exception.
            # Account for every accepted group and keep the worker terminal;
            # retrying the factory would hide the failure or repeat side effects.
            with self._condition:
                self._terminal_error = f"worker-terminated:{type(exc).__name__}"
                self._queue.clear()
                for group_id, pending in self._outstanding.items():
                    if group_id not in self._ready:
                        self._ready[group_id] = GroupedRefineOutcome(
                            pending, self.decoder_id, "error", reason=self._terminal_error
                        )
                self._condition.notify_all()
        finally:
            decoder = None
            with self._condition:
                self._active = None
                self._condition.notify_all()

    def poll(self) -> tuple[GroupedRefineOutcome, ...]:
        output: list[GroupedRefineOutcome] = []
        with self._condition:
            while self._outstanding:
                first = next(iter(self._outstanding))
                if first not in self._ready:
                    break
                output.append(self._ready.pop(first))
                del self._outstanding[first]
        return tuple(output)

    def drain(self, timeout_seconds: float) -> tuple[GroupedRefineOutcome, ...]:
        deadline = time.monotonic() + _timeout(timeout_seconds)
        with self._condition:
            while len(self._ready) < len(self._outstanding):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
        return self.poll()

    def discard(self, reason: str) -> tuple[GroupedRefineOutcome, ...]:
        with self._condition:
            receipts = tuple(self.receipt(item, reason) for item in self._outstanding.values())
            self._epoch += 1
            self._queue.clear()
            self._ready.clear()
            self._outstanding.clear()
            self._condition.notify_all()
            return receipts

    def close(self, timeout_seconds: float) -> tuple[GroupedRefineOutcome, ...]:
        timeout_seconds = _timeout(timeout_seconds)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_seconds)
        output = self.poll()
        if self.is_alive:
            output += self.discard("shutdown-timeout")
        return output


class GroupedRealtimeReazon:
    """Serial PCM/VAD coordinator plus one bounded second-pass worker.

    The factory MUST create a separately owned decoder, not return the fast
    decoder. The WAV runner enforces this by constructing a second pinned adapter
    lazily inside the worker. Call flush() and emit its fast events before close()
    when final delivery must not wait for the bounded EOF drain.
    """

    def __init__(
        self,
        decoder: Decoder,
        *,
        refine_decoder_factory: Callable[[], Decoder],
        refine_decoder_id: str,
        config: RealtimeReazonConfig | None = None,
        group_config: GroupedRefineConfig | None = None,
        max_outstanding_groups: int = 2,
        max_text_chars: int = 16_000,
        session_id: str | None = None,
    ) -> None:
        config = config or RealtimeReazonConfig()
        group_config = group_config or GroupedRefineConfig()
        if group_config.sample_rate != config.sample_rate:
            raise ValueError("first-pass and grouped sample rates must match")
        # RealtimeReazonSession closes at a chunk boundary, so allow one chunk overshoot.
        if group_config.max_group_ms < config.max_speech_ms + config.max_chunk_ms:
            raise ValueError("max_group_ms must cover max_speech_ms plus one PCM chunk")
        if group_config.max_group_ms > 30_000:
            raise ValueError("grouped Reazon re-decode cannot exceed 30 seconds")
        self.session = RealtimeReazonSession(decoder, config=config, session_id=session_id)
        self.scheduler = GroupedRefineScheduler(
            config=group_config, session_id=self.session.session_id
        )
        self._worker = _FifoRefiner(
            refine_decoder_factory,
            decoder_id=refine_decoder_id,
            capacity=max_outstanding_groups,
            max_text_chars=max_text_chars,
        )
        self._ended = False
        self._closed = False
        self._failed = False
        self._failed_requests: list[GroupedRefineRequest] = []

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def worker_alive(self) -> bool:
        return self._worker.is_alive

    @property
    def pending_groups(self) -> int:
        return self._worker.pending_count

    @property
    def peak_pending_groups(self) -> int:
        return self._worker.peak_outstanding

    def _register(self, events: tuple[RealtimeEvent, ...]) -> list[GroupedRefineRequest]:
        requests: list[GroupedRefineRequest] = []
        finals = {item.utterance_id: item for item in self.session.finals}
        for event in events:
            if event.kind != "final":
                continue
            final = finals[event.utterance_id]
            if (
                event.session_id != self.scheduler.session_id
                or event.evidence_digest != final.final_digest
                or event.audio_sha256 != final.audio_sha256
                or event.text != final.text
            ):
                raise ValueError("first-pass final evidence binding mismatch")
            pcm = self.scheduler.history.read(final.start_sample, final.end_sample)
            if pcm != final.pcm16le or hashlib.sha256(pcm).hexdigest() != final.audio_sha256:
                raise ValueError("first-pass final does not match continuous group PCM")
            parent = RefineParentFinal(
                utterance_id=final.utterance_id,
                final_digest=final.final_digest,
                audio_sha256=final.audio_sha256,
                start_sample=final.start_sample,
                end_sample=final.end_sample,
                observed_text=final.text,
            )
            requests.extend(self.scheduler.add_final(parent))
        return requests

    def _submit(self, requests: list[GroupedRefineRequest]) -> list[GroupedRefineOutcome]:
        output = []
        for request in requests:
            receipt = self._worker.submit(request)
            if receipt is not None:
                output.append(receipt)
        output.extend(self.poll())
        return output

    def poll(self) -> tuple[GroupedRefineOutcome, ...]:
        return self._worker.poll()

    def feed_pcm16(
        self, pcm16le: bytes, *, speech: bool
    ) -> tuple[RealtimeEvent | GroupedRefineOutcome, ...]:
        if self._closed or self._ended:
            raise RuntimeError("cannot feed a closed or flushed session")
        if self._failed:
            raise RuntimeError("failed session requires abort or close")
        if not isinstance(speech, bool):
            raise TypeError("speech must be bool")
        if not isinstance(pcm16le, bytes):
            raise TypeError("pcm16le must be bytes")
        if not pcm16le or len(pcm16le) % 2:
            raise ValueError("PCM must contain non-empty whole int16 samples")
        if len(pcm16le) // 2 > self.session.config.max_chunk_samples:
            raise ValueError("PCM chunk exceeds max_chunk_ms")
        requests = list(
            self.scheduler.feed_pcm16(
                pcm16le, speech=speech, start_sample=self.session.sample_cursor
            )
        )
        try:
            events = self.session.feed_pcm16(pcm16le, speech=speech)
            requests.extend(self._register(events))
        except BaseException:
            # The scheduler may have closed parents before the fast decoder
            # failed. Keep those requests until abort can publish their receipts.
            self._failed_requests.extend(requests)
            self._failed = True
            raise
        return (*events, *self._submit(requests))

    def _finish(
        self, trigger: Literal["eof", "reset"]
    ) -> tuple[tuple[RealtimeEvent, ...], list[GroupedRefineRequest]]:
        if self._ended:
            return (), []
        if self._failed:
            raise RuntimeError("failed session requires abort or close")
        requests: list[GroupedRefineRequest] = []
        try:
            events = self.session.flush()
            requests.extend(self._register(events))
            requests.extend(self.scheduler.force(trigger))
        except BaseException:
            self._failed_requests.extend(requests)
            self._failed = True
            raise
        self._ended = True
        return events, requests

    def flush(self) -> tuple[RealtimeEvent | GroupedRefineOutcome, ...]:
        if self._closed:
            raise RuntimeError("session is closed")
        events, requests = self._finish("eof")
        return (*events, *self._submit(requests))

    def reset(
        self, *, flush_pending: bool, timeout_seconds: float = 5.0
    ) -> tuple[RealtimeEvent | GroupedRefineOutcome, ...]:
        if self._closed:
            raise RuntimeError("session is closed")
        if not isinstance(flush_pending, bool):
            raise TypeError("flush_pending must be bool")
        timeout_seconds = _timeout(timeout_seconds)
        events, requests = self._finish("reset")
        output: list[RealtimeEvent | GroupedRefineOutcome] = list(events)
        if flush_pending:
            output.extend(self._submit(requests))
            output.extend(self._worker.drain(timeout_seconds))
        else:
            output.extend(self._worker.receipt(item, "reset-without-refine") for item in requests)
        output.extend(self._worker.discard("reset-timeout" if flush_pending else "reset"))
        output.extend(self.session.reset())
        self.scheduler.reset(flush_pending=False, new_session_id=self.session.session_id)
        self._ended = False
        return tuple(output)

    def close(
        self, *, timeout_seconds: float = 5.0
    ) -> tuple[RealtimeEvent | GroupedRefineOutcome, ...]:
        timeout_seconds = _timeout(timeout_seconds)
        if self._closed:
            return ()
        if self._failed:
            return self.abort()
        events = self.flush()
        self._closed = True
        return (*events, *self._worker.close(timeout_seconds))

    def abort(self) -> tuple[GroupedRefineOutcome, ...]:
        """Stop without retrying a failed fast decoder; emit pending-group discards."""
        if self._closed:
            return ()
        self._closed = True
        requests = [*self._failed_requests, *self.scheduler.force("reset")]
        self._failed_requests.clear()
        output = [self._worker.receipt(item, "aborted") for item in requests]
        output.extend(self._worker.discard("aborted"))
        output.extend(self._worker.close(0))
        return tuple(output)
