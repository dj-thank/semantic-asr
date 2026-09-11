"""Bounded, dependency-free realtime contract for the opt-in Japanese Reazon path.

The design is intentionally narrow.  It does not introduce a generic streaming
plugin framework and it does not change any existing default profile.  A caller
supplies a speech/non-speech decision for each PCM16 chunk and a decoder callback.
The session then owns the temporal invariants that are easy to get wrong in a live
UI:

* partial text is display-only and never becomes immutable observed evidence;
* one first-pass final is bound to the exact buffered PCM bytes;
* a later refine event is a child of that final instead of mutating it;
* audio, event history and wall-clock work are explicitly bounded.

The defaults mirror the useful shape of Hayamimi's Japanese live path (500 ms
partials, 350 ms endpointing, 12 s maximum speech, 800 ms pre-roll, 2 s idle
before refinement) without claiming its latency or accuracy measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

EventKind = Literal[
    "speech_start",
    "partial",
    "final",
    "refine",
    "warning",
    "session_summary",
]
DecodeMode = Literal["partial", "final"]


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


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


@dataclass(frozen=True, slots=True)
class RealtimeReazonConfig:
    """Finite runtime limits for one Japanese realtime session."""

    sample_rate: int = 16_000
    partial_interval_ms: int = 500
    min_silence_ms: int = 350
    max_speech_ms: int = 12_000
    preroll_ms: int = 800
    refine_idle_ms: int = 2_000
    max_chunk_ms: int = 1_000
    max_history_utterances: int = 16

    def __post_init__(self) -> None:
        _strict_int(self.sample_rate, name="sample_rate", minimum=1)
        _strict_int(self.partial_interval_ms, name="partial_interval_ms", minimum=1)
        _strict_int(self.min_silence_ms, name="min_silence_ms", minimum=1)
        _strict_int(self.max_speech_ms, name="max_speech_ms", minimum=1)
        _strict_int(self.preroll_ms, name="preroll_ms", minimum=0)
        _strict_int(self.refine_idle_ms, name="refine_idle_ms", minimum=0)
        _strict_int(self.max_chunk_ms, name="max_chunk_ms", minimum=1)
        _strict_int(self.max_history_utterances, name="max_history_utterances", minimum=1)
        if self.sample_rate != 16_000:
            raise ValueError("the current Reazon realtime contract requires 16 kHz PCM")
        if self.max_speech_ms > 30_000:
            raise ValueError("the Reazon decoder contract cannot exceed 30 second windows")
        if self.partial_interval_ms > self.max_speech_ms:
            raise ValueError("partial_interval_ms cannot exceed max_speech_ms")
        if self.min_silence_ms >= self.max_speech_ms:
            raise ValueError("min_silence_ms must be smaller than max_speech_ms")
        if self.preroll_ms > self.max_speech_ms:
            raise ValueError("preroll_ms cannot exceed max_speech_ms")

    def samples_for_ms(self, milliseconds: int) -> int:
        return round(self.sample_rate * milliseconds / 1000)

    @property
    def partial_interval_samples(self) -> int:
        return self.samples_for_ms(self.partial_interval_ms)

    @property
    def min_silence_samples(self) -> int:
        return self.samples_for_ms(self.min_silence_ms)

    @property
    def max_speech_samples(self) -> int:
        return self.samples_for_ms(self.max_speech_ms)

    @property
    def preroll_samples(self) -> int:
        return self.samples_for_ms(self.preroll_ms)

    @property
    def refine_idle_samples(self) -> int:
        return self.samples_for_ms(self.refine_idle_ms)

    @property
    def max_chunk_samples(self) -> int:
        return self.samples_for_ms(self.max_chunk_ms)


@dataclass(frozen=True, slots=True)
class RealtimeDecodeInput:
    """Exact PCM evidence passed to a caller-owned decoder."""

    session_id: str
    utterance_id: str
    mode: DecodeMode
    pcm16le: bytes
    sample_rate: int
    start_sample: int
    end_sample: int
    audio_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"partial", "final"}:
            raise ValueError("mode must be partial or final")
        if not self.session_id or not self.utterance_id:
            raise ValueError("session_id and utterance_id are required")
        if not isinstance(self.pcm16le, bytes) or not self.pcm16le or len(self.pcm16le) % 2:
            raise ValueError("pcm16le must contain non-empty whole int16 samples")
        _strict_int(self.sample_rate, name="sample_rate", minimum=1)
        _strict_int(self.start_sample, name="start_sample", minimum=0)
        _strict_int(self.end_sample, name="end_sample", minimum=1)
        expected_samples = len(self.pcm16le) // 2
        if self.end_sample - self.start_sample != expected_samples:
            raise ValueError("decode sample bounds do not match PCM length")
        if _sha256(self.pcm16le) != self.audio_sha256:
            raise ValueError("decode audio_sha256 does not match PCM bytes")


@dataclass(frozen=True, slots=True)
class RealtimeEvent:
    """Structured event whose evidence digest excludes runtime latency metadata."""

    sequence: int
    kind: EventKind
    session_id: str
    utterance_id: str | None
    emitted_at_sample: int
    text: str | None = None
    audio_sha256: str | None = None
    parent_final_digest: str | None = None
    reason: str | None = None
    decode_duration_ms: float | None = None

    def __post_init__(self) -> None:
        _strict_int(self.sequence, name="sequence", minimum=1)
        _strict_int(self.emitted_at_sample, name="emitted_at_sample", minimum=0)
        if self.kind not in {
            "speech_start",
            "partial",
            "final",
            "refine",
            "warning",
            "session_summary",
        }:
            raise ValueError(f"unsupported event kind: {self.kind!r}")
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.kind in {"speech_start", "partial", "final", "refine"} and not self.utterance_id:
            raise ValueError(f"{self.kind} requires utterance_id")
        if self.kind in {"partial", "final", "refine"} and self.audio_sha256 is None:
            raise ValueError(f"{self.kind} requires audio_sha256")
        if self.kind == "refine" and self.parent_final_digest is None:
            raise ValueError("refine requires parent_final_digest")
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
        return _canonical_sha256(
            {
                "schema": "semantic-asr-realtime-event-v1",
                "sequence": self.sequence,
                "kind": self.kind,
                "sessionId": self.session_id,
                "utteranceId": self.utterance_id,
                "emittedAtSample": self.emitted_at_sample,
                "text": self.text,
                "audioSha256": self.audio_sha256,
                "parentFinalDigest": self.parent_final_digest,
                "reason": self.reason,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "semantic-asr-realtime-event-v1",
            "sequence": self.sequence,
            "kind": self.kind,
            "sessionId": self.session_id,
            "utteranceId": self.utterance_id,
            "emittedAtSample": self.emitted_at_sample,
            "emittedAtMs": round(self.emitted_at_sample * 1000 / 16_000, 3),
            "text": self.text,
            "audioSha256": self.audio_sha256,
            "parentFinalDigest": self.parent_final_digest,
            "reason": self.reason,
            "decodeDurationMs": self.decode_duration_ms,
            "evidenceDigest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalUtterance:
    utterance_id: str
    pcm16le: bytes
    start_sample: int
    end_sample: int
    text: str
    audio_sha256: str
    final_digest: str

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id is required")
        if not isinstance(self.pcm16le, bytes) or not self.pcm16le or len(self.pcm16le) % 2:
            raise ValueError("final PCM must contain non-empty whole int16 samples")
        if _sha256(self.pcm16le) != self.audio_sha256:
            raise ValueError("final audio digest does not match PCM")
        if self.end_sample - self.start_sample != len(self.pcm16le) // 2:
            raise ValueError("final sample bounds do not match PCM")


Decoder = Callable[[RealtimeDecodeInput], str]
Refiner = Callable[[FinalUtterance], str | None]


class RealtimeReazonSession:
    """One serial, bounded Japanese realtime session.

    `speech` is deliberately injected by the caller.  This keeps the evidence and
    lifecycle contract dependency-free while allowing a concrete runner to use
    Silero/sherpa-onnx, WebRTC VAD, a hardware VAD, or deterministic test labels.
    The session must be driven from one serialized decode loop; it is not a
    thread-safe queue.
    """

    def __init__(
        self,
        decoder: Decoder,
        *,
        config: RealtimeReazonConfig | None = None,
        partial_decoder: Decoder | None = None,
        refiner: Refiner | None = None,
        session_id: str | None = None,
    ) -> None:
        if not callable(decoder):
            raise TypeError("decoder must be callable")
        if partial_decoder is not None and not callable(partial_decoder):
            raise TypeError("partial_decoder must be callable")
        if refiner is not None and not callable(refiner):
            raise TypeError("refiner must be callable")
        self.config = config or RealtimeReazonConfig()
        self.decoder = decoder
        self.partial_decoder = partial_decoder or decoder
        self.refiner = refiner
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        self._sequence = 0
        self._sample_cursor = 0
        self._utterance_number = 0
        self._active_id: str | None = None
        self._active_start_sample = 0
        self._active = bytearray()
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_partial_sample = 0
        self._preroll: deque[bytes] = deque()
        self._preroll_sample_count = 0
        self._history: deque[FinalUtterance] = deque(maxlen=self.config.max_history_utterances)
        self._refined: set[str] = set()
        self._final_count = 0
        self._partial_count = 0
        self._refine_count = 0

    @property
    def sample_cursor(self) -> int:
        return self._sample_cursor

    @property
    def active_utterance_id(self) -> str | None:
        return self._active_id

    @property
    def finals(self) -> tuple[FinalUtterance, ...]:
        return tuple(self._history)

    def _emit(
        self,
        kind: EventKind,
        *,
        utterance_id: str | None = None,
        text: str | None = None,
        audio_sha256: str | None = None,
        parent_final_digest: str | None = None,
        reason: str | None = None,
        decode_duration_ms: float | None = None,
    ) -> RealtimeEvent:
        self._sequence += 1
        return RealtimeEvent(
            sequence=self._sequence,
            kind=kind,
            session_id=self.session_id,
            utterance_id=utterance_id,
            emitted_at_sample=self._sample_cursor,
            text=text,
            audio_sha256=audio_sha256,
            parent_final_digest=parent_final_digest,
            reason=reason,
            decode_duration_ms=decode_duration_ms,
        )

    def _append_preroll(self, pcm16le: bytes) -> None:
        if self.config.preroll_samples == 0:
            self._preroll.clear()
            self._preroll_sample_count = 0
            return
        self._preroll.append(pcm16le)
        self._preroll_sample_count += len(pcm16le) // 2
        while self._preroll and self._preroll_sample_count > self.config.preroll_samples:
            overflow = self._preroll_sample_count - self.config.preroll_samples
            first = self._preroll[0]
            first_samples = len(first) // 2
            if overflow >= first_samples:
                self._preroll.popleft()
                self._preroll_sample_count -= first_samples
                continue
            cut_bytes = overflow * 2
            self._preroll[0] = first[cut_bytes:]
            self._preroll_sample_count -= overflow
            break

    def _start(self, pcm16le: bytes) -> RealtimeEvent:
        self._utterance_number += 1
        self._active_id = f"{self.session_id}:u{self._utterance_number}"
        prefix = b"".join(self._preroll)
        self._active_start_sample = self._sample_cursor - self._preroll_sample_count
        self._active = bytearray(prefix)
        self._active.extend(pcm16le)
        self._speech_samples = len(pcm16le) // 2
        self._silence_samples = 0
        self._last_partial_sample = self._sample_cursor
        self._preroll.clear()
        self._preroll_sample_count = 0
        return self._emit("speech_start", utterance_id=self._active_id, reason="vad-speech")

    def _decode_active(self, *, mode: DecodeMode) -> tuple[str, str, float]:
        if self._active_id is None or not self._active:
            raise RuntimeError("no active utterance to decode")
        pcm = bytes(self._active)
        audio_sha256 = _sha256(pcm)
        request = RealtimeDecodeInput(
            session_id=self.session_id,
            utterance_id=self._active_id,
            mode=mode,
            pcm16le=pcm,
            sample_rate=self.config.sample_rate,
            start_sample=self._active_start_sample,
            end_sample=self._active_start_sample + len(pcm) // 2,
            audio_sha256=audio_sha256,
        )
        decoder = self.partial_decoder if mode == "partial" else self.decoder
        started = time.perf_counter_ns()
        text = decoder(request)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        if not isinstance(text, str):
            raise TypeError("decoder must return str")
        return text, audio_sha256, elapsed_ms

    def _maybe_partial(self) -> RealtimeEvent | None:
        if self._active_id is None:
            return None
        active_end = self._active_start_sample + len(self._active) // 2
        if active_end - self._last_partial_sample < self.config.partial_interval_samples:
            return None
        text, audio_sha256, elapsed_ms = self._decode_active(mode="partial")
        self._last_partial_sample = active_end
        self._partial_count += 1
        return self._emit(
            "partial",
            utterance_id=self._active_id,
            text=text,
            audio_sha256=audio_sha256,
            reason="display-only",
            decode_duration_ms=elapsed_ms,
        )

    def _finalize(self, *, reason: str) -> RealtimeEvent:
        if self._active_id is None:
            raise RuntimeError("no active utterance to finalize")
        utterance_id = self._active_id
        start_sample = self._active_start_sample
        pcm = bytes(self._active)
        text, audio_sha256, elapsed_ms = self._decode_active(mode="final")
        end_sample = start_sample + len(pcm) // 2
        event = self._emit(
            "final",
            utterance_id=utterance_id,
            text=text,
            audio_sha256=audio_sha256,
            reason=reason,
            decode_duration_ms=elapsed_ms,
        )
        final = FinalUtterance(
            utterance_id=utterance_id,
            pcm16le=pcm,
            start_sample=start_sample,
            end_sample=end_sample,
            text=text,
            audio_sha256=audio_sha256,
            final_digest=event.evidence_digest,
        )
        evicted = self._history[0].utterance_id if len(self._history) == self._history.maxlen else None
        self._history.append(final)
        if evicted is not None and all(item.utterance_id != evicted for item in self._history):
            self._refined.discard(evicted)
        self._final_count += 1
        self._active_id = None
        self._active = bytearray()
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_partial_sample = 0
        return event

    def _run_ready_refinements(self) -> list[RealtimeEvent]:
        if self.refiner is None:
            return []
        output: list[RealtimeEvent] = []
        for final in tuple(self._history):
            if final.utterance_id in self._refined:
                continue
            if self._sample_cursor - final.end_sample < self.config.refine_idle_samples:
                continue
            try:
                started = time.perf_counter_ns()
                text = self.refiner(final)
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if text is not None and not isinstance(text, str):
                    raise TypeError("refiner must return str or None")
            except Exception as exc:
                self._refined.add(final.utterance_id)
                output.append(
                    self._emit(
                        "warning",
                        utterance_id=final.utterance_id,
                        audio_sha256=None,
                        reason=f"refine-failed:{type(exc).__name__}",
                    )
                )
                continue
            self._refined.add(final.utterance_id)
            if text is None:
                continue
            self._refine_count += 1
            output.append(
                self._emit(
                    "refine",
                    utterance_id=final.utterance_id,
                    text=text,
                    audio_sha256=final.audio_sha256,
                    parent_final_digest=final.final_digest,
                    reason="child-of-first-pass-final",
                    decode_duration_ms=elapsed_ms,
                )
            )
        return output

    def feed_pcm16(self, pcm16le: bytes, *, speech: bool) -> tuple[RealtimeEvent, ...]:
        """Consume one serialized PCM16LE chunk and return newly emitted events."""

        if not isinstance(speech, bool):
            raise TypeError("speech must be bool")
        if not isinstance(pcm16le, bytes):
            raise TypeError("pcm16le must be bytes")
        if not pcm16le or len(pcm16le) % 2:
            raise ValueError("pcm16le must contain non-empty whole int16 samples")
        chunk_samples = len(pcm16le) // 2
        if chunk_samples > self.config.max_chunk_samples:
            raise ValueError("PCM chunk exceeds max_chunk_ms")

        events: list[RealtimeEvent] = []
        chunk_end = self._sample_cursor + chunk_samples
        if self._active_id is None:
            if speech:
                events.append(self._start(pcm16le))
            else:
                self._append_preroll(pcm16le)
        else:
            self._active.extend(pcm16le)
            if speech:
                self._speech_samples += chunk_samples
                self._silence_samples = 0
            else:
                self._silence_samples += chunk_samples

        self._sample_cursor = chunk_end

        if self._active_id is not None:
            total_samples = len(self._active) // 2
            if total_samples >= self.config.max_speech_samples:
                events.append(self._finalize(reason="max-speech"))
            elif not speech and self._silence_samples >= self.config.min_silence_samples:
                events.append(self._finalize(reason="silence"))
                self._append_preroll(pcm16le)
            elif speech:
                partial = self._maybe_partial()
                if partial is not None:
                    events.append(partial)

        events.extend(self._run_ready_refinements())
        return tuple(events)

    def flush(self) -> tuple[RealtimeEvent, ...]:
        """Finalize an in-progress utterance without fabricating more PCM."""

        events: list[RealtimeEvent] = []
        if self._active_id is not None:
            events.append(self._finalize(reason="flush"))
        events.extend(self._run_ready_refinements())
        return tuple(events)

    def reset(self) -> tuple[RealtimeEvent, ...]:
        """End the current session and clear every audio/evidence reference."""

        events = list(self.flush())
        events.append(
            self._emit(
                "session_summary",
                reason=(
                    f"partials={self._partial_count};finals={self._final_count};"
                    f"refines={self._refine_count}"
                ),
            )
        )
        self.session_id = uuid.uuid4().hex
        self._sequence = 0
        self._sample_cursor = 0
        self._utterance_number = 0
        self._active_id = None
        self._active_start_sample = 0
        self._active = bytearray()
        self._speech_samples = 0
        self._silence_samples = 0
        self._last_partial_sample = 0
        self._preroll.clear()
        self._preroll_sample_count = 0
        self._history.clear()
        self._refined.clear()
        self._final_count = 0
        self._partial_count = 0
        self._refine_count = 0
        return tuple(events)
