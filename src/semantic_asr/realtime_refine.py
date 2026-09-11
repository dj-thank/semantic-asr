"""Evidence-safe grouping for a later realtime context re-decode pass.

This module is intentionally transport- and model-free.  It schedules *when* a
set of immutable first-pass finals may be re-decoded with more continuous audio
context; it does not decide *how* that second pass is decoded.

The design follows the useful lifecycle of Hayamimi's two-pass Refiner while
preserving Semantic ASR's stronger evidence rules:

* fast finals are parents and are never mutated;
* group audio is cut once from the original continuous PCM timeline, rather than
  concatenating per-final buffers that may contain overlapping pre-roll/silence;
* every group binds ordered parent final digests and an exact PCM SHA-256;
* idle, duration, history and parent-count bounds are finite;
* EOF/reset behavior is explicit and evidence-producing.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Literal

GroupTrigger = Literal["idle-gap", "max-duration", "parent-limit", "eof", "reset", "manual"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class GroupedRefineConfig:
    """Finite scheduler limits for one realtime session."""

    sample_rate: int = 16_000
    idle_gap_ms: int = 2_000
    max_group_ms: int = 25_000
    min_group_ms: int = 500
    history_keep_ms: int = 30_000
    max_pending_finals: int = 64

    def __post_init__(self) -> None:
        _strict_int(self.sample_rate, name="sample_rate", minimum=1)
        _strict_int(self.idle_gap_ms, name="idle_gap_ms", minimum=1)
        _strict_int(self.max_group_ms, name="max_group_ms", minimum=1)
        _strict_int(self.min_group_ms, name="min_group_ms", minimum=1)
        _strict_int(self.history_keep_ms, name="history_keep_ms", minimum=1)
        _strict_int(self.max_pending_finals, name="max_pending_finals", minimum=1)
        if self.sample_rate != 16_000:
            raise ValueError("the grouped realtime refine contract currently requires 16 kHz PCM")
        if self.min_group_ms > self.max_group_ms:
            raise ValueError("min_group_ms cannot exceed max_group_ms")
        if self.history_keep_ms < self.max_group_ms + self.idle_gap_ms:
            raise ValueError("history_keep_ms must cover max_group_ms + idle_gap_ms")

    def samples_for_ms(self, milliseconds: int) -> int:
        return round(self.sample_rate * milliseconds / 1000)

    @property
    def idle_gap_samples(self) -> int:
        return self.samples_for_ms(self.idle_gap_ms)

    @property
    def max_group_samples(self) -> int:
        return self.samples_for_ms(self.max_group_ms)

    @property
    def min_group_samples(self) -> int:
        return self.samples_for_ms(self.min_group_ms)

    @property
    def history_keep_samples(self) -> int:
        return self.samples_for_ms(self.history_keep_ms)


class BoundedPcmHistory:
    """Exact continuous PCM16LE history indexed by absolute sample position."""

    def __init__(self, *, sample_rate: int = 16_000, keep_samples: int = 480_000) -> None:
        _strict_int(sample_rate, name="sample_rate", minimum=1)
        _strict_int(keep_samples, name="keep_samples", minimum=1)
        self.sample_rate = sample_rate
        self.keep_samples = keep_samples
        self._start_sample = 0
        self._end_sample = 0
        self._pcm = bytearray()

    @property
    def start_sample(self) -> int:
        return self._start_sample

    @property
    def end_sample(self) -> int:
        return self._end_sample

    @property
    def retained_samples(self) -> int:
        return len(self._pcm) // 2

    def append(self, pcm16le: bytes, *, start_sample: int | None = None) -> None:
        if not isinstance(pcm16le, bytes):
            raise TypeError("pcm16le must be bytes")
        if not pcm16le or len(pcm16le) % 2:
            raise ValueError("pcm16le must contain non-empty whole int16 samples")
        if start_sample is None:
            start_sample = self._end_sample
        _strict_int(start_sample, name="start_sample", minimum=0)
        if start_sample != self._end_sample:
            raise ValueError(
                f"PCM history must be contiguous: expected start {self._end_sample}, "
                f"got {start_sample}"
            )
        samples = len(pcm16le) // 2
        self._pcm.extend(pcm16le)
        self._end_sample += samples
        overflow = self.retained_samples - self.keep_samples
        if overflow > 0:
            del self._pcm[: overflow * 2]
            self._start_sample += overflow

    def read(self, start_sample: int, end_sample: int) -> bytes:
        _strict_int(start_sample, name="start_sample", minimum=0)
        _strict_int(end_sample, name="end_sample", minimum=1)
        if end_sample <= start_sample:
            raise ValueError("end_sample must be greater than start_sample")
        if start_sample < self._start_sample or end_sample > self._end_sample:
            raise ValueError(
                "requested PCM range is outside retained history: "
                f"requested=[{start_sample},{end_sample}) retained="
                f"[{self._start_sample},{self._end_sample})"
            )
        lo = (start_sample - self._start_sample) * 2
        hi = (end_sample - self._start_sample) * 2
        result = bytes(self._pcm[lo:hi])
        if len(result) != (end_sample - start_sample) * 2:
            raise RuntimeError("retained PCM range length mismatch")
        return result

    def clear(self, *, start_sample: int = 0) -> None:
        _strict_int(start_sample, name="start_sample", minimum=0)
        self._pcm.clear()
        self._start_sample = start_sample
        self._end_sample = start_sample


@dataclass(frozen=True, slots=True)
class RefineParentFinal:
    """Minimum immutable first-pass evidence needed to bind a refine group."""

    utterance_id: str
    final_digest: str
    audio_sha256: str
    start_sample: int
    end_sample: int
    observed_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.utterance_id, str) or not self.utterance_id:
            raise ValueError("utterance_id is required")
        _require_sha256(self.final_digest, name="final_digest")
        _require_sha256(self.audio_sha256, name="audio_sha256")
        _strict_int(self.start_sample, name="start_sample", minimum=0)
        _strict_int(self.end_sample, name="end_sample", minimum=1)
        if self.end_sample <= self.start_sample:
            raise ValueError("parent end_sample must be greater than start_sample")
        if not isinstance(self.observed_text, str):
            raise TypeError("observed_text must be str")

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample


@dataclass(frozen=True, slots=True)
class GroupedRefineRequest:
    """One exact, provenance-bound context re-decode request."""

    sequence: int
    session_id: str
    group_id: str
    trigger: GroupTrigger
    parents: tuple[RefineParentFinal, ...]
    pcm16le: bytes
    sample_rate: int
    start_sample: int
    end_sample: int
    group_audio_sha256: str
    eligible_for_decode: bool
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        _strict_int(self.sequence, name="sequence", minimum=1)
        if not self.session_id or not self.group_id:
            raise ValueError("session_id and group_id are required")
        if self.trigger not in {
            "idle-gap",
            "max-duration",
            "parent-limit",
            "eof",
            "reset",
            "manual",
        }:
            raise ValueError(f"unsupported group trigger: {self.trigger!r}")
        if not self.parents:
            raise ValueError("a grouped refine request requires at least one parent final")
        if not isinstance(self.pcm16le, bytes) or not self.pcm16le or len(self.pcm16le) % 2:
            raise ValueError("pcm16le must contain non-empty whole int16 samples")
        _strict_int(self.sample_rate, name="sample_rate", minimum=1)
        _strict_int(self.start_sample, name="start_sample", minimum=0)
        _strict_int(self.end_sample, name="end_sample", minimum=1)
        if self.end_sample <= self.start_sample:
            raise ValueError("group end_sample must be greater than start_sample")
        if len(self.pcm16le) // 2 != self.end_sample - self.start_sample:
            raise ValueError("group sample bounds do not match PCM length")
        _require_sha256(self.group_audio_sha256, name="group_audio_sha256")
        if _sha256(self.pcm16le) != self.group_audio_sha256:
            raise ValueError("group_audio_sha256 does not match PCM bytes")
        if not isinstance(self.eligible_for_decode, bool):
            raise TypeError("eligible_for_decode must be bool")
        if self.eligible_for_decode and self.skip_reason is not None:
            raise ValueError("eligible grouped refine requests cannot have skip_reason")
        if not self.eligible_for_decode and not self.skip_reason:
            raise ValueError("ineligible grouped refine requests require skip_reason")

        previous: RefineParentFinal | None = None
        seen_ids: set[str] = set()
        for parent in self.parents:
            if parent.utterance_id in seen_ids:
                raise ValueError("duplicate parent utterance_id in refine group")
            seen_ids.add(parent.utterance_id)
            if parent.start_sample < self.start_sample or parent.end_sample > self.end_sample:
                raise ValueError("parent final falls outside grouped PCM range")
            if previous is not None:
                if parent.start_sample < previous.start_sample:
                    raise ValueError("parent finals must be ordered by timeline")
                if parent.end_sample <= previous.end_sample:
                    raise ValueError("parent final end positions must strictly increase")
            previous = parent

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def evidence_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": "semantic-asr-grouped-refine-request-v1",
                "sequence": self.sequence,
                "sessionId": self.session_id,
                "groupId": self.group_id,
                "trigger": self.trigger,
                "parentFinals": [
                    {
                        "utteranceId": parent.utterance_id,
                        "finalDigest": parent.final_digest,
                        "audioSha256": parent.audio_sha256,
                        "startSample": parent.start_sample,
                        "endSample": parent.end_sample,
                    }
                    for parent in self.parents
                ],
                "sampleRate": self.sample_rate,
                "startSample": self.start_sample,
                "endSample": self.end_sample,
                "groupAudioSha256": self.group_audio_sha256,
                "eligibleForDecode": self.eligible_for_decode,
                "skipReason": self.skip_reason,
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "semantic-asr-grouped-refine-request-v1",
            "sequence": self.sequence,
            "sessionId": self.session_id,
            "groupId": self.group_id,
            "trigger": self.trigger,
            "parentFinals": [
                {
                    "utteranceId": parent.utterance_id,
                    "finalDigest": parent.final_digest,
                    "audioSha256": parent.audio_sha256,
                    "startSample": parent.start_sample,
                    "endSample": parent.end_sample,
                    "observedText": parent.observed_text,
                }
                for parent in self.parents
            ],
            "sampleRate": self.sample_rate,
            "startSample": self.start_sample,
            "endSample": self.end_sample,
            "durationMs": round(self.duration_samples * 1000 / self.sample_rate, 3),
            "groupAudioSha256": self.group_audio_sha256,
            "eligibleForDecode": self.eligible_for_decode,
            "skipReason": self.skip_reason,
            "evidenceDigest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class GroupedRefineDiscard:
    """Evidence that pending parents were deliberately discarded on reset."""

    session_id: str
    parent_final_digests: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id is required")
        if not self.parent_final_digests:
            raise ValueError("discard requires at least one parent final digest")
        for digest in self.parent_final_digests:
            _require_sha256(digest, name="parent_final_digest")
        if not self.reason:
            raise ValueError("discard reason is required")

    @property
    def evidence_digest(self) -> str:
        return _canonical_sha256(
            {
                "schema": "semantic-asr-grouped-refine-discard-v1",
                "sessionId": self.session_id,
                "parentFinalDigests": list(self.parent_final_digests),
                "reason": self.reason,
            }
        )


class GroupedRefineScheduler:
    """Serial scheduler for bounded longer-context second-pass requests.

    Raw stream PCM must be appended in exact timeline order with `feed_pcm16()`.
    Immutable first-pass parents are then registered with `add_final()`.  The
    scheduler closes a group when:

    * true idle audio reaches `idle_gap_ms`,
    * adding another parent would exceed `max_group_ms`,
    * the parent count reaches its finite limit, or
    * the caller explicitly forces EOF/reset/manual closure.

    The returned request owns a copy of the exact grouped PCM, so resetting the
    rolling history afterwards cannot change the evidence already emitted.
    """

    def __init__(
        self,
        *,
        config: GroupedRefineConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config or GroupedRefineConfig()
        self.session_id = session_id or uuid.uuid4().hex
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        self.history = BoundedPcmHistory(
            sample_rate=self.config.sample_rate,
            keep_samples=self.config.history_keep_samples,
        )
        self._pending: list[RefineParentFinal] = []
        self._sequence = 0

    @property
    def pending(self) -> tuple[RefineParentFinal, ...]:
        return tuple(self._pending)

    def _close(self, trigger: GroupTrigger) -> GroupedRefineRequest:
        if not self._pending:
            raise RuntimeError("no pending finals to close")
        first = self._pending[0]
        last = self._pending[-1]
        start_sample = first.start_sample
        end_sample = last.end_sample
        pcm = self.history.read(start_sample, end_sample)
        self._sequence += 1
        parents = tuple(self._pending)
        self._pending.clear()
        duration = end_sample - start_sample
        eligible = duration >= self.config.min_group_samples
        return GroupedRefineRequest(
            sequence=self._sequence,
            session_id=self.session_id,
            group_id=f"{self.session_id}:g{self._sequence}",
            trigger=trigger,
            parents=parents,
            pcm16le=pcm,
            sample_rate=self.config.sample_rate,
            start_sample=start_sample,
            end_sample=end_sample,
            group_audio_sha256=_sha256(pcm),
            eligible_for_decode=eligible,
            skip_reason=None if eligible else "group-too-short",
        )

    def _close_if_idle(self, now_sample: int) -> list[GroupedRefineRequest]:
        if not self._pending:
            return []
        if now_sample < self._pending[-1].end_sample:
            raise ValueError("now_sample cannot precede the latest pending final")
        if now_sample - self._pending[-1].end_sample < self.config.idle_gap_samples:
            return []
        return [self._close("idle-gap")]

    def feed_pcm16(
        self,
        pcm16le: bytes,
        *,
        start_sample: int | None = None,
    ) -> tuple[GroupedRefineRequest, ...]:
        """Append exact raw stream PCM and close a pending group after true idle."""

        self.history.append(pcm16le, start_sample=start_sample)
        return tuple(self._close_if_idle(self.history.end_sample))

    def add_final(self, parent: RefineParentFinal) -> tuple[GroupedRefineRequest, ...]:
        """Register one immutable first-pass final in chronological order."""

        if not isinstance(parent, RefineParentFinal):
            raise TypeError("parent must be RefineParentFinal")
        if parent.start_sample < self.history.start_sample or parent.end_sample > self.history.end_sample:
            raise ValueError("parent final audio is outside retained continuous PCM history")

        output: list[GroupedRefineRequest] = []
        if self._pending:
            previous = self._pending[-1]
            if parent.start_sample < previous.start_sample or parent.end_sample <= previous.end_sample:
                raise ValueError("parent finals must arrive in strictly advancing timeline order")
            if parent.start_sample - previous.end_sample >= self.config.idle_gap_samples:
                output.append(self._close("idle-gap"))
            elif parent.end_sample - self._pending[0].start_sample > self.config.max_group_samples:
                output.append(self._close("max-duration"))
            elif len(self._pending) >= self.config.max_pending_finals:
                output.append(self._close("parent-limit"))

        self._pending.append(parent)
        if parent.end_sample - self._pending[0].start_sample >= self.config.max_group_samples:
            output.append(self._close("max-duration"))
        return tuple(output)

    def force(self, trigger: Literal["eof", "reset", "manual"] = "manual") -> tuple[GroupedRefineRequest, ...]:
        if trigger not in {"eof", "reset", "manual"}:
            raise ValueError("force trigger must be eof, reset, or manual")
        if not self._pending:
            return ()
        return (self._close(trigger),)

    def reset(
        self,
        *,
        flush_pending: bool,
        new_session_id: str | None = None,
    ) -> tuple[tuple[GroupedRefineRequest, ...], GroupedRefineDiscard | None]:
        """Start a fresh session, explicitly flushing or discarding pending parents."""

        flushed: tuple[GroupedRefineRequest, ...] = ()
        discarded: GroupedRefineDiscard | None = None
        if self._pending:
            if flush_pending:
                flushed = self.force("reset")
            else:
                discarded = GroupedRefineDiscard(
                    session_id=self.session_id,
                    parent_final_digests=tuple(parent.final_digest for parent in self._pending),
                    reason="reset-without-refine",
                )
                self._pending.clear()

        self.session_id = new_session_id or uuid.uuid4().hex
        if not self.session_id:
            raise ValueError("new_session_id cannot be empty")
        self._sequence = 0
        self.history.clear(start_sample=0)
        return flushed, discarded
