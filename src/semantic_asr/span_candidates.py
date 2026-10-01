"""Bounded, unverified full-window proposals from explicitly aligned re-decodes.

No recognition scores, probabilities, or observed transcripts are produced here.
Every draft changes one anchored interval; other characters remain byte-for-byte
unchanged. An alignment digest records lineage, not proof of alignment accuracy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Literal


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@dataclass(frozen=True, slots=True)
class SpanWindow:
    """Immutable parent identity; times are absolute recording milliseconds."""

    candidate_id: str
    text: str
    source_audio_sha256: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be nonempty")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("parent text must be nonempty")
        _sha256(self.source_audio_sha256, "source_audio_sha256")
        _integer(self.start_ms, "start_ms")
        _integer(self.end_ms, "end_ms", self.start_ms + 1)

    @property
    def digest(self) -> str:
        return _digest({"schema": "span-window-v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class SpanAnchor:
    """Caller-supplied text/audio correspondence, never inferred from text length.

    The complete decoded crop replaces the complete character interval. Padding
    is only valid when its corresponding text is also included in this interval.
    Character offsets are Python Unicode code-point indices, not UTF-8 bytes.
    """

    window_digest: str
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int
    alignment_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.window_digest, "window_digest")
        _sha256(self.alignment_sha256, "alignment_sha256")
        _integer(self.char_start, "char_start")
        _integer(self.char_end, "char_end", self.char_start + 1)
        _integer(self.start_ms, "start_ms")
        _integer(self.end_ms, "end_ms", self.start_ms + 1)

    def validate(self, window: SpanWindow) -> None:
        if self.window_digest != window.digest:
            raise ValueError("anchor belongs to a different parent window")
        if self.char_end > len(window.text):
            raise ValueError("anchor exceeds parent text")
        if self.start_ms < window.start_ms or self.end_ms > window.end_ms:
            raise ValueError("anchor exceeds parent audio window")

    @property
    def digest(self) -> str:
        return _digest({"schema": "span-anchor-v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class SpanDecode:
    """A crop transcript and the digest of its original decoder evidence."""

    text: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("decoded text must be a string")
        _sha256(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True, slots=True)
class SpanDraft:
    window: SpanWindow
    anchor: SpanAnchor
    replacement: str
    decoder_fingerprint: str
    decode_evidence_sha256: str

    def __post_init__(self) -> None:
        self.anchor.validate(self.window)
        if not isinstance(self.replacement, str) or not self.replacement.strip():
            raise ValueError("replacement must contain speech text")
        if self.replacement == self.window.text[self.anchor.char_start : self.anchor.char_end]:
            raise ValueError("a draft must change the anchored text")
        _sha256(self.decoder_fingerprint, "decoder_fingerprint")
        _sha256(self.decode_evidence_sha256, "decode_evidence_sha256")

    @property
    def text(self) -> str:
        return (
            self.window.text[: self.anchor.char_start]
            + self.replacement
            + self.window.text[self.anchor.char_end :]
        )

    @property
    def observed_eligible(self) -> bool:
        return False

    @property
    def digest(self) -> str:
        return _digest({"schema": "unverified-span-draft-v1", **asdict(self)})


@dataclass(frozen=True, slots=True)
class SpanBudget:
    max_spans: int = 4
    max_hypotheses_per_span: int = 5
    max_candidates: int = 16
    max_audio_ms: int = 20_000
    max_replacement_chars: int = 256

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _integer(value, name, 1)


AttemptStatus = Literal[
    "decoded",
    "error",
    "duplicate-anchor",
    "span-budget",
    "audio-budget",
    "candidate-budget",
]


@dataclass(frozen=True, slots=True)
class SpanAttempt:
    anchor_digest: str
    status: AttemptStatus
    decoded_hypotheses: int = 0
    emitted_candidates: int = 0
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class SpanExpansion:
    window: SpanWindow
    drafts: tuple[SpanDraft, ...] = ()
    attempts: tuple[SpanAttempt, ...] = ()
    decoder_calls: int = 0
    requested_audio_ms: int = 0
    enabled: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        _integer(self.decoder_calls, "decoder_calls")
        _integer(self.requested_audio_ms, "requested_audio_ms")
        object.__setattr__(self, "drafts", tuple(self.drafts))
        object.__setattr__(self, "attempts", tuple(self.attempts))
        if not self.enabled and (
            self.drafts or self.attempts or self.decoder_calls or self.requested_audio_ms
        ):
            raise ValueError("disabled expansion cannot contain executed work")
        if any(draft.window != self.window for draft in self.drafts):
            raise ValueError("draft belongs to a different parent window")
        if len({draft.text for draft in self.drafts}) != len(self.drafts):
            raise ValueError("draft surfaces must be unique")

    @property
    def texts(self) -> tuple[str, ...]:
        """Original first; these are alternatives for evaluation, not decisions."""
        return (self.window.text, *(draft.text for draft in self.drafts))

    @property
    def digest(self) -> str:
        return _digest({"schema": "span-expansion-v1", **asdict(self)})


def generate_span_drafts(
    window: SpanWindow,
    anchors: Sequence[SpanAnchor],
    decode: Callable[[SpanAnchor], Iterable[SpanDecode]],
    *,
    decoder_fingerprint: str,
    budget: SpanBudget | None = None,
    enabled: bool = False,
) -> SpanExpansion:
    """Generate independent single-edit proposals with deterministic call limits.

    Input order is priority order. Overlapping anchors are separate alternatives,
    never cumulative edits. A failing iterator invalidates that anchor's entire
    batch. Calls and requested audio still count after a decoder failure. Timeouts
    and process isolation remain the decoder's responsibility, not a claimed limit
    on wall-clock time. Disabled mode performs no callbacks.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be a boolean")
    if not enabled:
        return SpanExpansion(window)
    _sha256(decoder_fingerprint, "decoder_fingerprint")
    budget = budget or SpanBudget()
    anchors = tuple(anchors)
    for anchor in anchors:
        anchor.validate(window)
    drafts: list[SpanDraft] = []
    attempts: list[SpanAttempt] = []
    seen_anchors: set[str] = set()
    seen_texts = {window.text}
    calls = audio_ms = 0
    for anchor in anchors:
        duration = anchor.end_ms - anchor.start_ms
        status: AttemptStatus | None = None
        if anchor.digest in seen_anchors:
            status = "duplicate-anchor"
        elif len(drafts) >= budget.max_candidates:
            status = "candidate-budget"
        elif calls >= budget.max_spans:
            status = "span-budget"
        elif audio_ms + duration > budget.max_audio_ms:
            status = "audio-budget"
        seen_anchors.add(anchor.digest)
        if status is not None:
            attempts.append(SpanAttempt(anchor.digest, status))
            continue
        calls += 1
        audio_ms += duration
        count = 0
        staged: list[SpanDraft] = []
        staged_texts: set[str] = set()
        try:
            for row in islice(decode(anchor), budget.max_hypotheses_per_span):
                count += 1
                if not isinstance(row, SpanDecode):
                    raise TypeError("decoder must return SpanDecode records")
                if not row.text.strip() or len(row.text) > budget.max_replacement_chars:
                    continue
                if row.text == window.text[anchor.char_start : anchor.char_end]:
                    continue
                draft = SpanDraft(
                    window, anchor, row.text, decoder_fingerprint, row.evidence_sha256
                )
                if draft.text in seen_texts or draft.text in staged_texts:
                    continue
                staged.append(draft)
                staged_texts.add(draft.text)
                if len(drafts) + len(staged) >= budget.max_candidates:
                    break
        except Exception as exc:
            # No exception message: adapters may include local paths or request text.
            attempts.append(SpanAttempt(anchor.digest, "error", count, 0, type(exc).__name__))
            continue
        drafts.extend(staged)
        seen_texts.update(staged_texts)
        attempts.append(SpanAttempt(anchor.digest, "decoded", count, len(staged)))
    return SpanExpansion(window, tuple(drafts), tuple(attempts), calls, audio_ms, True)
