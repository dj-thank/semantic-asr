"""Candidate-independent phone/mora posterior evidence.

The ASR candidate does not create these posteriors. A frozen acoustic model emits a frame-level
posterior distribution first; candidate pronunciations are then scored with the exact CTC forward
algorithm. Phone and mora scores stay in separate score domains until a held-out normalization
profile converts them to bounded utilities for deliberation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .contracts import sha256_json
from .score_semantics import EvidenceScore, ScoreKind

PosteriorKind = Literal["phone", "mora"]


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _logsumexp(values: Iterable[float]) -> float:
    rows = tuple(values)
    if not rows:
        return -math.inf
    maximum = max(rows)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in rows))


@dataclass(frozen=True, slots=True)
class PosteriorFrame:
    """One time-ordered acoustic posterior frame.

    ``probabilities`` is stored as a sorted tuple so its digest is stable and caller mutation cannot
    change evidence after construction.
    """

    start_ms: int
    end_ms: int
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if isinstance(self.start_ms, bool) or isinstance(self.end_ms, bool):
            raise TypeError("frame timestamps must be integers")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("posterior frame requires 0 <= start_ms < end_ms")
        if not self.probabilities:
            raise ValueError("posterior frame requires a probability distribution")
        normalized: list[tuple[str, float]] = []
        seen: set[str] = set()
        for symbol, value in self.probabilities:
            if not symbol:
                raise ValueError("posterior symbols must not be empty")
            if symbol in seen:
                raise ValueError(f"duplicate posterior symbol: {symbol!r}")
            seen.add(symbol)
            probability = _strict_float(value, name="posterior probability")
            if not 0.0 <= probability <= 1.0:
                raise ValueError("posterior probabilities must be in [0, 1]")
            normalized.append((str(symbol), probability))
        normalized.sort(key=lambda item: item[0])
        total = sum(value for _, value in normalized)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("posterior probabilities must sum to one")
        object.__setattr__(self, "probabilities", tuple(normalized))

    @classmethod
    def from_mapping(
        cls,
        *,
        start_ms: int,
        end_ms: int,
        probabilities: Mapping[str, float],
    ) -> PosteriorFrame:
        return cls(start_ms, end_ms, tuple(probabilities.items()))

    def probability(self, symbol: str) -> float:
        for candidate, value in self.probabilities:
            if candidate == symbol:
                return value
        return 0.0


@dataclass(frozen=True, slots=True)
class PosteriorSequence:
    """Frozen phone or mora posteriorgram emitted from audio alone."""

    kind: PosteriorKind
    blank_symbol: str
    vocabulary: tuple[str, ...]
    frames: tuple[PosteriorFrame, ...]
    encoder: str
    encoder_revision: str
    label_set_revision: str
    source_audio_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.kind not in {"phone", "mora"}:
            raise ValueError("posterior kind must be 'phone' or 'mora'")
        if not self.blank_symbol:
            raise ValueError("blank_symbol is required")
        vocabulary = tuple(dict.fromkeys(str(value) for value in self.vocabulary))
        if len(vocabulary) != len(self.vocabulary) or any(not value for value in vocabulary):
            raise ValueError("posterior vocabulary must contain unique non-empty symbols")
        if self.blank_symbol not in vocabulary:
            raise ValueError("blank_symbol must be present in the posterior vocabulary")
        if not self.frames:
            raise ValueError("posterior sequence requires at least one frame")
        if not self.encoder or not self.encoder_revision or not self.label_set_revision:
            raise ValueError("encoder and label-set provenance are required")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 hex digest")
        allowed = set(vocabulary)
        previous_start = -1
        for frame in self.frames:
            if frame.start_ms < previous_start:
                raise ValueError("posterior frames must be ordered by start time")
            previous_start = frame.start_ms
            symbols = {symbol for symbol, _ in frame.probabilities}
            if symbols != allowed:
                raise ValueError("every posterior frame must cover the frozen vocabulary exactly")
        object.__setattr__(self, "vocabulary", vocabulary)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "kind": self.kind,
                "blankSymbol": self.blank_symbol,
                "vocabulary": self.vocabulary,
                "frames": self.frames,
                "encoder": self.encoder,
                "encoderRevision": self.encoder_revision,
                "labelSetRevision": self.label_set_revision,
                "sourceAudioSha256": self.source_audio_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class CandidatePronunciation:
    """A candidate-bound phone or mora sequence produced by a frozen G2P/lexicon adapter."""

    candidate_id: str
    text: str
    kind: PosteriorKind
    symbols: tuple[str, ...]
    source_text_sha256: str
    producer: str
    producer_revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text:
            raise ValueError("candidate pronunciation requires candidate_id and text")
        if self.kind not in {"phone", "mora"}:
            raise ValueError("pronunciation kind must be 'phone' or 'mora'")
        if not self.symbols or any(not symbol for symbol in self.symbols):
            raise ValueError("candidate pronunciation requires non-empty symbols")
        if self.source_text_sha256 != _sha256_text(self.text):
            raise ValueError("source_text_sha256 does not match the exact candidate text")
        if not self.producer or not self.producer_revision:
            raise ValueError("pronunciation producer provenance is required")

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        text: str,
        kind: PosteriorKind,
        symbols: Sequence[str],
        producer: str,
        producer_revision: str,
    ) -> CandidatePronunciation:
        return cls(
            candidate_id=candidate_id,
            text=text,
            kind=kind,
            symbols=tuple(symbols),
            source_text_sha256=_sha256_text(text),
            producer=producer,
            producer_revision=producer_revision,
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "candidateId": self.candidate_id,
                "sourceTextSha256": self.source_text_sha256,
                "kind": self.kind,
                "symbols": self.symbols,
                "producer": self.producer,
                "producerRevision": self.producer_revision,
            }
        )


@dataclass(frozen=True, slots=True)
class CTCPronunciationScore:
    """Raw CTC sequence likelihood. It is not a correctness probability."""

    candidate_id: str
    kind: PosteriorKind
    log_likelihood: float
    mean_frame_log_likelihood: float
    target_symbol_count: int
    frame_count: int
    posterior_digest: str
    pronunciation_digest: str
    evidence: EvidenceScore

    def __post_init__(self) -> None:
        values = (
            _strict_float(self.log_likelihood, name="CTC log_likelihood"),
            _strict_float(
                self.mean_frame_log_likelihood,
                name="CTC mean_frame_log_likelihood",
            ),
        )
        if self.target_symbol_count < 1 or self.frame_count < 1:
            raise ValueError("CTC score counts must be positive")
        if any(
            not _is_sha256(value) for value in (self.posterior_digest, self.pronunciation_digest)
        ):
            raise ValueError("CTC score provenance digests must be SHA-256 values")
        if self.evidence.kind != ScoreKind.LOG_LIKELIHOOD:
            raise ValueError("CTC evidence must retain log-likelihood semantics")
        if not math.isclose(self.evidence.value, values[1], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("CTC evidence must equal the mean-frame log likelihood")
        metadata = self.evidence.metadata or {}
        if metadata.get("posteriorDigest") != self.posterior_digest:
            raise ValueError("CTC evidence is not bound to its posterior sequence")
        if metadata.get("pronunciationDigest") != self.pronunciation_digest:
            raise ValueError("CTC evidence is not bound to its candidate pronunciation")


@dataclass(frozen=True, slots=True)
class PhoneticCandidateEvidence:
    candidate_id: str
    phone: CTCPronunciationScore | None = None
    mora: CTCPronunciationScore | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        if self.phone is None and self.mora is None:
            raise ValueError("at least one phone or mora score is required")
        if (
            self.phone is not None
            and (self.phone.candidate_id != self.candidate_id or self.phone.kind != "phone")
        ):
            raise ValueError("phone evidence does not match the candidate")
        if (
            self.mora is not None
            and (self.mora.candidate_id != self.candidate_id or self.mora.kind != "mora")
        ):
            raise ValueError("mora evidence does not match the candidate")

    @property
    def scores(self) -> tuple[EvidenceScore, ...]:
        return tuple(item.evidence for item in (self.phone, self.mora) if item is not None)


def ctc_pronunciation_score(
    posterior: PosteriorSequence,
    pronunciation: CandidatePronunciation,
    *,
    probability_floor: float = 1e-12,
) -> CTCPronunciationScore:
    """Compute ``log P(symbols | audio)`` with the exact CTC forward recurrence."""

    floor = _strict_float(probability_floor, name="probability_floor")
    if not 0.0 < floor < 1.0:
        raise ValueError("probability_floor must be in (0, 1)")
    if posterior.kind != pronunciation.kind:
        raise ValueError("posterior and pronunciation kinds must match")
    if posterior.blank_symbol in pronunciation.symbols:
        raise ValueError("candidate pronunciation must not contain the CTC blank symbol")
    unknown = sorted(set(pronunciation.symbols) - set(posterior.vocabulary))
    if unknown:
        raise ValueError(f"candidate pronunciation contains unknown symbols: {unknown}")

    blank = posterior.blank_symbol
    expanded: list[str] = [blank]
    for symbol in pronunciation.symbols:
        expanded.extend((symbol, blank))

    previous = [-math.inf] * len(expanded)
    first = posterior.frames[0]
    previous[0] = math.log(max(first.probability(blank), floor))
    if len(expanded) > 1:
        previous[1] = math.log(max(first.probability(expanded[1]), floor))

    for frame in posterior.frames[1:]:
        current = [-math.inf] * len(expanded)
        for index, symbol in enumerate(expanded):
            predecessors = [previous[index]]
            if index > 0:
                predecessors.append(previous[index - 1])
            if index > 1 and symbol != blank and symbol != expanded[index - 2]:
                predecessors.append(previous[index - 2])
            current[index] = _logsumexp(predecessors) + math.log(
                max(frame.probability(symbol), floor)
            )
        previous = current

    final_states = previous[-2:] if len(expanded) > 1 else previous
    log_likelihood = _logsumexp(final_states)
    if not math.isfinite(log_likelihood):
        raise ValueError("CTC score is not finite")
    mean_frame = log_likelihood / len(posterior.frames)
    source = (
        f"ctc-{posterior.kind}:{posterior.encoder}@{posterior.encoder_revision}:"
        f"{posterior.label_set_revision}"
    )
    evidence = EvidenceScore(
        value=mean_frame,
        kind=ScoreKind.LOG_LIKELIHOOD,
        source=source,
        calibrated=False,
        higher_is_better=True,
        metadata={
            "candidateId": pronunciation.candidate_id,
            "posteriorDigest": posterior.digest,
            "pronunciationDigest": pronunciation.digest,
            "sourceAudioSha256": posterior.source_audio_sha256,
            "frameCount": len(posterior.frames),
            "targetSymbolCount": len(pronunciation.symbols),
        },
    )
    return CTCPronunciationScore(
        candidate_id=pronunciation.candidate_id,
        kind=posterior.kind,
        log_likelihood=log_likelihood,
        mean_frame_log_likelihood=mean_frame,
        target_symbol_count=len(pronunciation.symbols),
        frame_count=len(posterior.frames),
        posterior_digest=posterior.digest,
        pronunciation_digest=pronunciation.digest,
        evidence=evidence,
    )


def rank_candidate_pronunciations(
    posterior: PosteriorSequence,
    pronunciations: Sequence[CandidatePronunciation],
    *,
    probability_floor: float = 1e-12,
) -> tuple[CTCPronunciationScore, ...]:
    """Rank fixed candidates without generating or rewriting transcript text."""

    if not pronunciations:
        raise ValueError("at least one candidate pronunciation is required")
    if len({row.candidate_id for row in pronunciations}) != len(pronunciations):
        raise ValueError("candidate pronunciation IDs must be unique")
    scores = [
        ctc_pronunciation_score(
            posterior,
            pronunciation,
            probability_floor=probability_floor,
        )
        for pronunciation in pronunciations
    ]
    return tuple(
        sorted(
            scores,
            key=lambda row: (-row.mean_frame_log_likelihood, row.candidate_id),
        )
    )
