from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceName = Literal[
    "acoustic",
    "mora",
    "lexical",
    "preservation",
    "cross_model",
]
ObservationDecision = Literal["accepted", "provisional"]
NormalizationMode = Literal["deterministic", "rank-only", "guarded-rewrite"]


def _canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonical_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical evidence cannot contain NaN or infinity")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_optional(value: float | None, *, name: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class MoraUnit:
    index: int
    kana: str
    surface: str | None = None
    kind: Literal["regular", "moraic-nasal", "geminate", "long-vowel"] = "regular"
    start_ms: float | None = None
    end_ms: float | None = None
    confidence: float | None = None
    phones: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("mora index must be non-negative")
        if not self.kana:
            raise ValueError("mora kana must not be empty")
        for name, value in (
            ("start_ms", self.start_ms),
            ("end_ms", self.end_ms),
            ("confidence", self.confidence),
        ):
            _finite_optional(value, name=name)
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("mora start_ms must be non-negative")
        if self.end_ms is not None and self.end_ms < 0:
            raise ValueError("mora end_ms must be non-negative")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("mora end_ms must be >= start_ms")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("mora confidence must be in [0, 1]")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("char_start must be non-negative")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must be >= char_start")


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    candidate_id: str
    text: str
    token_ids: tuple[int, ...] = ()
    acoustic: float | None = None
    mora: float | None = None
    lexical: float | None = None
    preservation: float | None = None
    cross_model: float | None = None
    teacher: float | None = None
    reading: str | None = None
    mora_units: tuple[MoraUnit, ...] = ()
    rank: int | None = None
    hypothesis_count: int | None = None
    sequence_score: float | None = None
    avg_logprob: float | None = None
    beam_confidence: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.text:
            raise ValueError("candidate text must not be empty")
        if self.rank is not None and self.rank < 1:
            raise ValueError("candidate rank is one-based")
        if self.hypothesis_count is not None and self.hypothesis_count < 1:
            raise ValueError("hypothesis_count must be positive")
        if (
            self.rank is not None
            and self.hypothesis_count is not None
            and self.hypothesis_count < self.rank
        ):
            raise ValueError("hypothesis_count must be >= rank")
        for name, value in (
            ("acoustic", self.acoustic),
            ("mora", self.mora),
            ("lexical", self.lexical),
            ("preservation", self.preservation),
            ("cross_model", self.cross_model),
            ("teacher", self.teacher),
            ("sequence_score", self.sequence_score),
            ("avg_logprob", self.avg_logprob),
            ("beam_confidence", self.beam_confidence),
        ):
            _finite_optional(value, name=name)
        if self.beam_confidence is not None and not 0 <= self.beam_confidence <= 1:
            raise ValueError("beam_confidence must be in [0, 1]")
        if self.teacher is not None and not 0 <= self.teacher <= 1:
            raise ValueError("teacher probability must be in [0, 1]")

    def score(self, name: EvidenceName) -> float | None:
        return getattr(self, name)

    @property
    def evidence_source(self) -> str:
        return self.source or str(self.metadata.get("adapter") or "unknown")

    @property
    def source_support(self) -> tuple[str, ...]:
        values = self.metadata.get("sourceSupport", ())
        if not isinstance(values, (list, tuple, set)):
            values = ()
        sources = {self.evidence_source, *(str(value) for value in values if str(value))}
        return tuple(sorted(sources))

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> CandidateEvidence:
        allowed = {item.name for item in dataclasses.fields(cls)}
        values = {key: row[key] for key in allowed if key in row}
        values["token_ids"] = tuple(int(value) for value in values.get("token_ids", ()))
        mora_units = []
        for value in values.get("mora_units", ()):
            if isinstance(value, MoraUnit):
                mora_units.append(value)
                continue
            mora_row = dict(value)
            mora_row["phones"] = tuple(mora_row.get("phones", ()))
            mora_units.append(MoraUnit(**mora_row))
        values["mora_units"] = tuple(mora_units)
        values["metadata"] = dict(values.get("metadata", {}))
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GateDecision:
    weights: dict[EvidenceName, float]
    posterior: dict[str, float]
    entropy: float
    disagreement: float
    evidence_coverage: float
    selective_risk: float
    needs_relisten: bool
    abstain: bool
    reasons: tuple[str, ...] = ()
    calibration_digest: str | None = None
    uncertainty: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: CandidateEvidence
    final_score: float
    posterior: float
    calibrated_scores: dict[str, float]
    gate: GateDecision
    grammar_honeytrap_penalty: float = 0.0
    missing_evidence_penalty: float = 0.0
    source_diversity_bonus: float = 0.0


@dataclass(frozen=True, slots=True)
class ObservedTranscript:
    text: str
    selected_candidate_id: str
    candidates: tuple[CandidateEvidence, ...]
    ranked: tuple[RankedCandidate, ...]
    uncertainty_spans: tuple[dict[str, Any], ...]
    source_audio_sha256: str | None
    evidence_sha256: str
    decision: ObservationDecision = "accepted"
    selected_posterior: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        selected: RankedCandidate,
        ranked: list[RankedCandidate],
        uncertainty_spans: list[dict[str, Any]],
        source_audio_sha256: str | None = None,
    ) -> ObservedTranscript:
        candidates = tuple(item.candidate for item in ranked)
        decision: ObservationDecision = "provisional" if selected.gate.abstain else "accepted"
        payload = {
            "text": selected.candidate.text,
            "selectedCandidateId": selected.candidate.candidate_id,
            "candidates": candidates,
            "ranked": tuple(ranked),
            "uncertaintySpans": tuple(uncertainty_spans),
            "sourceAudioSha256": source_audio_sha256,
            "decision": decision,
            "selectedPosterior": selected.posterior,
        }
        return cls(
            text=selected.candidate.text,
            selected_candidate_id=selected.candidate.candidate_id,
            candidates=candidates,
            ranked=tuple(ranked),
            uncertainty_spans=tuple(uncertainty_spans),
            source_audio_sha256=source_audio_sha256,
            evidence_sha256=sha256_json(payload),
            decision=decision,
            selected_posterior=selected.posterior,
        )

    def verify(self) -> None:
        payload = {
            "text": self.text,
            "selectedCandidateId": self.selected_candidate_id,
            "candidates": self.candidates,
            "ranked": self.ranked,
            "uncertaintySpans": self.uncertainty_spans,
            "sourceAudioSha256": self.source_audio_sha256,
            "decision": self.decision,
            "selectedPosterior": self.selected_posterior,
        }
        if sha256_json(payload) != self.evidence_sha256:
            raise ValueError("observed transcript evidence was modified")
        selected = next(
            (
                candidate
                for candidate in self.candidates
                if candidate.candidate_id == self.selected_candidate_id
            ),
            None,
        )
        if selected is None or selected.text != self.text:
            raise ValueError("observed text is detached from selected acoustic evidence")


@dataclass(frozen=True, slots=True)
class NormalizedTranscript:
    text: str
    observed_evidence_sha256: str
    mode: NormalizationMode
    selected_candidate_id: str | None = None
    rejected_edits: tuple[str, ...] = ()
    semantic_change_warnings: tuple[str, ...] = ()

    @classmethod
    def attach(
        cls,
        observed: ObservedTranscript,
        *,
        text: str,
        mode: NormalizationMode,
        selected_candidate_id: str | None = None,
        rejected_edits: tuple[str, ...] = (),
        semantic_change_warnings: tuple[str, ...] = (),
    ) -> NormalizedTranscript:
        result = cls(
            text=text,
            observed_evidence_sha256=observed.evidence_sha256,
            mode=mode,
            selected_candidate_id=selected_candidate_id,
            rejected_edits=rejected_edits,
            semantic_change_warnings=semantic_change_warnings,
        )
        result.verify(observed)
        return result

    def verify(self, observed: ObservedTranscript) -> None:
        observed.verify()
        if self.observed_evidence_sha256 != observed.evidence_sha256:
            raise ValueError("normalization is linked to different observed evidence")
        if self.mode not in {"deterministic", "rank-only", "guarded-rewrite"}:
            raise ValueError("unsupported normalization mode")
        if self.mode == "rank-only" and self.selected_candidate_id is None:
            raise ValueError("rank-only normalization requires a selected candidate ID")
        if self.selected_candidate_id is not None:
            selected = next(
                (
                    candidate
                    for candidate in observed.candidates
                    if candidate.candidate_id == self.selected_candidate_id
                ),
                None,
            )
            if selected is None:
                raise ValueError("normalization selected a candidate outside observed evidence")
            if self.mode == "rank-only" and self.text != selected.text:
                raise ValueError("rank-only normalization cannot rewrite candidate text")
