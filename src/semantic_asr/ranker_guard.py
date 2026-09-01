"""Acoustic safety guard for language-model candidate rerankers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import CandidateEvidence
from .rerankers import CandidateRanker


@dataclass(frozen=True, slots=True)
class AcousticGuardConfig:
    acoustic_temperature: float = 0.20
    ranker_temperature: float = 1.0
    deadband: float = 0.08
    strength: float = 6.0

    def __post_init__(self) -> None:
        if self.acoustic_temperature <= 0 or self.ranker_temperature <= 0:
            raise ValueError("guard temperatures must be positive")
        if self.deadband < 0 or self.strength < 0:
            raise ValueError("guard deadband and strength must be non-negative")


def _softmax(values: Mapping[str, float], temperature: float) -> dict[str, float]:
    maximum = max(values.values())
    mass = {
        key: math.exp(max(-80.0, min(80.0, (value - maximum) / temperature)))
        for key, value in values.items()
    }
    total = sum(mass.values()) or 1.0
    return {key: value / total for key, value in mass.items()}


def _candidate_acoustic_value(candidate: CandidateEvidence) -> float:
    metadata = dict(candidate.metadata)
    aggregate = metadata.get("aggregateAverageLogprob")
    if metadata.get("pathProbabilityMassAggregated") and aggregate is not None:
        try:
            value = float(aggregate)
        except (TypeError, ValueError):
            value = math.nan
        if math.isfinite(value):
            return value
    for raw in (candidate.avg_logprob, candidate.acoustic, candidate.sequence_score):
        if raw is not None and math.isfinite(float(raw)):
            return float(raw)
    if candidate.rank is not None:
        return -0.5 * max(0, candidate.rank - 1)
    return -20.0


class AcousticGuardedRanker:
    """Penalize language preference unsupported by relative acoustic mass.

    The wrapper changes ranker logits, so any calibration profile used after it
    must be fitted on the guarded ranker output rather than on the inner ranker.
    """

    def __init__(
        self,
        inner: CandidateRanker,
        *,
        config: AcousticGuardConfig | None = None,
    ) -> None:
        self.inner = inner
        self.config = config or AcousticGuardConfig()
        self.name = f"acoustic-guarded:{inner.name}"
        self.last_diagnostics: dict[str, dict[str, float]] = {}

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        if len(candidates) < 1:
            raise ValueError("guarded ranker requires candidates")
        raw = {
            str(candidate_id): float(value)
            for candidate_id, value in self.inner.score(
                candidates,
                context=context,
                consensus=consensus,
                contradiction=contradiction,
            ).items()
        }
        identifiers = {candidate.candidate_id for candidate in candidates}
        if set(raw) != identifiers:
            raise ValueError("inner ranker must score every candidate exactly once")
        if any(not math.isfinite(value) for value in raw.values()):
            raise ValueError("inner ranker returned a non-finite score")
        ranker_preference = _softmax(raw, self.config.ranker_temperature)
        acoustic_values = {
            candidate.candidate_id: _candidate_acoustic_value(candidate) for candidate in candidates
        }
        acoustic_mass = _softmax(acoustic_values, self.config.acoustic_temperature)
        guarded: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            candidate_id = candidate.candidate_id
            unsupported = max(
                0.0,
                ranker_preference[candidate_id]
                - acoustic_mass[candidate_id]
                - self.config.deadband,
            )
            penalty = self.config.strength * unsupported
            guarded[candidate_id] = raw[candidate_id] - penalty
            diagnostics[candidate_id] = {
                "innerScore": raw[candidate_id],
                "rankerPreference": ranker_preference[candidate_id],
                "acousticMass": acoustic_mass[candidate_id],
                "unsupportedPreference": unsupported,
                "penalty": penalty,
                "guardedScore": guarded[candidate_id],
            }
        self.last_diagnostics = diagnostics
        return guarded
