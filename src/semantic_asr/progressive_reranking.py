from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import CandidateEvidence
from .rerankers import CandidateRanker


@dataclass(frozen=True, slots=True)
class ProgressiveStage:
    name: str
    ranker: CandidateRanker
    estimated_cost_ms: int
    weight: float = 1.0
    minimum_margin: float = 0.25
    maximum_entropy: float = 0.72

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("progressive stage name is required")
        if self.estimated_cost_ms < 0:
            raise ValueError("progressive stage cost must be non-negative")
        if self.weight <= 0 or not math.isfinite(self.weight):
            raise ValueError("progressive stage weight must be finite and positive")
        if not 0 <= self.minimum_margin <= 1 or not 0 <= self.maximum_entropy <= 1:
            raise ValueError("progressive confidence thresholds must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ProgressiveStageResult:
    stage: str
    raw_scores: dict[str, float]
    preference_distribution: dict[str, float]
    cumulative_preference: dict[str, float]
    top_candidate_id: str
    margin: float
    entropy: float
    estimated_cost_ms: int


@dataclass(frozen=True, slots=True)
class ProgressiveRerankDecision:
    selected_candidate_id: str
    selected_text: str
    preference_distribution: dict[str, float]
    stages: tuple[ProgressiveStageResult, ...]
    used_budget_ms: int
    early_exit: bool
    stopping_reason: str
    calibrated_probability: bool = False


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(max(-80.0, min(80.0, value - maximum))) for value in values]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    raw = -sum(value * math.log(value + 1e-12) for value in probabilities)
    return min(1.0, max(0.0, raw / math.log(len(probabilities))))


def _normalize_scores(
    candidates: Sequence[CandidateEvidence], scores: Mapping[str, float]
) -> dict[str, float]:
    identifiers = [candidate.candidate_id for candidate in candidates]
    if set(scores) != set(identifiers):
        raise ValueError("progressive ranker must score every candidate ID exactly once")
    values = [float(scores[candidate_id]) for candidate_id in identifiers]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("progressive ranker scores must be finite")
    probabilities = _softmax(values)
    return dict(zip(identifiers, probabilities, strict=True))


def _log_opinion_pool(
    distributions: Sequence[tuple[Mapping[str, float], float]],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    if not distributions:
        uniform = 1.0 / len(candidate_ids)
        return {candidate_id: uniform for candidate_id in candidate_ids}
    logits = {
        candidate_id: sum(
            weight * math.log(max(1e-12, distribution[candidate_id]))
            for distribution, weight in distributions
        )
        for candidate_id in candidate_ids
    }
    values = _softmax([logits[candidate_id] for candidate_id in candidate_ids])
    return dict(zip(candidate_ids, values, strict=True))


def _confidence(distribution: Mapping[str, float]) -> tuple[str, float, float]:
    ordered = sorted(distribution.items(), key=lambda row: (-row[1], row[0]))
    top_id, top_value = ordered[0]
    second_value = ordered[1][1] if len(ordered) > 1 else 0.0
    return top_id, top_value - second_value, _entropy([value for _, value in ordered])


def progressive_rerank(
    candidates: Sequence[CandidateEvidence],
    stages: Sequence[ProgressiveStage],
    *,
    budget_ms: int,
    context: str = "",
    consensus: str = "",
    contradiction: str = "",
    minimum_stages: int = 1,
) -> ProgressiveRerankDecision:
    if len(candidates) < 2:
        raise ValueError("progressive reranking requires at least two candidates")
    if not stages:
        raise ValueError("progressive reranking requires at least one stage")
    if budget_ms < 0 or minimum_stages < 1:
        raise ValueError("invalid progressive reranking budget")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    distributions: list[tuple[Mapping[str, float], float]] = []
    results: list[ProgressiveStageResult] = []
    used = 0
    early_exit = False
    stopping_reason = "all-stages-completed"
    cumulative = {candidate_id: 1.0 / len(candidate_ids) for candidate_id in candidate_ids}

    for stage in stages:
        if used + stage.estimated_cost_ms > budget_ms:
            stopping_reason = "budget-frontier"
            break
        raw_scores = dict(
            stage.ranker.score(
                candidates,
                context=context,
                consensus=consensus,
                contradiction=contradiction,
            )
        )
        distribution = _normalize_scores(candidates, raw_scores)
        distributions.append((distribution, stage.weight))
        cumulative = _log_opinion_pool(distributions, candidate_ids)
        top_id, margin, entropy = _confidence(cumulative)
        used += stage.estimated_cost_ms
        results.append(
            ProgressiveStageResult(
                stage=stage.name,
                raw_scores={key: float(value) for key, value in raw_scores.items()},
                preference_distribution=distribution,
                cumulative_preference=cumulative,
                top_candidate_id=top_id,
                margin=margin,
                entropy=entropy,
                estimated_cost_ms=stage.estimated_cost_ms,
            )
        )
        if (
            len(results) >= minimum_stages
            and margin >= stage.minimum_margin
            and entropy <= stage.maximum_entropy
        ):
            early_exit = True
            stopping_reason = f"confident-after:{stage.name}"
            break

    if not results:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.rank is None,
                candidate.rank if candidate.rank is not None else 10**9,
                candidate.candidate_id,
            ),
        )
        selected = ordered[0]
        uniform = 1.0 / len(candidates)
        return ProgressiveRerankDecision(
            selected_candidate_id=selected.candidate_id,
            selected_text=selected.text,
            preference_distribution={
                candidate.candidate_id: uniform for candidate in candidates
            },
            stages=(),
            used_budget_ms=0,
            early_exit=False,
            stopping_reason="no-stage-fit-budget",
        )

    selected_id, _margin, _entropy_value = _confidence(cumulative)
    selected = by_id[selected_id]
    return ProgressiveRerankDecision(
        selected_candidate_id=selected_id,
        selected_text=selected.text,
        preference_distribution=cumulative,
        stages=tuple(results),
        used_budget_ms=used,
        early_exit=early_exit,
        stopping_reason=stopping_reason,
    )
