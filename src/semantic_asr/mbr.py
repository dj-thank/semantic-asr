from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .candidate_pool import CandidatePool, SurfaceCandidate
from .evaluation import (
    critical_entity_sequence,
    date_time_error_rate,
    filler_sequence,
    negation_error_rate,
    number_error_rate,
)


class CandidateLoss(Protocol):
    def __call__(self, hypothesis: SurfaceCandidate, possible_truth: SurfaceCandidate) -> float: ...


@dataclass(frozen=True, slots=True)
class SemanticLossWeights:
    character: float = 0.44
    mora: float = 0.16
    number: float = 0.10
    date_time: float = 0.06
    negation: float = 0.09
    critical_entity: float = 0.08
    disfluency: float = 0.04
    unsupported_insertion: float = 0.03

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("semantic loss weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one semantic loss weight must be positive")

    @property
    def normalized(self) -> SemanticLossWeights:
        total = sum(asdict(self).values())
        return SemanticLossWeights(**{key: value / total for key, value in asdict(self).items()})

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    character: float
    mora: float
    number: float
    date_time: float
    negation: float
    critical_entity: float
    disfluency: float
    unsupported_insertion: float
    weighted_total: float


@dataclass(frozen=True, slots=True)
class CandidateRisk:
    candidate_id: str
    text: str
    expected_risk: float
    posterior: float
    pairwise: dict[str, LossBreakdown] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MBRDecision:
    selected: SurfaceCandidate
    risks: tuple[CandidateRisk, ...]
    posterior: dict[str, float]
    loss_weights_digest: str
    decision_digest: str


def _characters(text: str) -> tuple[str, ...]:
    return tuple(
        character
        for character in unicodedata.normalize("NFKC", text)
        if not character.isspace()
    )


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return min(1.0, _edit_distance(reference, hypothesis) / len(reference))


def _optional_rate(value: float | None) -> float:
    return 0.0 if value is None else min(1.0, max(0.0, float(value)))


def _reading(candidate: SurfaceCandidate) -> tuple[str, ...] | None:
    readings: list[str] = []
    for path in candidate.paths:
        value = path.metadata.get("reading") or path.metadata.get("moraReading")
        if value:
            readings.append(str(value))
    if not readings:
        return None
    # The best path is first. Retain its declared reading rather than inventing one.
    return _characters(readings[0])


def _entity_rate(reference: str, hypothesis: str) -> float:
    return _rate(
        tuple(critical_entity_sequence(reference)),
        tuple(critical_entity_sequence(hypothesis)),
    )


def _disfluency_rate(reference: str, hypothesis: str) -> float:
    return _rate(tuple(filler_sequence(reference)), tuple(filler_sequence(hypothesis)))


def _unsupported_insertion_rate(reference: str, hypothesis: str) -> float:
    reference_chars = _characters(reference)
    hypothesis_chars = _characters(hypothesis)
    if not hypothesis_chars:
        return 0.0
    # Directional insertion pressure. Exact insertion attribution is evaluated against
    # references later; MBR uses this conservative proxy only as one small component.
    excess = max(0, len(hypothesis_chars) - len(reference_chars))
    return excess / len(hypothesis_chars)


def semantic_pairwise_loss(
    hypothesis: SurfaceCandidate,
    possible_truth: SurfaceCandidate,
    *,
    weights: SemanticLossWeights | None = None,
) -> LossBreakdown:
    weights = (weights or SemanticLossWeights()).normalized
    hypothesis_text = hypothesis.text
    truth_text = possible_truth.text
    character = _rate(_characters(truth_text), _characters(hypothesis_text))
    truth_reading = _reading(possible_truth)
    hypothesis_reading = _reading(hypothesis)
    mora = (
        _rate(truth_reading, hypothesis_reading)
        if truth_reading is not None and hypothesis_reading is not None
        else character
    )
    number = _optional_rate(number_error_rate(truth_text, hypothesis_text))
    date_time = _optional_rate(date_time_error_rate(truth_text, hypothesis_text))
    negation = _optional_rate(negation_error_rate(truth_text, hypothesis_text))
    critical_entity = _entity_rate(truth_text, hypothesis_text)
    disfluency = _disfluency_rate(truth_text, hypothesis_text)
    unsupported_insertion = _unsupported_insertion_rate(truth_text, hypothesis_text)
    weighted_total = (
        weights.character * character
        + weights.mora * mora
        + weights.number * number
        + weights.date_time * date_time
        + weights.negation * negation
        + weights.critical_entity * critical_entity
        + weights.disfluency * disfluency
        + weights.unsupported_insertion * unsupported_insertion
    )
    return LossBreakdown(
        character=character,
        mora=mora,
        number=number,
        date_time=date_time,
        negation=negation,
        critical_entity=critical_entity,
        disfluency=disfluency,
        unsupported_insertion=unsupported_insertion,
        weighted_total=weighted_total,
    )


def _validate_posterior(
    candidates: Sequence[SurfaceCandidate], posterior: Mapping[str, float]
) -> dict[str, float]:
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if set(posterior) != candidate_ids:
        missing = candidate_ids - set(posterior)
        extra = set(posterior) - candidate_ids
        raise ValueError(f"posterior IDs mismatch; missing={missing}, extra={extra}")
    values = {candidate_id: float(value) for candidate_id, value in posterior.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("posterior values must be finite and non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("posterior mass must be positive")
    return {candidate_id: value / total for candidate_id, value in values.items()}


def decode_mbr(
    pool: CandidatePool,
    *,
    posterior: Mapping[str, float] | None = None,
    weights: SemanticLossWeights | None = None,
    loss_fn: Callable[[SurfaceCandidate, SurfaceCandidate], float | LossBreakdown] | None = None,
) -> MBRDecision:
    """Select an existing candidate with minimum posterior expected loss."""

    if not pool.candidates:
        raise ValueError("candidate pool is empty")
    probabilities = _validate_posterior(
        pool.candidates,
        posterior or pool.posterior(),
    )
    resolved_weights = weights or SemanticLossWeights()
    risks: list[CandidateRisk] = []
    for hypothesis in pool.candidates:
        expected = 0.0
        pairwise: dict[str, LossBreakdown] = {}
        for possible_truth in pool.candidates:
            if loss_fn is None:
                breakdown = semantic_pairwise_loss(
                    hypothesis,
                    possible_truth,
                    weights=resolved_weights,
                )
            else:
                raw = loss_fn(hypothesis, possible_truth)
                if isinstance(raw, LossBreakdown):
                    breakdown = raw
                else:
                    value = min(1.0, max(0.0, float(raw)))
                    breakdown = LossBreakdown(
                        character=value,
                        mora=0.0,
                        number=0.0,
                        date_time=0.0,
                        negation=0.0,
                        critical_entity=0.0,
                        disfluency=0.0,
                        unsupported_insertion=0.0,
                        weighted_total=value,
                    )
            pairwise[possible_truth.candidate_id] = breakdown
            expected += probabilities[possible_truth.candidate_id] * breakdown.weighted_total
        risks.append(
            CandidateRisk(
                candidate_id=hypothesis.candidate_id,
                text=hypothesis.text,
                expected_risk=expected,
                posterior=probabilities[hypothesis.candidate_id],
                pairwise=pairwise,
            )
        )
    ordered = tuple(
        sorted(
            risks,
            key=lambda item: (
                item.expected_risk,
                -item.posterior,
                item.candidate_id,
            ),
        )
    )
    selected = next(
        candidate
        for candidate in pool.candidates
        if candidate.candidate_id == ordered[0].candidate_id
    )
    payload = {
        "selected": selected.candidate_id,
        "posterior": probabilities,
        "risks": [
            {
                "candidateId": risk.candidate_id,
                "expectedRisk": risk.expected_risk,
                "posterior": risk.posterior,
            }
            for risk in ordered
        ],
        "lossWeightsDigest": resolved_weights.digest,
    }
    decision_digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return MBRDecision(
        selected=selected,
        risks=ordered,
        posterior=probabilities,
        loss_weights_digest=resolved_weights.digest,
        decision_digest=decision_digest,
    )
