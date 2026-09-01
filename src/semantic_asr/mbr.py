from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .candidate_pool import CandidatePool, SurfaceCandidate
from .contracts import CandidateEvidence
from .evaluation import (
    CRITICAL_ENTITY_PATTERN,
    CURRENCY_PATTERN,
    DATE_TIME_PATTERN,
    NEGATION_PATTERN,
    NUMBER_PATTERN,
    critical_entity_sequence,
    date_time_error_rate,
    edit_distance,
    filler_sequence,
    negation_error_rate,
    normalize_characters,
    number_error_rate,
)
from .japanese import mora_sequence

LossFunction = Callable[[CandidateEvidence, CandidateEvidence], float]


@dataclass(frozen=True, slots=True)
class SemanticMBRConfig:
    surface_weight: float = 0.34
    mora_weight: float = 0.30
    critical_weight: float = 0.28
    preservation_weight: float = 0.08
    maximum_loss: float = 2.0

    def __post_init__(self) -> None:
        weights = (
            self.surface_weight,
            self.mora_weight,
            self.critical_weight,
            self.preservation_weight,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("MBR weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one MBR weight must be positive")
        if self.maximum_loss <= 0:
            raise ValueError("maximum_loss must be positive")


@dataclass(frozen=True, slots=True)
class CandidateRisk:
    candidate_id: str
    text: str
    risk: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MBRDecision:
    selected_candidate_id: str
    selected_text: str
    risks: tuple[CandidateRisk, ...]
    posterior: dict[str, float]
    loss_name: str
    expected_risk: float
    risk_margin: float

    @property
    def selected(self) -> CandidateRisk:
        return self.risks[0]


def _normalize_distribution(
    candidates: Sequence[CandidateEvidence],
    posterior: Mapping[str, float] | None,
) -> dict[str, float]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    if posterior is None:
        probability = 1.0 / len(candidates)
        return {candidate_id: probability for candidate_id in identifiers}
    values: dict[str, float] = {}
    for candidate_id in identifiers:
        raw = posterior.get(candidate_id, 0.0)
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("posterior values must be finite and non-negative")
        values[candidate_id] = value
    total = sum(values.values())
    if total <= 0:
        raise ValueError("posterior has no mass on supplied candidates")
    return {candidate_id: value / total for candidate_id, value in values.items()}


def _symmetric_distance(left: Sequence[str], right: Sequence[str]) -> float:
    if not left and not right:
        return 0.0
    return edit_distance(left, right) / max(1, len(left), len(right))


def surface_loss(left: CandidateEvidence, right: CandidateEvidence) -> float:
    return _symmetric_distance(
        normalize_characters(left.text),
        normalize_characters(right.text),
    )


def _candidate_mora(candidate: CandidateEvidence) -> list[str]:
    if candidate.mora_units:
        return [unit.kana for unit in candidate.mora_units]
    if candidate.reading:
        return mora_sequence(candidate.reading)
    return mora_sequence(candidate.text)


def mora_loss(left: CandidateEvidence, right: CandidateEvidence) -> float:
    left_mora = _candidate_mora(left)
    right_mora = _candidate_mora(right)
    # Kanji-only candidates have no trustworthy reading unless an ASR/lexicon
    # supplied one. Treat that case as unavailable mora evidence instead of the
    # previous false zero-loss match.
    if not left_mora or not right_mora:
        return surface_loss(left, right)
    return _symmetric_distance(left_mora, right_mora)


_CRITICAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("currency", CURRENCY_PATTERN),
    ("date-time", DATE_TIME_PATTERN),
    ("number", NUMBER_PATTERN),
    ("negation", NEGATION_PATTERN),
)


def critical_units(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    units: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    # Specific semantic types are collected before the broad entity pattern.
    # Fully-contained generic entities are then suppressed to avoid counting the
    # same number/date/currency token twice.
    for label, pattern in _CRITICAL_PATTERNS:
        for match in pattern.finditer(normalized):
            units.append((match.start(), match.end(), f"{label}:{match.group(0)}"))
            occupied.append((match.start(), match.end()))
    for match in CRITICAL_ENTITY_PATTERN.finditer(normalized):
        start, end = match.span()
        if any(start >= low and end <= high for low, high in occupied):
            continue
        units.append((start, end, f"entity:{match.group(0)}"))
    return [
        value
        for _start, _end, value in sorted(
            units,
            key=lambda item: (item[0], item[1], item[2]),
        )
    ]


def critical_loss(left: CandidateEvidence, right: CandidateEvidence) -> float:
    return _symmetric_distance(critical_units(left.text), critical_units(right.text))


def preservation_loss(left: CandidateEvidence, right: CandidateEvidence) -> float:
    left_score = left.preservation
    right_score = right.preservation
    if left_score is None or right_score is None:
        return 0.0
    return min(1.0, abs(float(left_score) - float(right_score)))


def semantic_loss(
    left: CandidateEvidence,
    right: CandidateEvidence,
    *,
    config: SemanticMBRConfig | None = None,
) -> tuple[float, dict[str, float]]:
    config = config or SemanticMBRConfig()
    components = {
        "surface": surface_loss(left, right),
        "mora": mora_loss(left, right),
        "critical": critical_loss(left, right),
        "preservation": preservation_loss(left, right),
    }
    weights = {
        "surface": config.surface_weight,
        "mora": config.mora_weight,
        "critical": config.critical_weight,
        "preservation": config.preservation_weight,
    }
    normalizer = sum(weights.values())
    total = sum(weights[name] * components[name] for name in components) / normalizer
    return min(config.maximum_loss, max(0.0, total)), components


def minimum_bayes_risk(
    candidates: Sequence[CandidateEvidence],
    *,
    posterior: Mapping[str, float] | None = None,
    loss: LossFunction = surface_loss,
    loss_name: str = "surface",
) -> MBRDecision:
    probabilities = _normalize_distribution(candidates, posterior)
    rows: list[CandidateRisk] = []
    for candidate in candidates:
        expected = 0.0
        for reference in candidates:
            value = float(loss(candidate, reference))
            if not math.isfinite(value) or value < 0:
                raise ValueError("loss function must return finite non-negative values")
            expected += probabilities[reference.candidate_id] * value
        rows.append(
            CandidateRisk(
                candidate_id=candidate.candidate_id,
                text=candidate.text,
                risk=expected,
            )
        )
    rows.sort(key=lambda row: (row.risk, -probabilities[row.candidate_id], row.candidate_id))
    second = rows[1].risk if len(rows) > 1 else rows[0].risk
    return MBRDecision(
        selected_candidate_id=rows[0].candidate_id,
        selected_text=rows[0].text,
        risks=tuple(rows),
        posterior=probabilities,
        loss_name=loss_name,
        expected_risk=rows[0].risk,
        risk_margin=max(0.0, second - rows[0].risk),
    )


def semantic_minimum_bayes_risk(
    candidates: Sequence[CandidateEvidence],
    *,
    posterior: Mapping[str, float] | None = None,
    config: SemanticMBRConfig | None = None,
) -> MBRDecision:
    resolved = config or SemanticMBRConfig()
    probabilities = _normalize_distribution(candidates, posterior)
    rows: list[CandidateRisk] = []
    for candidate in candidates:
        expected = 0.0
        aggregate = {
            "surface": 0.0,
            "mora": 0.0,
            "critical": 0.0,
            "preservation": 0.0,
        }
        for reference in candidates:
            value, components = semantic_loss(candidate, reference, config=resolved)
            probability = probabilities[reference.candidate_id]
            expected += probability * value
            for name, component in components.items():
                aggregate[name] += probability * component
        rows.append(
            CandidateRisk(
                candidate_id=candidate.candidate_id,
                text=candidate.text,
                risk=expected,
                components=aggregate,
            )
        )
    rows.sort(key=lambda row: (row.risk, -probabilities[row.candidate_id], row.candidate_id))
    second = rows[1].risk if len(rows) > 1 else rows[0].risk
    return MBRDecision(
        selected_candidate_id=rows[0].candidate_id,
        selected_text=rows[0].text,
        risks=tuple(rows),
        posterior=probabilities,
        loss_name="semantic-mbr",
        expected_risk=rows[0].risk,
        risk_margin=max(0.0, second - rows[0].risk),
    )


# Typed CandidatePool MBR API used by the reproducible research stack.


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
class PoolCandidateRisk:
    candidate_id: str
    text: str
    expected_risk: float
    posterior: float
    pairwise: dict[str, LossBreakdown] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PoolMBRDecision:
    selected: SurfaceCandidate
    risks: tuple[PoolCandidateRisk, ...]
    posterior: dict[str, float]
    loss_weights_digest: str
    decision_digest: str


def _characters(text: str) -> tuple[str, ...]:
    return tuple(
        character for character in unicodedata.normalize("NFKC", text) if not character.isspace()
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
    number_raw = number_error_rate(truth_text, hypothesis_text)
    date_time_raw = date_time_error_rate(truth_text, hypothesis_text)
    negation_raw = negation_error_rate(truth_text, hypothesis_text)
    number = (
        1.0
        if number_raw is None
        and NUMBER_PATTERN.search(hypothesis_text)
        and not NUMBER_PATTERN.search(truth_text)
        else _optional_rate(number_raw)
    )
    date_time = (
        1.0
        if date_time_raw is None
        and DATE_TIME_PATTERN.search(hypothesis_text)
        and not DATE_TIME_PATTERN.search(truth_text)
        else _optional_rate(date_time_raw)
    )
    negation = (
        1.0
        if negation_raw is None
        and NEGATION_PATTERN.search(hypothesis_text)
        and not NEGATION_PATTERN.search(truth_text)
        else _optional_rate(negation_raw)
    )
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
) -> PoolMBRDecision:
    """Select an existing candidate with minimum posterior expected loss."""

    if not pool.candidates:
        raise ValueError("candidate pool is empty")
    probabilities = _validate_posterior(
        pool.candidates,
        posterior or pool.posterior(),
    )
    resolved_weights = weights or SemanticLossWeights()
    risks: list[PoolCandidateRisk] = []
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
            PoolCandidateRisk(
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
    return PoolMBRDecision(
        selected=selected,
        risks=ordered,
        posterior=probabilities,
        loss_weights_digest=resolved_weights.digest,
        decision_digest=decision_digest,
    )


TypedCandidateRisk = PoolCandidateRisk
TypedMBRDecision = PoolMBRDecision
