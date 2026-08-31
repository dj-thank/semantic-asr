from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .contracts import CandidateEvidence
from .evaluation import (
    CURRENCY_PATTERN,
    DATE_TIME_PATTERN,
    NEGATION_PATTERN,
    NUMBER_PATTERN,
    critical_entity_sequence,
    edit_distance,
    normalize_characters,
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
    return _symmetric_distance(_candidate_mora(left), _candidate_mora(right))


_CRITICAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("currency", CURRENCY_PATTERN),
    ("date-time", DATE_TIME_PATTERN),
    ("number", NUMBER_PATTERN),
    ("negation", NEGATION_PATTERN),
)


def critical_units(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    units: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for label, pattern in _CRITICAL_PATTERNS:
        for match in pattern.finditer(normalized):
            units.append((match.start(), f"{label}:{match.group(0)}"))
            occupied.append((match.start(), match.end()))
    for value in critical_entity_sequence(normalized):
        start = normalized.find(value)
        if start < 0:
            continue
        end = start + len(value)
        if any(start >= low and end <= high for low, high in occupied):
            continue
        units.append((start, f"entity:{value}"))
    return [value for _index, value in sorted(units, key=lambda item: (item[0], item[1]))]


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
        aggregate = {"surface": 0.0, "mora": 0.0, "critical": 0.0, "preservation": 0.0}
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
