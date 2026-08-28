from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Literal

_EPSILON = 1e-9


def _clip_probability(value: float) -> float:
    return min(1.0 - _EPSILON, max(_EPSILON, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _logit(probability: float) -> float:
    value = _clip_probability(probability)
    return math.log(value / (1.0 - value))


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    name: str
    center: float = 0.0
    scale: float = 1.0
    temperature: float = 1.0
    direction: float = 1.0
    input_kind: Literal["score", "probability", "logit"] = "score"
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name is required")
        values = (self.center, self.scale, self.temperature, self.direction)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("calibration profile values must be finite")
        if self.scale <= 0 or self.temperature <= 0:
            raise ValueError("scale and temperature must be positive")
        if self.direction == 0:
            raise ValueError("direction must not be zero")

    def transform(self, value: float | None) -> float | None:
        if value is None or not math.isfinite(float(value)):
            return None
        numeric = float(value)
        if self.input_kind == "probability":
            transformed = _logit(numeric) / self.temperature
        elif self.input_kind == "logit":
            transformed = numeric / self.temperature
        else:
            transformed = self.direction * (numeric - self.center)
            transformed /= self.scale * self.temperature
        return _clip_probability(_sigmoid(transformed))

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def robust_profile(
    values: Iterable[float | None],
    *,
    name: str,
    temperature: float = 1.0,
    minimum_scale: float = 0.05,
) -> CalibrationProfile | None:
    finite = sorted(
        float(value) for value in values if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return None
    center = median(finite)
    deviations = [abs(value - center) for value in finite]
    scale = max(minimum_scale, 1.4826 * median(deviations))
    return CalibrationProfile(
        name=name,
        center=center,
        scale=scale,
        temperature=temperature,
        input_kind="score",
    )


def calibrate_values(
    values: Sequence[float | None],
    *,
    profile: CalibrationProfile | None = None,
    stream_name: str = "evidence",
) -> list[float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return [None] * len(values)
    if profile is not None:
        return [profile.transform(value) for value in values]
    if all(0.0 <= value <= 1.0 for value in finite):
        return [
            None
            if value is None or not math.isfinite(float(value))
            else min(1.0, max(0.0, float(value)))
            for value in values
        ]
    fallback = robust_profile(values, name=f"robust:{stream_name}")
    assert fallback is not None
    return [fallback.transform(value) for value in values]


@dataclass(frozen=True, slots=True)
class ScoreRankFeatures:
    rank: int
    hypothesis_count: int
    avg_logprob: float | None = None
    margin_to_next: float | None = None
    token_count: int | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank is one-based")
        if self.hypothesis_count < self.rank:
            raise ValueError("hypothesis_count must be >= rank")
        if self.token_count is not None and self.token_count < 1:
            raise ValueError("token_count must be positive")


def score_rank_confidence(features: ScoreRankFeatures) -> float:
    if features.avg_logprob is None or not math.isfinite(features.avg_logprob):
        token_probability = 0.5
    else:
        token_probability = min(1.0, max(0.0, math.exp(min(0.0, features.avg_logprob))))
    if features.hypothesis_count <= 1:
        rank_fraction = 1.0
    else:
        rank_fraction = (features.hypothesis_count - features.rank) / (
            features.hypothesis_count - 1
        )
    if features.margin_to_next is None or not math.isfinite(features.margin_to_next):
        margin_confidence = 0.5
    else:
        margin_confidence = _sigmoid(features.margin_to_next / 0.15)
    confidence = 0.55 * token_probability + 0.25 * rank_fraction + 0.20 * margin_confidence
    return min(1.0, max(0.0, confidence))


def negative_log_likelihood(probabilities: Sequence[float], labels: Sequence[int | bool]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    return -sum(
        int(bool(label)) * math.log(_clip_probability(probability))
        + (1 - int(bool(label))) * math.log(1 - _clip_probability(probability))
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(probabilities)


def brier_score(probabilities: Sequence[float], labels: Sequence[int | bool]) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    return sum(
        (_clip_probability(probability) - int(bool(label))) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    bins: int = 15,
) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    total = len(probabilities)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not indices:
            continue
        confidence = sum(probabilities[index] for index in indices) / len(indices)
        accuracy = sum(bool(labels[index]) for index in indices) / len(indices)
        error += len(indices) / total * abs(confidence - accuracy)
    return error


def fit_temperature(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    candidates: Sequence[float] | None = None,
) -> float:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must have equal non-zero length")
    grid = tuple(
        candidates
        or [
            0.25,
            0.35,
            0.5,
            0.7,
            0.85,
            1.0,
            1.2,
            1.5,
            2.0,
            3.0,
            5.0,
        ]
    )
    logits = [_logit(probability) for probability in probabilities]

    def loss(temperature: float) -> float:
        if temperature <= 0 or not math.isfinite(temperature):
            return math.inf
        calibrated = [_sigmoid(logit / temperature) for logit in logits]
        return negative_log_likelihood(calibrated, labels)

    return min(grid, key=lambda temperature: (loss(float(temperature)), float(temperature)))


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[int | bool]
) -> list[tuple[float, float]]:
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must have equal non-zero length")
    ordered = sorted(
        zip(confidences, correct, strict=True),
        key=lambda row: float(row[0]),
        reverse=True,
    )
    points: list[tuple[float, float]] = []
    errors = 0
    for index, (_confidence, is_correct) in enumerate(ordered, 1):
        errors += int(not bool(is_correct))
        points.append((index / len(ordered), errors / index))
    return points


def area_under_risk_coverage(confidences: Sequence[float], correct: Sequence[int | bool]) -> float:
    points = risk_coverage_curve(confidences, correct)
    previous_coverage = 0.0
    previous_risk = 0.0
    area = 0.0
    for coverage, risk in points:
        area += (coverage - previous_coverage) * (risk + previous_risk) / 2
        previous_coverage = coverage
        previous_risk = risk
    return area
