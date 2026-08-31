from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .calibration import brier_score, expected_calibration_error, negative_log_likelihood
from .contracts import canonical_json

CalibrationSplit = Literal["calibration"]


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True, slots=True)
class RankerCalibrationSample:
    sample_id: str
    group_id: str
    score: float
    correct: bool
    split: CalibrationSplit = "calibration"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise ValueError("calibration sample and group IDs are required")
        if not math.isfinite(float(self.score)):
            raise ValueError("calibration score must be finite")
        if self.split != "calibration":
            raise ValueError("ranker calibration may consume only the calibration split")


@dataclass(frozen=True, slots=True)
class RankerCalibrationMetrics:
    negative_log_likelihood: float
    brier: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class RankerCalibrationProfile:
    name: str
    source_ranker: str
    slope: float
    intercept: float
    sample_count: int
    group_count: int
    calibration_manifest_sha256: str
    l2: float = 1e-3
    split_name: str = "calibration"
    method: str = "monotonic-platt"
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.source_ranker:
            raise ValueError("calibration profile name and source ranker are required")
        if not math.isfinite(self.slope) or self.slope <= 0:
            raise ValueError("calibration slope must be positive and finite")
        if not math.isfinite(self.intercept):
            raise ValueError("calibration intercept must be finite")
        if self.sample_count < 1 or self.group_count < 1:
            raise ValueError("calibration sample and group counts must be positive")
        if len(self.calibration_manifest_sha256) != 64:
            raise ValueError("calibration manifest digest must be SHA-256 hex")
        if self.l2 < 0 or not math.isfinite(self.l2):
            raise ValueError("calibration regularization must be finite and non-negative")
        if self.split_name != "calibration":
            raise ValueError("ranker calibration profile must be fitted on calibration split")

    def transform(self, value: float | None) -> float | None:
        if value is None or not math.isfinite(float(value)):
            return None
        return _clip_probability(_sigmoid(self.slope * float(value) + self.intercept))

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "digest": self.digest}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> RankerCalibrationProfile:
        values = dict(row)
        values.pop("digest", None)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RankerCalibrationResult:
    profile: RankerCalibrationProfile
    before: RankerCalibrationMetrics
    after: RankerCalibrationMetrics
    iterations: int
    converged: bool


def _metrics(probabilities: Sequence[float], labels: Sequence[bool]) -> RankerCalibrationMetrics:
    return RankerCalibrationMetrics(
        negative_log_likelihood=negative_log_likelihood(probabilities, labels),
        brier=brier_score(probabilities, labels),
        expected_calibration_error=expected_calibration_error(probabilities, labels),
    )


def _manifest_digest(samples: Sequence[RankerCalibrationSample]) -> str:
    payload = [
        {
            "sampleId": sample.sample_id,
            "groupId": sample.group_id,
            "score": sample.score,
            "correct": sample.correct,
            "split": sample.split,
        }
        for sample in sorted(samples, key=lambda row: (row.group_id, row.sample_id))
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _objective(
    normalized_scores: Sequence[float],
    labels: Sequence[float],
    *,
    slope: float,
    intercept: float,
    l2: float,
) -> float:
    probabilities = [
        _clip_probability(_sigmoid(slope * score + intercept))
        for score in normalized_scores
    ]
    nll = -sum(
        label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability)
        for probability, label in zip(probabilities, labels, strict=True)
    ) / len(labels)
    return nll + 0.5 * l2 * slope * slope


def fit_ranker_calibration(
    samples: Sequence[RankerCalibrationSample],
    *,
    name: str,
    source_ranker: str,
    l2: float = 1e-3,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
    minimum_samples: int = 8,
    minimum_groups: int = 2,
) -> RankerCalibrationResult:
    if len(samples) < minimum_samples:
        raise ValueError(f"at least {minimum_samples} calibration samples are required")
    if any(sample.split != "calibration" for sample in samples):
        raise ValueError("training and test samples are forbidden during calibration fitting")
    groups = {sample.group_id for sample in samples}
    if len(groups) < minimum_groups:
        raise ValueError(f"at least {minimum_groups} calibration groups are required")
    labels_bool = [bool(sample.correct) for sample in samples]
    if len(set(labels_bool)) < 2:
        raise ValueError("calibration requires both correct and incorrect examples")
    if l2 < 0 or maximum_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid calibration optimizer configuration")

    raw_scores = [float(sample.score) for sample in samples]
    labels = [float(value) for value in labels_bool]
    mean = sum(raw_scores) / len(raw_scores)
    variance = sum((value - mean) ** 2 for value in raw_scores) / len(raw_scores)
    scale = max(1e-6, math.sqrt(variance))
    normalized = [(value - mean) / scale for value in raw_scores]

    positive_rate = _clip_probability(sum(labels) / len(labels))
    slope = 1.0
    intercept = math.log(positive_rate / (1.0 - positive_rate))
    converged = False
    iterations = 0

    for iteration in range(1, maximum_iterations + 1):
        probabilities = [
            _clip_probability(_sigmoid(slope * score + intercept)) for score in normalized
        ]
        weights = [probability * (1.0 - probability) for probability in probabilities]
        gradient_slope = (
            sum(
                (probability - label) * score
                for probability, label, score in zip(
                    probabilities, labels, normalized, strict=True
                )
            )
            / len(labels)
            + l2 * slope
        )
        gradient_intercept = sum(
            probability - label
            for probability, label in zip(probabilities, labels, strict=True)
        ) / len(labels)
        hessian_ss = (
            sum(weight * score * score for weight, score in zip(weights, normalized, strict=True))
            / len(labels)
            + l2
            + 1e-9
        )
        hessian_si = sum(
            weight * score for weight, score in zip(weights, normalized, strict=True)
        ) / len(labels)
        hessian_ii = sum(weights) / len(labels) + 1e-9
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if determinant <= 1e-12:
            delta_slope = gradient_slope
            delta_intercept = gradient_intercept
        else:
            delta_slope = (
                hessian_ii * gradient_slope - hessian_si * gradient_intercept
            ) / determinant
            delta_intercept = (
                -hessian_si * gradient_slope + hessian_ss * gradient_intercept
            ) / determinant

        current_loss = _objective(
            normalized,
            labels,
            slope=slope,
            intercept=intercept,
            l2=l2,
        )
        step = 1.0
        accepted = False
        next_slope = slope
        next_intercept = intercept
        for _backtrack in range(24):
            candidate_slope = max(1e-6, slope - step * delta_slope)
            candidate_intercept = intercept - step * delta_intercept
            candidate_loss = _objective(
                normalized,
                labels,
                slope=candidate_slope,
                intercept=candidate_intercept,
                l2=l2,
            )
            if candidate_loss <= current_loss + 1e-12:
                next_slope = candidate_slope
                next_intercept = candidate_intercept
                accepted = True
                break
            step *= 0.5
        if not accepted:
            next_slope = max(1e-6, slope - 0.01 * gradient_slope)
            next_intercept = intercept - 0.01 * gradient_intercept

        iterations = iteration
        movement = max(abs(next_slope - slope), abs(next_intercept - intercept))
        slope = next_slope
        intercept = next_intercept
        if movement <= tolerance:
            converged = True
            break

    raw_slope = slope / scale
    raw_intercept = intercept - slope * mean / scale
    profile = RankerCalibrationProfile(
        name=name,
        source_ranker=source_ranker,
        slope=raw_slope,
        intercept=raw_intercept,
        sample_count=len(samples),
        group_count=len(groups),
        calibration_manifest_sha256=_manifest_digest(samples),
        l2=float(l2),
    )
    before_probabilities = [_clip_probability(_sigmoid(score)) for score in raw_scores]
    after_probabilities = [profile.transform(score) for score in raw_scores]
    assert all(probability is not None for probability in after_probabilities)
    return RankerCalibrationResult(
        profile=profile,
        before=_metrics(before_probabilities, labels_bool),
        after=_metrics(
            [float(probability) for probability in after_probabilities], labels_bool
        ),
        iterations=iterations,
        converged=converged,
    )


def calibration_sample_from_row(
    row: Mapping[str, Any], *, line_number: int = 0
) -> RankerCalibrationSample:
    split = str(row.get("split") or "calibration")
    if split != "calibration":
        raise ValueError(
            f"ranker calibration row {line_number} belongs to forbidden split {split!r}"
        )
    return RankerCalibrationSample(
        sample_id=str(row.get("sampleId") or row.get("sample_id") or line_number),
        group_id=str(row.get("groupId") or row.get("group_id") or ""),
        score=float(row["score"]),
        correct=bool(row["correct"]),
    )


def load_calibration_samples(path: str | Path) -> list[RankerCalibrationSample]:
    output: list[RankerCalibrationSample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"ranker calibration row {line_number} must be an object")
        output.append(calibration_sample_from_row(payload, line_number=line_number))
    if not output:
        raise ValueError("ranker calibration dataset is empty")
    return output


def write_calibration_result(
    result: RankerCalibrationResult, path: str | Path
) -> None:
    payload = {
        "schemaVersion": "ranker-calibration-v1",
        "profile": result.profile.as_dict(),
        "before": asdict(result.before),
        "after": asdict(result.after),
        "iterations": result.iterations,
        "converged": result.converged,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
