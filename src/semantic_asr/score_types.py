from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Protocol, Self

_EPS = 1e-12


class ScoreSemantics(StrEnum):
    """Meaning of a numeric score.

    Numeric range alone never determines semantics. In particular, a model-authored
    value in ``[0, 1]`` is still a preference unless it passed a declared calibrator.
    """

    CUMULATIVE_LOG_LIKELIHOOD = "cumulative_log_likelihood"
    AVERAGE_LOG_LIKELIHOOD = "average_log_likelihood"
    LOG_PROBABILITY = "log_probability"
    PROBABILITY = "probability"
    LOGIT = "logit"
    UNCALIBRATED_SCORE = "uncalibrated_score"
    PREFERENCE = "preference"
    LOSS = "loss"
    COST = "cost"


@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    scorer: str
    model: str | None = None
    revision: str | None = None
    runtime: str | None = None
    runtime_version: str | None = None
    configuration_digest: str | None = None
    calibration_digest: str | None = None
    input_evidence_digest: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scorer.strip():
            raise ValueError("scorer is required")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceScore:
    value: float
    semantics: ScoreSemantics
    provenance: ScoreProvenance
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value)):
            raise ValueError("score value must be finite")
        if self.semantics == ScoreSemantics.PROBABILITY:
            if not 0.0 <= float(self.value) <= 1.0:
                raise ValueError("probability must be in [0, 1]")
            if not self.calibrated:
                raise ValueError("probability must be produced by an explicit calibrator")
        elif self.calibrated:
            raise ValueError("only probability scores may be marked calibrated")

    @classmethod
    def raw(
        cls,
        value: float,
        *,
        semantics: ScoreSemantics,
        scorer: str,
        model: str | None = None,
        revision: str | None = None,
        runtime: str | None = None,
        runtime_version: str | None = None,
        configuration_digest: str | None = None,
        input_evidence_digest: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Self:
        if semantics == ScoreSemantics.PROBABILITY:
            raise ValueError("use a fitted calibrator to construct probabilities")
        return cls(
            value=float(value),
            semantics=semantics,
            provenance=ScoreProvenance(
                scorer=scorer,
                model=model,
                revision=revision,
                runtime=runtime,
                runtime_version=runtime_version,
                configuration_digest=configuration_digest,
                input_evidence_digest=input_evidence_digest,
                metadata=dict(metadata or {}),
            ),
        )


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    score: float
    correct: bool
    group: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)):
            raise ValueError("calibration score must be finite")
        if not math.isfinite(float(self.weight)) or self.weight <= 0:
            raise ValueError("calibration weight must be finite and positive")


class FittedCalibrator(Protocol):
    name: str

    def probability(self, score: EvidenceScore) -> EvidenceScore: ...

    @property
    def digest(self) -> str: ...


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(80.0, value))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(-80.0, value))
    return exponent / (1.0 + exponent)


def _clip_probability(value: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(value)))


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """A held-out logistic calibrator fitted without third-party dependencies."""

    slope: float
    intercept: float
    source_semantics: ScoreSemantics
    dataset_digest: str
    iterations: int
    l2: float
    name: str = "platt-v1"

    def __post_init__(self) -> None:
        if self.source_semantics == ScoreSemantics.PROBABILITY:
            raise ValueError("source semantics should describe the uncalibrated input")
        if not all(
            math.isfinite(value)
            for value in (self.slope, self.intercept, float(self.iterations), self.l2)
        ):
            raise ValueError("calibrator parameters must be finite")
        if self.iterations < 1 or self.l2 < 0:
            raise ValueError("invalid calibrator settings")
        if not self.dataset_digest:
            raise ValueError("dataset digest is required")

    @classmethod
    def fit(
        cls,
        examples: list[CalibrationExample],
        *,
        source_semantics: ScoreSemantics,
        dataset_digest: str,
        learning_rate: float = 0.05,
        iterations: int = 2_000,
        l2: float = 1e-4,
    ) -> Self:
        if len(examples) < 4:
            raise ValueError("at least four held-out examples are required")
        if not any(example.correct for example in examples) or all(
            example.correct for example in examples
        ):
            raise ValueError("calibration examples must contain both outcomes")
        if learning_rate <= 0 or iterations < 1 or l2 < 0:
            raise ValueError("invalid optimization settings")

        total_weight = sum(example.weight for example in examples)
        positive = sum(example.weight for example in examples if example.correct)
        intercept = math.log(_clip_probability(positive / total_weight)) - math.log(
            _clip_probability(1.0 - positive / total_weight)
        )
        slope = 0.0

        # Full-batch gradient descent is intentionally deterministic. The feature is
        # standardized so the same learning rate is stable across decoder score domains.
        mean = sum(example.weight * example.score for example in examples) / total_weight
        variance = (
            sum(example.weight * (example.score - mean) ** 2 for example in examples)
            / total_weight
        )
        scale = max(math.sqrt(variance), 1e-6)

        for step in range(iterations):
            grad_slope = 0.0
            grad_intercept = 0.0
            for example in examples:
                x = (example.score - mean) / scale
                prediction = _sigmoid(slope * x + intercept)
                error = prediction - float(example.correct)
                grad_slope += example.weight * error * x
                grad_intercept += example.weight * error
            grad_slope = grad_slope / total_weight + l2 * slope
            grad_intercept /= total_weight
            # Mild inverse-square-root decay avoids oscillation without hidden state.
            rate = learning_rate / math.sqrt(1.0 + step / 200.0)
            slope -= rate * grad_slope
            intercept -= rate * grad_intercept

        # Fold standardization into the persisted affine transform.
        folded_slope = slope / scale
        folded_intercept = intercept - slope * mean / scale
        return cls(
            slope=folded_slope,
            intercept=folded_intercept,
            source_semantics=source_semantics,
            dataset_digest=dataset_digest,
            iterations=iterations,
            l2=l2,
        )

    def probability(self, score: EvidenceScore) -> EvidenceScore:
        if score.semantics != self.source_semantics:
            raise ValueError(
                f"calibrator expects {self.source_semantics}, got {score.semantics}"
            )
        value = _sigmoid(self.slope * score.value + self.intercept)
        provenance = ScoreProvenance(
            scorer=score.provenance.scorer,
            model=score.provenance.model,
            revision=score.provenance.revision,
            runtime=score.provenance.runtime,
            runtime_version=score.provenance.runtime_version,
            configuration_digest=score.provenance.configuration_digest,
            calibration_digest=self.digest,
            input_evidence_digest=score.provenance.input_evidence_digest,
            metadata={
                **score.provenance.metadata,
                "sourceSemantics": score.semantics.value,
                "calibrationDatasetDigest": self.dataset_digest,
                "calibrator": self.name,
            },
        )
        return EvidenceScore(
            value=value,
            semantics=ScoreSemantics.PROBABILITY,
            provenance=provenance,
            calibrated=True,
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "slope": self.slope,
                "intercept": self.intercept,
                "sourceSemantics": self.source_semantics.value,
                "datasetDigest": self.dataset_digest,
                "iterations": self.iterations,
                "l2": self.l2,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    """Monotone non-parametric calibration using pair-adjacent violators."""

    thresholds: tuple[float, ...]
    probabilities: tuple[float, ...]
    source_semantics: ScoreSemantics
    dataset_digest: str
    name: str = "isotonic-pav-v1"

    def __post_init__(self) -> None:
        if not self.thresholds or len(self.thresholds) != len(self.probabilities):
            raise ValueError("thresholds and probabilities must have equal non-zero length")
        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("thresholds must be sorted")
        if any(not 0 <= value <= 1 for value in self.probabilities):
            raise ValueError("isotonic probabilities must be in [0, 1]")
        if any(
            left > right
            for left, right in zip(self.probabilities, self.probabilities[1:], strict=False)
        ):
            raise ValueError("isotonic probabilities must be monotone")

    @classmethod
    def fit(
        cls,
        examples: list[CalibrationExample],
        *,
        source_semantics: ScoreSemantics,
        dataset_digest: str,
    ) -> Self:
        if len(examples) < 4:
            raise ValueError("at least four held-out examples are required")
        ordered = sorted(examples, key=lambda example: (example.score, example.correct))
        # Merge identical x values before PAV.
        blocks: list[list[float]] = []  # [min_x, max_x, weighted_y, weight]
        for example in ordered:
            if blocks and example.score == blocks[-1][1]:
                blocks[-1][2] += example.weight * float(example.correct)
                blocks[-1][3] += example.weight
            else:
                blocks.append(
                    [
                        float(example.score),
                        float(example.score),
                        example.weight * float(example.correct),
                        float(example.weight),
                    ]
                )
        index = 0
        while index < len(blocks) - 1:
            left = blocks[index][2] / blocks[index][3]
            right = blocks[index + 1][2] / blocks[index + 1][3]
            if left <= right:
                index += 1
                continue
            blocks[index] = [
                blocks[index][0],
                blocks[index + 1][1],
                blocks[index][2] + blocks[index + 1][2],
                blocks[index][3] + blocks[index + 1][3],
            ]
            del blocks[index + 1]
            index = max(0, index - 1)

        return cls(
            thresholds=tuple(block[1] for block in blocks),
            probabilities=tuple(block[2] / block[3] for block in blocks),
            source_semantics=source_semantics,
            dataset_digest=dataset_digest,
        )

    def probability(self, score: EvidenceScore) -> EvidenceScore:
        if score.semantics != self.source_semantics:
            raise ValueError(
                f"calibrator expects {self.source_semantics}, got {score.semantics}"
            )
        index = 0
        while index < len(self.thresholds) - 1 and score.value > self.thresholds[index]:
            index += 1
        value = self.probabilities[index]
        provenance = ScoreProvenance(
            scorer=score.provenance.scorer,
            model=score.provenance.model,
            revision=score.provenance.revision,
            runtime=score.provenance.runtime,
            runtime_version=score.provenance.runtime_version,
            configuration_digest=score.provenance.configuration_digest,
            calibration_digest=self.digest,
            input_evidence_digest=score.provenance.input_evidence_digest,
            metadata={
                **score.provenance.metadata,
                "sourceSemantics": score.semantics.value,
                "calibrationDatasetDigest": self.dataset_digest,
                "calibrator": self.name,
            },
        )
        return EvidenceScore(
            value=value,
            semantics=ScoreSemantics.PROBABILITY,
            provenance=provenance,
            calibrated=True,
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "name": self.name,
                "thresholds": self.thresholds,
                "probabilities": self.probabilities,
                "sourceSemantics": self.source_semantics.value,
                "datasetDigest": self.dataset_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def calibration_dataset_digest(examples: list[CalibrationExample]) -> str:
    payload = [
        {
            "score": example.score,
            "correct": example.correct,
            "group": example.group,
            "weight": example.weight,
        }
        for example in examples
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
