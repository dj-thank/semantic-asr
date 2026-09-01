from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .contracts import canonical_json

ArtifactFormat = Literal[
    "pytorch",
    "torchao-int8",
    "torchao-int4",
    "bitsandbytes-int8",
    "bitsandbytes-int4",
    "onnx",
    "ctranslate2",
    "gguf",
]


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    name: str
    source_model: str
    source_revision: str
    artifact_format: ArtifactFormat
    artifact_sha256: str
    tokenizer_sha256: str
    runtime: str
    runtime_version: str
    quantization: dict[str, Any] = field(default_factory=dict)
    calibration_digest: str | None = None
    build_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.source_model or not self.source_revision:
            raise ValueError("artifact name, source model, and revision are required")
        if self.artifact_format not in {
            "pytorch",
            "torchao-int8",
            "torchao-int4",
            "bitsandbytes-int8",
            "bitsandbytes-int4",
            "onnx",
            "ctranslate2",
            "gguf",
        }:
            raise ValueError("unsupported model artifact format")
        for name, value in (
            ("artifact_sha256", self.artifact_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} must be SHA-256 hex")
        for name, value in (
            ("calibration_digest", self.calibration_digest),
            ("build_manifest_sha256", self.build_manifest_sha256),
        ):
            if value is not None and len(value) != 64:
                raise ValueError(f"{name} must be SHA-256 hex when present")
        if not self.runtime or not self.runtime_version:
            raise ValueError("runtime and runtime version are required")

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DeploymentMetrics:
    candidate_top1_accuracy: float
    pairwise_accuracy: float
    semantic_loss: float
    critical_error_rate: float
    calibration_error: float
    aurc: float
    real_time_factor: float
    peak_memory_mb: float
    artifact_size_mb: float
    deterministic_replay_rate: float = 1.0

    def __post_init__(self) -> None:
        bounded = {
            "candidate_top1_accuracy": self.candidate_top1_accuracy,
            "pairwise_accuracy": self.pairwise_accuracy,
            "critical_error_rate": self.critical_error_rate,
            "calibration_error": self.calibration_error,
            "aurc": self.aurc,
            "deterministic_replay_rate": self.deterministic_replay_rate,
        }
        for name, value in bounded.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0, 1]")
        for name, value in (
            ("semantic_loss", self.semantic_loss),
            ("real_time_factor", self.real_time_factor),
            ("peak_memory_mb", self.peak_memory_mb),
            ("artifact_size_mb", self.artifact_size_mb),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DeploymentGatePolicy:
    maximum_top1_accuracy_drop: float = 0.005
    maximum_pairwise_accuracy_drop: float = 0.010
    maximum_semantic_loss_increase: float = 0.010
    maximum_critical_error_increase: float = 0.0
    maximum_calibration_error_increase: float = 0.010
    maximum_aurc_increase: float = 0.010
    maximum_rtf_ratio: float = 1.05
    maximum_memory_ratio: float = 1.05
    maximum_artifact_size_ratio: float = 1.0
    minimum_deterministic_replay_rate: float = 0.999
    require_same_test_manifest: bool = True
    require_calibration_digest: bool = True

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DeploymentEvaluation:
    artifact: ModelArtifactManifest
    metrics: DeploymentMetrics
    test_manifest_sha256: str
    sample_count: int
    group_count: int
    runtime_hardware: str
    repeated_runs: int = 1

    def __post_init__(self) -> None:
        if len(self.test_manifest_sha256) != 64:
            raise ValueError("test manifest digest must be SHA-256 hex")
        if self.sample_count < 1 or self.group_count < 1 or self.repeated_runs < 1:
            raise ValueError("deployment evaluation counts must be positive")
        if not self.runtime_hardware:
            raise ValueError("runtime hardware description is required")

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DeploymentDecision:
    accepted: bool
    baseline_artifact_digest: str
    candidate_artifact_digest: str
    reasons: tuple[str, ...]
    deltas: dict[str, float]
    policy_digest: str


def _ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        return 1.0 if candidate <= 0 else math.inf
    return candidate / baseline


def evaluate_deployment_candidate(
    baseline: DeploymentEvaluation,
    candidate: DeploymentEvaluation,
    *,
    policy: DeploymentGatePolicy | None = None,
) -> DeploymentDecision:
    policy = policy or DeploymentGatePolicy()
    reasons: list[str] = []
    if policy.require_same_test_manifest and (
        baseline.test_manifest_sha256 != candidate.test_manifest_sha256
    ):
        reasons.append("test-manifest-mismatch")
    if baseline.sample_count != candidate.sample_count:
        reasons.append("sample-count-mismatch")
    if baseline.group_count != candidate.group_count:
        reasons.append("group-count-mismatch")
    if baseline.runtime_hardware != candidate.runtime_hardware:
        reasons.append("runtime-hardware-mismatch")
    if baseline.artifact.source_model != candidate.artifact.source_model:
        reasons.append("source-model-mismatch")
    if baseline.artifact.source_revision != candidate.artifact.source_revision:
        reasons.append("source-revision-mismatch")
    if policy.require_calibration_digest:
        if not candidate.artifact.calibration_digest:
            reasons.append("missing-calibration-digest")
        elif (
            baseline.artifact.calibration_digest
            and baseline.artifact.calibration_digest != candidate.artifact.calibration_digest
        ):
            reasons.append("calibration-digest-mismatch")

    baseline_metrics = baseline.metrics
    candidate_metrics = candidate.metrics
    deltas = {
        "top1AccuracyDrop": baseline_metrics.candidate_top1_accuracy
        - candidate_metrics.candidate_top1_accuracy,
        "pairwiseAccuracyDrop": baseline_metrics.pairwise_accuracy
        - candidate_metrics.pairwise_accuracy,
        "semanticLossIncrease": candidate_metrics.semantic_loss - baseline_metrics.semantic_loss,
        "criticalErrorIncrease": candidate_metrics.critical_error_rate
        - baseline_metrics.critical_error_rate,
        "calibrationErrorIncrease": candidate_metrics.calibration_error
        - baseline_metrics.calibration_error,
        "aurcIncrease": candidate_metrics.aurc - baseline_metrics.aurc,
        "rtfRatio": _ratio(candidate_metrics.real_time_factor, baseline_metrics.real_time_factor),
        "memoryRatio": _ratio(candidate_metrics.peak_memory_mb, baseline_metrics.peak_memory_mb),
        "artifactSizeRatio": _ratio(
            candidate_metrics.artifact_size_mb, baseline_metrics.artifact_size_mb
        ),
        "deterministicReplayRate": candidate_metrics.deterministic_replay_rate,
    }
    checks = (
        (
            "top1-accuracy-regression",
            deltas["top1AccuracyDrop"],
            policy.maximum_top1_accuracy_drop,
        ),
        (
            "pairwise-accuracy-regression",
            deltas["pairwiseAccuracyDrop"],
            policy.maximum_pairwise_accuracy_drop,
        ),
        (
            "semantic-loss-regression",
            deltas["semanticLossIncrease"],
            policy.maximum_semantic_loss_increase,
        ),
        (
            "critical-error-regression",
            deltas["criticalErrorIncrease"],
            policy.maximum_critical_error_increase,
        ),
        (
            "calibration-regression",
            deltas["calibrationErrorIncrease"],
            policy.maximum_calibration_error_increase,
        ),
        (
            "aurc-regression",
            deltas["aurcIncrease"],
            policy.maximum_aurc_increase,
        ),
    )
    for reason, value, threshold in checks:
        if value > threshold + 1e-12:
            reasons.append(reason)
    if deltas["rtfRatio"] > policy.maximum_rtf_ratio + 1e-12:
        reasons.append("runtime-slower-than-policy")
    if deltas["memoryRatio"] > policy.maximum_memory_ratio + 1e-12:
        reasons.append("memory-regression")
    if deltas["artifactSizeRatio"] > policy.maximum_artifact_size_ratio + 1e-12:
        reasons.append("artifact-size-regression")
    if candidate_metrics.deterministic_replay_rate < policy.minimum_deterministic_replay_rate:
        reasons.append("deterministic-replay-regression")

    policy_digest = hashlib.sha256(
        json.dumps(
            asdict(policy), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return DeploymentDecision(
        accepted=not reasons,
        baseline_artifact_digest=baseline.artifact.digest,
        candidate_artifact_digest=candidate.artifact.digest,
        reasons=tuple(dict.fromkeys(reasons)),
        deltas=deltas,
        policy_digest=policy_digest,
    )


def deployment_evaluation_from_dict(
    row: Mapping[str, Any],
) -> DeploymentEvaluation:
    artifact = ModelArtifactManifest(**dict(row["artifact"]))
    metrics = DeploymentMetrics(**dict(row["metrics"]))
    return DeploymentEvaluation(
        artifact=artifact,
        metrics=metrics,
        test_manifest_sha256=str(row["test_manifest_sha256"]),
        sample_count=int(row["sample_count"]),
        group_count=int(row["group_count"]),
        runtime_hardware=str(row["runtime_hardware"]),
        repeated_runs=int(row.get("repeated_runs", 1)),
    )
