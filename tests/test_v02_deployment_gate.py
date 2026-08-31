from __future__ import annotations

from semantic_asr.deployment_gate import (
    DeploymentEvaluation,
    DeploymentGatePolicy,
    DeploymentMetrics,
    ModelArtifactManifest,
    evaluate_deployment_candidate,
)


def _artifact(name: str, artifact_format: str, artifact_size: float) -> DeploymentEvaluation:
    manifest = ModelArtifactManifest(
        name=name,
        source_model="Qwen/Qwen3-Reranker-0.6B",
        source_revision="fixture-revision",
        artifact_format=artifact_format,
        artifact_sha256=("a" if artifact_format == "pytorch" else "b") * 64,
        tokenizer_sha256="c" * 64,
        runtime="transformers",
        runtime_version="5.9.0",
        quantization={} if artifact_format == "pytorch" else {"groupSize": 128},
        calibration_digest="d" * 64,
        build_manifest_sha256="e" * 64,
    )
    return DeploymentEvaluation(
        artifact=manifest,
        metrics=DeploymentMetrics(
            candidate_top1_accuracy=0.900,
            pairwise_accuracy=0.940,
            semantic_loss=0.100,
            critical_error_rate=0.020,
            calibration_error=0.030,
            aurc=0.070,
            real_time_factor=0.20,
            peak_memory_mb=1_000.0,
            artifact_size_mb=artifact_size,
            deterministic_replay_rate=1.0,
        ),
        test_manifest_sha256="f" * 64,
        sample_count=2_000,
        group_count=250,
        runtime_hardware="Intel i7 fixture",
        repeated_runs=3,
    )


def test_quantized_candidate_is_accepted_when_quality_is_preserved() -> None:
    baseline = _artifact("fp32", "pytorch", 1_200.0)
    candidate = _artifact("int4", "torchao-int4", 350.0)
    candidate = DeploymentEvaluation(
        artifact=candidate.artifact,
        metrics=DeploymentMetrics(
            candidate_top1_accuracy=0.899,
            pairwise_accuracy=0.936,
            semantic_loss=0.104,
            critical_error_rate=0.020,
            calibration_error=0.035,
            aurc=0.076,
            real_time_factor=0.16,
            peak_memory_mb=420.0,
            artifact_size_mb=350.0,
            deterministic_replay_rate=1.0,
        ),
        test_manifest_sha256=candidate.test_manifest_sha256,
        sample_count=candidate.sample_count,
        group_count=candidate.group_count,
        runtime_hardware=candidate.runtime_hardware,
        repeated_runs=3,
    )
    decision = evaluate_deployment_candidate(baseline, candidate)
    assert decision.accepted
    assert decision.reasons == ()
    assert decision.deltas["artifactSizeRatio"] < 1.0
    assert decision.deltas["memoryRatio"] < 1.0


def test_critical_error_regression_blocks_smaller_artifact() -> None:
    baseline = _artifact("fp32", "pytorch", 1_200.0)
    candidate = _artifact("int4", "torchao-int4", 300.0)
    candidate = DeploymentEvaluation(
        artifact=candidate.artifact,
        metrics=DeploymentMetrics(
            candidate_top1_accuracy=0.900,
            pairwise_accuracy=0.940,
            semantic_loss=0.100,
            critical_error_rate=0.025,
            calibration_error=0.030,
            aurc=0.070,
            real_time_factor=0.10,
            peak_memory_mb=300.0,
            artifact_size_mb=300.0,
            deterministic_replay_rate=1.0,
        ),
        test_manifest_sha256=candidate.test_manifest_sha256,
        sample_count=candidate.sample_count,
        group_count=candidate.group_count,
        runtime_hardware=candidate.runtime_hardware,
    )
    decision = evaluate_deployment_candidate(baseline, candidate)
    assert not decision.accepted
    assert "critical-error-regression" in decision.reasons


def test_manifest_or_hardware_mismatch_blocks_comparison() -> None:
    baseline = _artifact("fp32", "pytorch", 1_200.0)
    candidate = _artifact("int8", "torchao-int8", 600.0)
    candidate = DeploymentEvaluation(
        artifact=candidate.artifact,
        metrics=candidate.metrics,
        test_manifest_sha256="0" * 64,
        sample_count=candidate.sample_count,
        group_count=candidate.group_count,
        runtime_hardware="different hardware",
    )
    decision = evaluate_deployment_candidate(baseline, candidate)
    assert not decision.accepted
    assert "test-manifest-mismatch" in decision.reasons
    assert "runtime-hardware-mismatch" in decision.reasons


def test_policy_can_require_strict_deterministic_replay() -> None:
    baseline = _artifact("fp32", "pytorch", 1_200.0)
    candidate = _artifact("int8", "torchao-int8", 600.0)
    candidate = DeploymentEvaluation(
        artifact=candidate.artifact,
        metrics=DeploymentMetrics(
            candidate_top1_accuracy=0.900,
            pairwise_accuracy=0.940,
            semantic_loss=0.100,
            critical_error_rate=0.020,
            calibration_error=0.030,
            aurc=0.070,
            real_time_factor=0.15,
            peak_memory_mb=600.0,
            artifact_size_mb=600.0,
            deterministic_replay_rate=0.995,
        ),
        test_manifest_sha256=candidate.test_manifest_sha256,
        sample_count=candidate.sample_count,
        group_count=candidate.group_count,
        runtime_hardware=candidate.runtime_hardware,
    )
    decision = evaluate_deployment_candidate(
        baseline,
        candidate,
        policy=DeploymentGatePolicy(minimum_deterministic_replay_rate=0.999),
    )
    assert not decision.accepted
    assert "deterministic-replay-regression" in decision.reasons
