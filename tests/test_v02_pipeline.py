from __future__ import annotations

import pytest

from semantic_asr.pipeline import (
    ArtifactEnvelope,
    FunctionStage,
    FunctionalPipeline,
    PipelineContext,
    StageSpec,
    effort_profile,
)


def test_functional_pipeline_preserves_digest_lineage() -> None:
    initial = ArtifactEnvelope.create(kind="numbers", payload=[1, 2, 3])
    double = FunctionStage(
        StageSpec(
            name="double",
            input_kind="numbers",
            output_kind="numbers-doubled",
            estimated_cost_ms=3,
        ),
        lambda artifact, _context: (
            [value * 2 for value in artifact.payload],
            2,
            {"count": len(artifact.payload)},
        ),
    )
    summarize = FunctionStage(
        StageSpec(
            name="sum",
            input_kind="numbers-doubled",
            output_kind="summary",
            estimated_cost_ms=2,
        ),
        lambda artifact, _context: (
            {"sum": sum(artifact.payload)},
            1,
            {},
        ),
    )
    state = FunctionalPipeline([double, summarize]).run(
        initial,
        context=PipelineContext(
            run_id="test-run",
            effort="cpu-quality",
            total_budget_ms=10,
        ),
    )
    assert not state.stopped
    assert state.artifact.payload == {"sum": 12}
    assert state.used_budget_ms == 3
    assert len(state.history) == 2
    assert state.history[0].input_sha256 == initial.sha256
    assert state.history[1].input_sha256 == state.history[0].output.sha256
    state.artifact.verify()


def test_pipeline_rejects_contract_mismatch() -> None:
    first = FunctionStage(
        StageSpec(name="first", input_kind="a", output_kind="b"),
        lambda artifact, _context: (artifact.payload, 0, {}),
    )
    second = FunctionStage(
        StageSpec(name="second", input_kind="c", output_kind="d"),
        lambda artifact, _context: (artifact.payload, 0, {}),
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        FunctionalPipeline([first, second])


def test_optional_identity_stage_can_be_skipped_under_budget() -> None:
    initial = ArtifactEnvelope.create(kind="candidate-set", payload=["a", "b"])
    optional = FunctionStage(
        StageSpec(
            name="expensive-reranker",
            input_kind="candidate-set",
            output_kind="candidate-set",
            estimated_cost_ms=100,
            optional=True,
        ),
        lambda artifact, _context: (artifact.payload, 100, {}),
    )
    state = FunctionalPipeline([optional]).run(
        initial,
        context=PipelineContext(
            run_id="budget-test",
            effort="ultra-light",
            total_budget_ms=0,
        ),
    )
    assert not state.stopped
    assert state.artifact.sha256 == initial.sha256
    assert state.history == ()
    assert state.stop_reasons == (
        "skipped-optional-stage-budget:expensive-reranker",
    )


def test_required_stage_stops_before_exceeding_budget() -> None:
    initial = ArtifactEnvelope.create(kind="audio", payload="digest")
    required = FunctionStage(
        StageSpec(
            name="decode",
            input_kind="audio",
            output_kind="candidates",
            estimated_cost_ms=10,
        ),
        lambda artifact, _context: (artifact.payload, 10, {}),
    )
    state = FunctionalPipeline([required]).run(
        initial,
        context=PipelineContext(
            run_id="budget-test",
            effort="ultra-light",
            total_budget_ms=5,
        ),
    )
    assert state.stopped
    assert state.history == ()
    assert state.stop_reasons == ("required-stage-budget:decode",)


def test_effort_profiles_expand_compute_monotonically() -> None:
    ultra = effort_profile("ultra-light")
    cpu = effort_profile("cpu-quality")
    edge = effort_profile("edge-gpu")
    research = effort_profile("research")
    assert ultra.maximum_candidates < cpu.maximum_candidates <= edge.maximum_candidates
    assert edge.maximum_candidates < research.maximum_candidates
    assert ultra.evidence_budget_ms < cpu.evidence_budget_ms < edge.evidence_budget_ms
    assert edge.evidence_budget_ms < research.evidence_budget_ms
    assert not ultra.enable_neural_reranker
    assert edge.enable_acoustic_verifier
    assert research.enable_second_ear
