from __future__ import annotations

import math
from dataclasses import replace

import pytest

from semantic_asr.candidate_pool import CandidatePath, CandidatePool, logsumexp
from semantic_asr.cascade import (
    CascadeController,
    ExecutionBudget,
    FunctionStage,
    PipelineState,
    StageEstimate,
    StageExecution,
)
from semantic_asr.experiment import (
    DatasetManifest,
    SampleResult,
    UtteranceRecord,
    paired_bootstrap_comparison,
)
from semantic_asr.hard_negatives import generate_hard_negatives
from semantic_asr.koemo_bridge import (
    KoemoMeetingEvidence,
    channel_span_from_samples,
    live_event,
    normalized_event,
    observed_event,
)
from semantic_asr.mbr import decode_mbr
from semantic_asr.ngram import CountNGramLanguageModel, CountNGramScorer
from semantic_asr.planner_v2 import (
    ActionObservation,
    ActionSpec,
    PlannerFeatures,
    PlannerFeatureSchema,
    fit_learned_planner,
    plan_learned_evidence,
)
from semantic_asr.reranking import (
    FeatureSchema,
    FeatureVector,
    RankingGroup,
    TrainingCandidate,
    train_constrained_linear_reranker,
)
from semantic_asr.research_registry import SourceStatus, default_research_registry
from semantic_asr.risk_control import (
    AdaptivePolicy,
    PolicyOutcome,
    RuntimeRiskState,
    fit_risk_control,
    select_policy,
)
from semantic_asr.score_types import (
    CalibrationExample,
    EvidenceScore,
    IsotonicCalibrator,
    PlattCalibrator,
    ScoreSemantics,
)
from semantic_asr.sequence_scorers import TextCandidate


def _path(identifier: str, text: str, score: float, source: str = "whisper") -> CandidatePath:
    return CandidatePath(
        path_id=identifier,
        text=text,
        cumulative_log_likelihood=score,
        token_ids=(1, 2, 3),
        source=source,
    )


def _pool() -> CandidatePool:
    return CandidatePool.from_paths(
        (
            _path("a1", "今日は東京に行く", -1.0),
            _path("a2", "今日は東京に行く", -2.0),
            _path("b1", "今日は東京へ行く", -1.1),
            _path("c1", "今日は京都に行く", -1.6, source="second-ear"),
        )
    )


def test_candidate_pool_preserves_duplicate_path_mass() -> None:
    pool = _pool()
    tokyo_ni = next(
        candidate for candidate in pool.candidates if candidate.text == "今日は東京に行く"
    )
    assert tokyo_ni.path_count == 2
    assert tokyo_ni.aggregate_log_likelihood == pytest.approx(logsumexp((-1.0, -2.0)))
    assert tokyo_ni.path_mass_bonus > 0
    assert sum(pool.posterior().values()) == pytest.approx(1.0)
    diagnostics = pool.diagnostics()
    assert diagnostics.path_count == 4
    assert diagnostics.surface_count == 3
    assert diagnostics.unique_surface_ratio == pytest.approx(0.75)
    assert diagnostics.source_count == 2


def test_score_semantics_require_explicit_calibration() -> None:
    with pytest.raises(ValueError, match="calibrator"):
        EvidenceScore.raw(
            0.8,
            semantics=ScoreSemantics.PROBABILITY,
            scorer="chat-self-report",
        )
    examples = [
        CalibrationExample(score=-3.0, correct=False),
        CalibrationExample(score=-2.0, correct=False),
        CalibrationExample(score=-1.0, correct=False),
        CalibrationExample(score=0.0, correct=True),
        CalibrationExample(score=1.0, correct=True),
        CalibrationExample(score=2.0, correct=True),
    ]
    platt = PlattCalibrator.fit(
        examples,
        source_semantics=ScoreSemantics.LOGIT,
        dataset_digest="calibration-set-a",
        iterations=500,
    )
    low = platt.probability(
        EvidenceScore.raw(-2.0, semantics=ScoreSemantics.LOGIT, scorer="reranker")
    )
    high = platt.probability(
        EvidenceScore.raw(2.0, semantics=ScoreSemantics.LOGIT, scorer="reranker")
    )
    assert low.calibrated and high.calibrated
    assert low.value < high.value
    assert high.provenance.calibration_digest == platt.digest

    isotonic = IsotonicCalibrator.fit(
        examples,
        source_semantics=ScoreSemantics.LOGIT,
        dataset_digest="calibration-set-a",
    )
    assert (
        isotonic.probability(
            EvidenceScore.raw(-3.0, semantics=ScoreSemantics.LOGIT, scorer="reranker")
        ).value
        <= isotonic.probability(
            EvidenceScore.raw(2.0, semantics=ScoreSemantics.LOGIT, scorer="reranker")
        ).value
    )


def test_mbr_selects_only_existing_candidate() -> None:
    pool = _pool()
    posterior = {
        next(
            candidate.candidate_id
            for candidate in pool.candidates
            if candidate.text == "今日は東京に行く"
        ): 0.40,
        next(
            candidate.candidate_id
            for candidate in pool.candidates
            if candidate.text == "今日は東京へ行く"
        ): 0.35,
        next(
            candidate.candidate_id
            for candidate in pool.candidates
            if candidate.text == "今日は京都に行く"
        ): 0.25,
    }
    decision = decode_mbr(pool, posterior=posterior)
    assert decision.selected in pool.candidates
    assert decision.selected.candidate_id == decision.risks[0].candidate_id
    assert decision.risks[0].expected_risk <= decision.risks[-1].expected_risk
    assert len(decision.decision_digest) == 64


def test_count_ngram_scores_familiar_text_higher() -> None:
    model = CountNGramLanguageModel.fit(
        ["東京に行く", "東京に行く", "東京へ行く", "今日は東京に行く"],
        order=3,
    )
    scorer = CountNGramScorer(model)
    scores = scorer.score(
        [
            TextCandidate("familiar", "東京に行く"),
            TextCandidate("unfamiliar", "冥王星を泳ぐ"),
        ]
    )
    by_id = {row.candidate_id: row for row in scores}
    assert by_id["familiar"].average.value > by_id["unfamiliar"].average.value
    assert by_id["familiar"].cumulative.semantics == ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD


def test_constrained_reranker_learns_safe_signs() -> None:
    schema = FeatureSchema(
        names=("acoustic", "missing"),
        monotonicity={"acoustic": 1, "missing": -1},
    )

    def vector(acoustic: float, missing: float) -> FeatureVector:
        return FeatureVector.create(
            schema,
            {"acoustic": acoustic, "missing": missing},
        )

    groups = tuple(
        RankingGroup(
            group_id=f"g{index}",
            candidates=(
                TrainingCandidate("good", vector(2.0 + index * 0.1, 0.0), target_loss=0.0),
                TrainingCandidate("middle", vector(0.5, 0.3), target_loss=0.35),
                TrainingCandidate("bad", vector(-1.0, 1.0), target_loss=0.90),
            ),
        )
        for index in range(4)
    )
    model, report = train_constrained_linear_reranker(
        groups,
        schema=schema,
        epochs=120,
        seed=7,
    )
    assert model.weights["acoustic"] >= 0
    assert model.weights["missing"] <= 0
    assert report.pairwise_accuracy >= 0.95
    ranked = model.rank(
        {
            "good": vector(2.5, 0.0),
            "bad": vector(-1.0, 1.0),
        }
    )
    assert ranked[0][0] == "good"


def test_risk_control_filters_policies_and_uses_state() -> None:
    policies = (
        AdaptivePolicy("cheap", candidate_count=1, stages=("acoustic",)),
        AdaptivePolicy(
            "rerank",
            candidate_count=3,
            stages=("acoustic", "mbr", "reranker"),
            minimum_entropy=0.2,
            maximum_cost_ms=500,
        ),
        AdaptivePolicy(
            "verified-fallback",
            candidate_count=3,
            stages=("acoustic", "reranker", "verifier"),
            conservative_fallback=True,
        ),
    )
    outcomes = []
    for index in range(500):
        outcomes.extend(
            (
                PolicyOutcome("cheap", 0.30, 20.0, f"s{index}"),
                PolicyOutcome("rerank", 0.0 if index % 100 else 1.0, 120.0, f"s{index}"),
                PolicyOutcome(
                    "verified-fallback",
                    0.0 if index % 200 else 1.0,
                    800.0,
                    f"s{index}",
                ),
            )
        )
    profile = fit_risk_control(
        policies,
        outcomes,
        target_risk=0.10,
        minimum_samples=100,
    )
    assert not profile.bound("cheap").passed
    assert profile.bound("rerank").passed
    selected = select_policy(
        profile,
        RuntimeRiskState(
            entropy=0.6,
            posterior_margin=0.2,
            disagreement=0.3,
            evidence_coverage=0.8,
            semantic_criticality=0.2,
            available_candidates=3,
        ),
        cost_budget_ms=500,
    )
    assert selected.policy.policy_id == "rerank"


def test_generated_candidate_requires_verification() -> None:
    pool = _pool()
    generated_id = pool.candidates[0].candidate_id
    state = PipelineState(
        candidate_pool=pool,
        generated_candidate_ids=(generated_id,),
    )
    with pytest.raises(ValueError, match="verified"):
        state.accept(generated_id)
    verified = replace(state, verified_candidate_ids=(generated_id,))
    assert verified.accept(generated_id).decision_status == "accepted"


def test_functional_cascade_records_budgeted_trace() -> None:
    state = PipelineState(candidate_pool=_pool(), selective_risk=0.8)

    def run(current: PipelineState) -> StageExecution:
        return StageExecution(replace(current, selective_risk=0.1))

    stage = FunctionStage(
        stage_id="cheap-rerank",
        stage_kind="reranker",
        runner=run,
        estimator=lambda _state: StageEstimate(
            cost_ms=10.0,
            expected_loss_reduction=0.2,
        ),
    )
    result = CascadeController((stage,)).execute(
        state,
        budget=ExecutionBudget(maximum_cost_ms=100.0),
    )
    assert result.state.decision_status == "provisional"
    assert len(result.state.traces) == 1
    assert result.state.traces[0].invoked
    assert result.state.selective_risk == pytest.approx(0.1)


def test_hard_negatives_label_high_impact_changes() -> None:
    negatives = generate_hard_negatives(
        "えっと2026年には行かないです",
        seed=4,
        maximum=20,
    )
    kinds = {negative.error_type for negative in negatives}
    assert {"number", "negation", "filler"}.issubset(kinds)
    assert all(negative.text != negative.source_text for negative in negatives)
    assert max(negative.criticality for negative in negatives) == pytest.approx(1.0)


def test_manifest_detects_cross_split_leakage_and_bootstrap_is_paired() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    manifest = DatasetManifest(
        records=(
            UtteranceRecord("train-a", "train", digest_a, "東京", speaker_id="speaker-a"),
            UtteranceRecord("test-a", "test", digest_b, "京都", speaker_id="speaker-a"),
        ),
        dataset_name="fixture",
        dataset_revision="1",
    )
    findings = manifest.leakage_findings(reference_near_duplicate=False)
    assert any(finding.kind == "speaker-id" for finding in findings)
    with pytest.raises(ValueError, match="leakage"):
        manifest.assert_leakage_free(reference_near_duplicate=False)

    results = []
    for index in range(20):
        results.extend(
            (
                SampleResult(f"s{index}", "baseline", {"cer": 0.30}, 10.0),
                SampleResult(f"s{index}", "candidate", {"cer": 0.10}, 12.0),
            )
        )
    comparison = paired_bootstrap_comparison(
        results,
        baseline_system="baseline",
        candidate_system="candidate",
        metric="cer",
        iterations=500,
    )
    assert comparison.samples == 20
    assert comparison.upper_delta < 0
    assert comparison.probability_candidate_better == pytest.approx(1.0)


def _planner_features(
    schema: PlannerFeatureSchema,
    *,
    entropy: float,
    duration: float,
    gpu: float,
) -> PlannerFeatures:
    return PlannerFeatures.create(
        schema,
        {
            "entropy": entropy,
            "disagreement": entropy / 2,
            "missing_evidence": 0.2,
            "posterior_margin_inverse": entropy,
            "semantic_criticality": 0.6,
            "span_duration_seconds": duration,
            "candidate_count_log": math.log1p(5),
            "cache_miss": 1.0,
            "cpu_tier": 1.0,
            "gpu_available": gpu,
        },
    )


def test_learned_planner_predicts_gain_and_respects_budget() -> None:
    schema = PlannerFeatureSchema()
    observations = []
    for index in range(20):
        features = _planner_features(
            schema,
            entropy=0.2 + index / 100,
            duration=1.0 + index / 20,
            gpu=0.0,
        )
        observations.append(
            ActionObservation(
                action_kind="relisten",
                features=features,
                loss_before=0.5,
                loss_after=0.25,
                measured_cost_ms=80.0 + index,
                sample_id=f"s{index}",
                platform_id="cpu-test",
            )
        )
    model = fit_learned_planner(
        observations,
        schema=schema,
        platform_id="cpu-test",
        minimum_samples_per_action=12,
    )
    action = ActionSpec(
        action_id="a1",
        action_kind="relisten",
        features=_planner_features(schema, entropy=0.5, duration=2.0, gpu=0.0),
    )
    plan = plan_learned_evidence(
        (action,),
        model,
        cost_budget_ms=200.0,
    )
    assert [row.action.action_id for row in plan.selected] == ["a1"]
    assert plan.expected_gain > 0


def test_koemo_bridge_keeps_live_observed_and_normalized_separate() -> None:
    span = channel_span_from_samples(
        span_id="span-1",
        channel="microphone",
        start_ms=0,
        sample_rate=16_000,
        sample_count=32_000,
        audio_sha256="a" * 64,
        source_recording_sha256="b" * 64,
    )
    live = live_event(span_id=span.span_id, text="速報", backend="winrt")
    observed = observed_event(
        span_id=span.span_id,
        text="えっと東京へ",
        evidence_digest="evidence-1",
        backend="semantic-asr",
    )
    normalized = normalized_event(
        span_id=span.span_id,
        text="東京へ",
        observed_evidence_digest="evidence-1",
        normalizer="readability",
    )
    meeting = KoemoMeetingEvidence(
        meeting_id="meeting-1",
        spans=(span,),
        events=(live, observed, normalized),
        semantic_asr_revision="test",
        configuration_digest="config",
    )
    assert meeting.authoritative_events() == (observed,)
    assert meeting.normalized_events() == (normalized,)
    assert len(meeting.digest) == 64


def test_research_registry_blocks_unverified_attribution() -> None:
    registry = default_research_registry()
    assert registry.source("qwen3.8-flash-next").status == SourceStatus.PINNED_PRIMARY
    provisional = {source.source_id for source in registry.provisional_sources()}
    assert {"glm-5.3", "kimi-k3"}.issubset(provisional)
    assert all(
        source_id not in {"glm-5.3", "kimi-k3"}
        for translation in registry.translations
        for source_id in translation.source_ids
    )
