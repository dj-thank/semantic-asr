from __future__ import annotations

import json

import pytest

from semantic_asr.candidate_pool import CandidatePath, CandidatePool
from semantic_asr.mbr import semantic_pairwise_loss
from semantic_asr.model_io import (
    deserialize_constrained_reranker,
    serialize_constrained_reranker,
)
from semantic_asr.planner_v2 import (
    ActionObservation,
    ActionSpec,
    PlannerFeatureSchema,
    PlannerFeatures,
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
from semantic_asr.risk_control import (
    AdaptivePolicy,
    PolicyOutcome,
    fit_risk_control,
)


def test_semantic_mbr_penalizes_inserted_number_and_negation() -> None:
    pool = CandidatePool.from_paths(
        (
            CandidatePath("truth", "会議を始めます", -1.0),
            CandidatePath("number", "会議を7時に始めます", -1.1),
            CandidatePath("negation", "会議を始めません", -1.2),
        )
    )
    truth = next(candidate for candidate in pool.candidates if candidate.text == "会議を始めます")
    number = next(candidate for candidate in pool.candidates if "7時" in candidate.text)
    negation = next(candidate for candidate in pool.candidates if "ません" in candidate.text)
    number_loss = semantic_pairwise_loss(number, truth)
    negation_loss = semantic_pairwise_loss(negation, truth)
    assert number_loss.number > 0
    assert number_loss.date_time > 0
    assert negation_loss.negation > 0


def test_risk_control_rejects_untested_policy_family() -> None:
    policies = (
        AdaptivePolicy("tested", candidate_count=1),
        AdaptivePolicy("untested-fallback", candidate_count=1, conservative_fallback=True),
    )
    with pytest.raises(ValueError, match="every policy requires held-out outcomes"):
        fit_risk_control(
            policies,
            (
                PolicyOutcome(
                    policy_id="tested",
                    bounded_loss=0.0,
                    measured_cost_ms=1.0,
                    sample_id="sample-1",
                ),
            ),
            target_risk=0.1,
            minimum_samples=1,
        )


def _small_reranker():
    schema = FeatureSchema(names=("acoustic",), monotonicity={"acoustic": 1})

    def vector(value: float) -> FeatureVector:
        return FeatureVector.create(schema, {"acoustic": value})

    model, _report = train_constrained_linear_reranker(
        (
            RankingGroup(
                "group-1",
                (
                    TrainingCandidate("good", vector(1.0), 0.0),
                    TrainingCandidate("bad", vector(-1.0), 1.0),
                ),
            ),
            RankingGroup(
                "group-2",
                (
                    TrainingCandidate("good", vector(0.8), 0.0),
                    TrainingCandidate("bad", vector(-0.8), 0.9),
                ),
            ),
        ),
        schema=schema,
        epochs=40,
    )
    return model


def test_model_io_rejects_unknown_objective_even_with_digest_check_disabled() -> None:
    payload = serialize_constrained_reranker(_small_reranker())
    tampered = json.loads(json.dumps(payload))
    tampered["objective"] = "make-text-fluent"
    with pytest.raises(ValueError, match="unsupported constrained reranker objective"):
        deserialize_constrained_reranker(tampered, verify_digest=False)


def _planner_features(schema: PlannerFeatureSchema, entropy: float) -> PlannerFeatures:
    return PlannerFeatures.create(
        schema,
        {
            "entropy": entropy,
            "disagreement": 0.3,
            "missing_evidence": 0.2,
            "posterior_margin_inverse": 0.7,
            "semantic_criticality": 0.6,
            "span_duration_seconds": 2.0,
            "candidate_count_log": 1.8,
            "cache_miss": 1.0,
            "cpu_tier": 1.0,
            "gpu_available": 0.0,
        },
    )


def test_learned_planner_does_not_duplicate_exclusive_rejections() -> None:
    schema = PlannerFeatureSchema()
    observations = tuple(
        ActionObservation(
            action_kind="relisten",
            features=_planner_features(schema, 0.4 + index / 100),
            loss_before=0.5,
            loss_after=0.2,
            measured_cost_ms=50.0,
            sample_id=f"sample-{index}",
            platform_id="cpu",
        )
        for index in range(12)
    )
    model = fit_learned_planner(
        observations,
        schema=schema,
        platform_id="cpu",
        minimum_samples_per_action=4,
    )
    actions = (
        ActionSpec(
            "a1",
            "relisten",
            _planner_features(schema, 0.8),
            exclusive_group="same-island",
        ),
        ActionSpec(
            "a2",
            "relisten",
            _planner_features(schema, 0.8),
            exclusive_group="same-island",
        ),
        ActionSpec(
            "a3",
            "relisten",
            _planner_features(schema, 0.8),
            exclusive_group="other-island",
        ),
    )
    plan = plan_learned_evidence(
        actions,
        model,
        cost_budget_ms=500.0,
        maximum_actions=2,
    )
    all_ids = [row.action.action_id for row in (*plan.selected, *plan.rejected)]
    assert sorted(all_ids) == ["a1", "a2", "a3"]
    assert len(all_ids) == len(set(all_ids))
