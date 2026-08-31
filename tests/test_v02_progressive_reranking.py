from __future__ import annotations

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.progressive_reranking import ProgressiveStage, progressive_rerank
from semantic_asr.rerankers import StaticCandidateRanker


def _candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence("a", "料金は3000円です", rank=1, hypothesis_count=3),
        CandidateEvidence("b", "料金は30000円です", rank=2, hypothesis_count=3),
        CandidateEvidence("c", "料金は3000円でした", rank=3, hypothesis_count=3),
    ]


def test_confident_cheap_stage_exits_before_expensive_teacher() -> None:
    stages = [
        ProgressiveStage(
            name="cheap-ngram",
            ranker=StaticCandidateRanker({"a": 8.0, "b": -2.0, "c": -3.0}),
            estimated_cost_ms=2,
            minimum_margin=0.50,
            maximum_entropy=0.40,
        ),
        ProgressiveStage(
            name="expensive-teacher",
            ranker=StaticCandidateRanker({"a": -2.0, "b": 9.0, "c": -3.0}),
            estimated_cost_ms=1_000,
        ),
    ]
    decision = progressive_rerank(_candidates(), stages, budget_ms=2_000)
    assert decision.selected_candidate_id == "a"
    assert decision.early_exit
    assert decision.used_budget_ms == 2
    assert [row.stage for row in decision.stages] == ["cheap-ngram"]
    assert decision.stopping_reason == "confident-after:cheap-ngram"
    assert not decision.calibrated_probability


def test_ambiguous_cheap_stage_escalates_to_expensive_teacher() -> None:
    stages = [
        ProgressiveStage(
            name="cheap-linear",
            ranker=StaticCandidateRanker({"a": 0.2, "b": 0.19, "c": 0.18}),
            estimated_cost_ms=3,
            minimum_margin=0.40,
            maximum_entropy=0.40,
        ),
        ProgressiveStage(
            name="quality-reranker",
            ranker=StaticCandidateRanker({"a": -1.0, "b": 5.0, "c": -2.0}),
            estimated_cost_ms=30,
            weight=2.0,
            minimum_margin=0.40,
            maximum_entropy=0.50,
        ),
    ]
    decision = progressive_rerank(_candidates(), stages, budget_ms=100)
    assert decision.selected_candidate_id == "b"
    assert decision.used_budget_ms == 33
    assert len(decision.stages) == 2
    assert decision.early_exit
    assert decision.stopping_reason == "confident-after:quality-reranker"


def test_budget_frontier_stops_before_expensive_stage() -> None:
    stages = [
        ProgressiveStage(
            name="cheap-linear",
            ranker=StaticCandidateRanker({"a": 0.2, "b": 0.19, "c": 0.18}),
            estimated_cost_ms=3,
            minimum_margin=0.90,
            maximum_entropy=0.01,
        ),
        ProgressiveStage(
            name="too-expensive",
            ranker=StaticCandidateRanker({"a": -1.0, "b": 5.0, "c": -2.0}),
            estimated_cost_ms=100,
        ),
    ]
    decision = progressive_rerank(_candidates(), stages, budget_ms=20)
    assert len(decision.stages) == 1
    assert decision.used_budget_ms == 3
    assert not decision.early_exit
    assert decision.stopping_reason == "budget-frontier"
    assert decision.selected_text in {candidate.text for candidate in _candidates()}


def test_no_stage_fits_budget_falls_back_to_existing_rank_order() -> None:
    stage = ProgressiveStage(
        name="expensive",
        ranker=StaticCandidateRanker({"a": -10.0, "b": 10.0, "c": 0.0}),
        estimated_cost_ms=100,
    )
    decision = progressive_rerank(_candidates(), [stage], budget_ms=0)
    assert decision.selected_candidate_id == "a"
    assert decision.stages == ()
    assert decision.used_budget_ms == 0
    assert decision.stopping_reason == "no-stage-fit-budget"
    assert sum(decision.preference_distribution.values()) == 1.0
