from semantic_asr.contracts import CandidateEvidence
from semantic_asr.fusion import fuse_candidates
from semantic_asr.planner import EvidenceBudget, plan_evidence
from semantic_asr.semantic_lattice import (
    build_semantic_lattice,
    semantic_change_warnings,
)


def test_same_reading_collapses_surface_difference() -> None:
    lattice = build_semantic_lattice(
        [
            CandidateEvidence("kanji", "今日", acoustic=0.8, reading="キョウ"),
            CandidateEvidence("kana", "きょう", acoustic=0.7, reading="キョウ"),
        ]
    )
    assert lattice.alignment_level == "mora"
    assert lattice.contradiction_islands == ()


def test_number_negation_and_entity_islands_are_high_risk() -> None:
    candidates = [
        CandidateEvidence("a", "Qwenは三人では使わない", acoustic=0.51, mora=0.50),
        CandidateEvidence("b", "Qwenは二人で使う", acoustic=0.50, mora=0.51),
    ]
    ranked = fuse_candidates(candidates)
    lattice = build_semantic_lattice(
        candidates,
        posterior=ranked[0].gate.posterior,
        pivot_candidate_id=ranked[0].candidate.candidate_id,
        segment_start_ms=1_000,
        segment_end_ms=5_000,
    )
    kinds = {kind for island in lattice.contradiction_islands for kind in island.kinds}
    assert "number-or-quantity" in kinds
    assert "negation-meaning-flip" in kinds
    assert "latin-acronym-or-term" in kinds
    assert max(island.semantic_criticality for island in lattice.contradiction_islands) == 1.0
    assert all(island.start_ms is not None for island in lattice.contradiction_islands)


def test_budgeted_plan_prefers_high_information_actions() -> None:
    candidates = [
        CandidateEvidence("a", "三万円です", acoustic=0.51, mora=0.49),
        CandidateEvidence("b", "二万円です", acoustic=0.50, mora=0.51),
    ]
    ranked = fuse_candidates(candidates)
    lattice = build_semantic_lattice(
        candidates,
        posterior=ranked[0].gate.posterior,
        pivot_candidate_id=ranked[0].candidate.candidate_id,
        segment_start_ms=0,
        segment_end_ms=3_000,
    )
    plan = plan_evidence(
        ranked,
        lattice,
        budget=EvidenceBudget(total_cost_ms=2_500, max_actions=2),
    )
    assert plan.selected
    assert plan.used_ms <= 2_500
    assert len(plan.selected) <= 2
    assert all(action.utility > 0 for action in plan.selected)
    assert any("currency" in action.reasons for action in plan.selected)


def test_semantic_change_warnings_detect_meaning_flip() -> None:
    warnings = semantic_change_warnings(
        "明日は行きません。料金は3000円です。",
        "明日は行きます。料金は30000円です。",
    )
    assert "negation-meaning-flip" in warnings
    assert "number-or-quantity" in warnings
    assert "currency" in warnings
