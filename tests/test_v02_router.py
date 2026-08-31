import pytest

from semantic_asr.evidence_router import (
    QuantileBalancedRouterConfig,
    ResidualBranch,
    RouterState,
    mix_residual_branches,
    route_evidence_actions,
    update_router_state,
)
from semantic_asr.planner import EvidenceAction, EvidenceBudget


def _action(action_id: str, kind: str, criticality: float = 0.5) -> EvidenceAction:
    return EvidenceAction(
        action_id=action_id,
        kind=kind,
        start_ms=0,
        end_ms=1000,
        estimated_cost_ms=100,
        expected_information_gain=0.5,
        semantic_criticality=criticality,
        utility=0.005,
        reasons=("test",),
        affects_observed_decision=True,
    )


def test_quantile_router_avoids_permanent_expert_starvation() -> None:
    state = RouterState(
        selection_count={"qwen-second-ear": 100, "whisper-relisten": 1},
        reward_sum={"qwen-second-ear": 20.0, "whisper-relisten": 0.2},
        total_selections=101,
    )
    result = route_evidence_actions(
        [
            _action("island-1:qwen-second-ear", "qwen-second-ear"),
            _action("island-2:whisper-relisten", "whisper-relisten"),
        ],
        budget=EvidenceBudget(total_cost_ms=100, max_actions=1, minimum_utility=0.0),
        state=state,
        config=QuantileBalancedRouterConfig(
            load_balance_strength=1.0,
            reward_strength=0.0,
            semantic_priority_strength=0.0,
            duplicate_island_penalty=0.0,
        ),
    )
    assert result.plan.selected[0].kind == "whisper-relisten"
    assert result.selected[0].load_balance_bonus > 0


def test_router_state_learns_empirical_reward() -> None:
    updated = update_router_state(
        RouterState(),
        {
            "whisper-relisten": [0.2, 0.4],
            "local-teacher": [-0.1, 0.0],
        },
    )
    assert updated.selection_count["whisper-relisten"] == 2
    assert updated.reward_sum["whisper-relisten"] == pytest.approx(0.6)
    assert updated.total_selections == 4


def test_residual_mixer_is_bounded_and_normalized() -> None:
    mixed = mix_residual_branches(
        [
            ResidualBranch("acoustic", {"a": 0.9, "b": 0.1}, reliability=0.9),
            ResidualBranch("mora", {"a": 0.8, "b": 0.2}, reliability=0.8),
            ResidualBranch("reranker", {"a": 0.2, "b": 0.8}, reliability=0.2),
            ResidualBranch("teacher", {"a": 0.0, "b": 1.0}, reliability=0.01),
        ],
        maximum_active_branches=3,
        minimum_reliability=0.05,
    )
    assert len(mixed.active_branches) == 3
    assert "teacher" not in mixed.active_branches
    assert sum(mixed.gates.values()) == pytest.approx(1.0)
    assert mixed.scores["a"] > mixed.scores["b"]
