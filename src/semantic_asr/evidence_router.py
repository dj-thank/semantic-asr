from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from .planner import EvidenceAction, EvidenceBudget, EvidencePlan


@dataclass(frozen=True, slots=True)
class RouterState:
    selection_count: dict[str, int] = field(default_factory=dict)
    reward_sum: dict[str, float] = field(default_factory=dict)
    total_selections: int = 0
    version: str = "1"

    def __post_init__(self) -> None:
        if self.total_selections < 0:
            raise ValueError("total_selections must be non-negative")
        if any(value < 0 for value in self.selection_count.values()):
            raise ValueError("selection counts must be non-negative")
        if any(not math.isfinite(value) for value in self.reward_sum.values()):
            raise ValueError("reward sums must be finite")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class QuantileBalancedRouterConfig:
    load_balance_strength: float = 0.18
    reward_strength: float = 0.12
    semantic_priority_strength: float = 0.30
    duplicate_island_penalty: float = 0.45
    maximum_actions_per_island: int = 2
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.load_balance_strength,
            self.reward_strength,
            self.semantic_priority_strength,
            self.duplicate_island_penalty,
            self.minimum_score,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("router configuration values must be finite and non-negative")
        if self.maximum_actions_per_island < 1:
            raise ValueError("maximum_actions_per_island must be positive")


@dataclass(frozen=True, slots=True)
class RoutedAction:
    action: EvidenceAction
    base_utility: float
    load_balance_bonus: float
    empirical_reward_bonus: float
    semantic_bonus: float
    redundancy_penalty: float
    routing_score: float


@dataclass(frozen=True, slots=True)
class RoutingResult:
    selected: tuple[RoutedAction, ...]
    rejected: tuple[RoutedAction, ...]
    plan: EvidencePlan
    state_digest: str


def _kind_share(kind: str, state: RouterState, available_kinds: int) -> float:
    count = state.selection_count.get(kind, 0)
    denominator = max(1, state.total_selections)
    if state.total_selections == 0:
        return 1.0 / max(1, available_kinds)
    return count / denominator


def _mean_reward(kind: str, state: RouterState) -> float:
    count = state.selection_count.get(kind, 0)
    if count <= 0:
        return 0.0
    return state.reward_sum.get(kind, 0.0) / count


def _island_id(action: EvidenceAction) -> str:
    return action.action_id.split(":", 1)[0]


def _route_score(
    action: EvidenceAction,
    *,
    state: RouterState,
    available_kinds: int,
    per_island: int,
    config: QuantileBalancedRouterConfig,
) -> RoutedAction:
    target_share = 1.0 / max(1, available_kinds)
    observed_share = _kind_share(action.kind, state, available_kinds)
    underuse = max(-target_share, min(target_share, target_share - observed_share))
    load_bonus = config.load_balance_strength * underuse
    reward_bonus = config.reward_strength * _mean_reward(action.kind, state)
    semantic_bonus = config.semantic_priority_strength * action.semantic_criticality
    redundancy = config.duplicate_island_penalty * per_island if per_island > 0 else 0.0
    score = action.utility + load_bonus + reward_bonus + semantic_bonus - redundancy
    return RoutedAction(
        action=action,
        base_utility=action.utility,
        load_balance_bonus=load_bonus,
        empirical_reward_bonus=reward_bonus,
        semantic_bonus=semantic_bonus,
        redundancy_penalty=redundancy,
        routing_score=score,
    )


def route_evidence_actions(
    actions: Sequence[EvidenceAction],
    *,
    budget: EvidenceBudget | None = None,
    state: RouterState | None = None,
    config: QuantileBalancedRouterConfig | None = None,
) -> RoutingResult:
    budget = budget or EvidenceBudget()
    state = state or RouterState()
    config = config or QuantileBalancedRouterConfig()
    kinds = {action.kind for action in actions}
    remaining = list(actions)
    selected: list[RoutedAction] = []
    rejected: list[RoutedAction] = []
    used_ms = 0
    per_island: dict[str, int] = {}

    while remaining and len(selected) < budget.max_actions:
        scored = [
            _route_score(
                action,
                state=state,
                available_kinds=len(kinds),
                per_island=per_island.get(_island_id(action), 0),
                config=config,
            )
            for action in remaining
        ]
        scored.sort(
            key=lambda row: (
                -row.routing_score,
                -row.action.semantic_criticality,
                row.action.estimated_cost_ms,
                row.action.action_id,
            )
        )
        best = scored[0]
        remaining.remove(best.action)
        island = _island_id(best.action)
        if best.routing_score < config.minimum_score:
            rejected.append(best)
            continue
        if per_island.get(island, 0) >= config.maximum_actions_per_island:
            rejected.append(best)
            continue
        if used_ms + best.action.estimated_cost_ms > budget.total_cost_ms:
            rejected.append(best)
            continue
        selected.append(best)
        used_ms += best.action.estimated_cost_ms
        per_island[island] = per_island.get(island, 0) + 1

    for action in remaining:
        rejected.append(
            _route_score(
                action,
                state=state,
                available_kinds=len(kinds),
                per_island=per_island.get(_island_id(action), 0),
                config=config,
            )
        )
    stopping_reason = (
        "no-actions"
        if not actions
        else "budget-exhausted"
        if selected and used_ms >= budget.total_cost_ms
        else "router-frontier-reached"
        if selected
        else "no-routable-action"
    )
    plan = EvidencePlan(
        selected=tuple(row.action for row in selected),
        rejected=tuple(row.action for row in rejected),
        budget_ms=budget.total_cost_ms,
        used_ms=used_ms,
        expected_information_gain=sum(row.action.expected_information_gain for row in selected),
        stopping_reason=stopping_reason,
    )
    return RoutingResult(
        selected=tuple(selected),
        rejected=tuple(rejected),
        plan=plan,
        state_digest=state.digest,
    )


def update_router_state(
    state: RouterState,
    outcomes: Mapping[str, Sequence[float]],
) -> RouterState:
    selection_count = dict(state.selection_count)
    reward_sum = dict(state.reward_sum)
    total = state.total_selections
    for kind, rewards in outcomes.items():
        finite = [float(value) for value in rewards if math.isfinite(float(value))]
        if not finite:
            continue
        selection_count[kind] = selection_count.get(kind, 0) + len(finite)
        reward_sum[kind] = reward_sum.get(kind, 0.0) + sum(finite)
        total += len(finite)
    return RouterState(
        selection_count=selection_count,
        reward_sum=reward_sum,
        total_selections=total,
        version=state.version,
    )


@dataclass(frozen=True, slots=True)
class ResidualBranch:
    name: str
    scores: dict[str, float]
    reliability: float = 1.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("residual branch name is required")
        if not 0 <= self.reliability <= 1:
            raise ValueError("branch reliability must be in [0, 1]")
        if self.cost < 0 or not math.isfinite(self.cost):
            raise ValueError("branch cost must be finite and non-negative")
        if any(not math.isfinite(value) for value in self.scores.values()):
            raise ValueError("branch scores must be finite")


@dataclass(frozen=True, slots=True)
class ResidualMix:
    scores: dict[str, float]
    gates: dict[str, float]
    active_branches: tuple[str, ...]


def mix_residual_branches(
    branches: Sequence[ResidualBranch],
    *,
    maximum_active_branches: int = 4,
    minimum_reliability: float = 0.05,
) -> ResidualMix:
    """Bounded multi-branch residual mixing.

    This is an orchestration analogue of gated residual/mHC mechanisms, not a
    reproduction of any transformer kernel.
    """

    if maximum_active_branches < 1:
        raise ValueError("maximum_active_branches must be positive")
    eligible = [branch for branch in branches if branch.reliability >= minimum_reliability]
    eligible.sort(key=lambda branch: (-branch.reliability, branch.cost, branch.name))
    active = eligible[:maximum_active_branches]
    identifiers = sorted({candidate_id for branch in active for candidate_id in branch.scores})
    if not active or not identifiers:
        return ResidualMix(scores={}, gates={}, active_branches=())
    raw_gates = {branch.name: branch.reliability / (1.0 + branch.cost) for branch in active}
    total = sum(raw_gates.values()) or 1.0
    gates = {name: value / total for name, value in raw_gates.items()}
    scores = {
        candidate_id: sum(
            gates[branch.name] * branch.scores.get(candidate_id, 0.0) for branch in active
        )
        for candidate_id in identifiers
    }
    return ResidualMix(
        scores=scores,
        gates=gates,
        active_branches=tuple(branch.name for branch in active),
    )
