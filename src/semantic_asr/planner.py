from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .contracts import RankedCandidate
from .semantic_lattice import SemanticIsland, SemanticLattice

ActionKind = Literal[
    "whisper-relisten",
    "qwen-second-ear",
    "forced-align",
    "local-teacher",
    "lexicon-lookup",
]


@dataclass(frozen=True, slots=True)
class EvidenceBudget:
    total_cost_ms: int = 12_000
    max_actions: int = 8
    minimum_utility: float = 0.00004

    def __post_init__(self) -> None:
        if self.total_cost_ms < 0 or self.max_actions < 0:
            raise ValueError("evidence budget values must be non-negative")
        if self.minimum_utility < 0:
            raise ValueError("minimum utility must be non-negative")


@dataclass(frozen=True, slots=True)
class EvidenceAction:
    action_id: str
    kind: ActionKind
    start_ms: int | None
    end_ms: int | None
    estimated_cost_ms: int
    expected_information_gain: float
    semantic_criticality: float
    utility: float
    reasons: tuple[str, ...]
    affects_observed_decision: bool


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    selected: tuple[EvidenceAction, ...]
    rejected: tuple[EvidenceAction, ...]
    budget_ms: int
    used_ms: int
    expected_information_gain: float
    stopping_reason: str


def _duration_ms(island: SemanticIsland) -> int:
    if island.start_ms is None or island.end_ms is None:
        return 2_000
    return max(160, island.end_ms - island.start_ms)


def _cost(kind: ActionKind, duration_ms: int) -> int:
    fixed, realtime_factor = {
        "whisper-relisten": (120, 0.18),
        "qwen-second-ear": (800, 0.55),
        "forced-align": (360, 0.22),
        "local-teacher": (180, 0.0),
        "lexicon-lookup": (20, 0.0),
    }[kind]
    return int(round(fixed + duration_ms * realtime_factor))


def _fit(kind: ActionKind, island: SemanticIsland) -> float:
    kinds = set(island.kinds)
    if kind == "whisper-relisten":
        return 0.72 + 0.22 * island.posterior_ambiguity
    if kind == "qwen-second-ear":
        high_impact = bool(
            kinds
            & {
                "number-or-quantity",
                "date-or-time",
                "currency",
                "percentage",
                "negation-meaning-flip",
                "entity-or-domain-term",
                "latin-acronym-or-term",
                "modality-or-intent",
            }
        )
        return 0.90 if high_impact else 0.58
    if kind == "forced-align":
        phonetic = bool(
            kinds
            & {
                "special-mora",
                "particle-or-functional",
                "phonetic-or-punctuation",
            }
        )
        return 0.84 if phonetic else 0.48
    if kind == "local-teacher":
        semantic = bool(
            kinds
            & {
                "entity-or-domain-term",
                "latin-acronym-or-term",
                "modality-or-intent",
                "particle-or-functional",
            }
        )
        return 0.52 if semantic else 0.28
    if kind == "lexicon-lookup":
        lexical = bool(kinds & {"entity-or-domain-term", "latin-acronym-or-term"})
        return 0.78 if lexical else 0.12
    raise AssertionError(kind)


def _candidate_actions(
    lattice: SemanticLattice,
    *,
    global_uncertainty: float,
) -> list[EvidenceAction]:
    output: list[EvidenceAction] = []
    for index, island in enumerate(lattice.contradiction_islands):
        duration = _duration_ms(island)
        base_gain = island.expected_information_gain * (0.55 + 0.45 * global_uncertainty)
        for kind in (
            "whisper-relisten",
            "forced-align",
            "qwen-second-ear",
            "lexicon-lookup",
            "local-teacher",
        ):
            gain = min(1.0, max(0.0, base_gain * _fit(kind, island)))
            cost = _cost(kind, duration)
            reasons = tuple(
                dict.fromkeys(
                    (
                        "semantic-contradiction-island",
                        *island.kinds,
                        f"timing:{island.timing_source}",
                        f"alignment:{lattice.alignment_level}",
                    )
                )
            )
            output.append(
                EvidenceAction(
                    action_id=f"island-{index:04d}:{kind}",
                    kind=kind,
                    start_ms=island.start_ms,
                    end_ms=island.end_ms,
                    estimated_cost_ms=cost,
                    expected_information_gain=gain,
                    semantic_criticality=island.semantic_criticality,
                    utility=gain / max(1, cost),
                    reasons=reasons,
                    affects_observed_decision=kind
                    not in {
                        "local-teacher",
                        "lexicon-lookup",
                    },
                )
            )
    return output


def plan_evidence(
    ranked: Sequence[RankedCandidate],
    lattice: SemanticLattice,
    *,
    budget: EvidenceBudget | None = None,
    enabled: Sequence[ActionKind] = (
        "whisper-relisten",
        "qwen-second-ear",
        "forced-align",
        "local-teacher",
        "lexicon-lookup",
    ),
) -> EvidencePlan:
    budget = budget or EvidenceBudget()
    if not ranked:
        return EvidencePlan((), (), budget.total_cost_ms, 0, 0.0, "no-ranked-candidates")
    if not ranked[0].gate.needs_relisten:
        return EvidencePlan((), (), budget.total_cost_ms, 0, 0.0, "observation-already-confident")
    if not lattice.contradiction_islands:
        return EvidencePlan((), (), budget.total_cost_ms, 0, 0.0, "no-localized-contradiction")
    enabled_set = set(enabled)
    gate = ranked[0].gate
    global_uncertainty = min(
        1.0,
        0.50 * gate.entropy + 0.30 * gate.disagreement + 0.20 * (1.0 - gate.evidence_coverage),
    )
    actions = [
        action
        for action in _candidate_actions(lattice, global_uncertainty=global_uncertainty)
        if action.kind in enabled_set
    ]
    actions.sort(
        key=lambda action: (
            -action.utility,
            -action.semantic_criticality,
            -action.expected_information_gain,
            action.estimated_cost_ms,
            action.action_id,
        )
    )
    selected: list[EvidenceAction] = []
    rejected: list[EvidenceAction] = []
    used = 0
    per_island: dict[str, int] = {}
    for action in actions:
        island_id = action.action_id.split(":", 1)[0]
        # Avoid spending the whole budget on many mechanisms for one island.
        if per_island.get(island_id, 0) >= 2:
            rejected.append(action)
            continue
        if action.utility < budget.minimum_utility:
            rejected.append(action)
            continue
        if len(selected) >= budget.max_actions:
            rejected.append(action)
            continue
        if used + action.estimated_cost_ms > budget.total_cost_ms:
            rejected.append(action)
            continue
        selected.append(action)
        per_island[island_id] = per_island.get(island_id, 0) + 1
        used += action.estimated_cost_ms
    stopping_reason = (
        "budget-exhausted"
        if selected and used >= budget.total_cost_ms
        else "utility-frontier-reached"
        if selected
        else "no-positive-utility-action"
    )
    return EvidencePlan(
        selected=tuple(
            sorted(
                selected,
                key=lambda action: (
                    action.start_ms is None,
                    action.start_ms or 0,
                    action.kind,
                ),
            )
        ),
        rejected=tuple(rejected),
        budget_ms=budget.total_cost_ms,
        used_ms=used,
        expected_information_gain=sum(action.expected_information_gain for action in selected),
        stopping_reason=stopping_reason,
    )
