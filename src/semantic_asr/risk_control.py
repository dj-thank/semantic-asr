from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class RuntimeRiskState:
    entropy: float
    posterior_margin: float
    disagreement: float
    evidence_coverage: float
    semantic_criticality: float = 0.0
    available_candidates: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("entropy", self.entropy),
            ("posterior_margin", self.posterior_margin),
            ("disagreement", self.disagreement),
            ("evidence_coverage", self.evidence_coverage),
            ("semantic_criticality", self.semantic_criticality),
        ):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.available_candidates < 1:
            raise ValueError("available_candidates must be positive")


@dataclass(frozen=True, slots=True)
class AdaptivePolicy:
    policy_id: str
    candidate_count: int
    stages: tuple[str, ...] = ()
    maximum_cost_ms: float = math.inf
    minimum_entropy: float = 0.0
    maximum_margin: float = 1.0
    minimum_disagreement: float = 0.0
    maximum_coverage: float = 1.0
    minimum_criticality: float = 0.0
    conservative_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not math.isfinite(self.maximum_cost_ms) and self.maximum_cost_ms != math.inf:
            raise ValueError("maximum_cost_ms must be finite or infinity")
        if self.maximum_cost_ms < 0:
            raise ValueError("maximum_cost_ms must be non-negative")
        for name, value in (
            ("minimum_entropy", self.minimum_entropy),
            ("maximum_margin", self.maximum_margin),
            ("minimum_disagreement", self.minimum_disagreement),
            ("maximum_coverage", self.maximum_coverage),
            ("minimum_criticality", self.minimum_criticality),
        ):
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def applies(self, state: RuntimeRiskState) -> bool:
        return (
            state.available_candidates >= self.candidate_count
            and state.entropy >= self.minimum_entropy
            and state.posterior_margin <= self.maximum_margin
            and state.disagreement >= self.minimum_disagreement
            and state.evidence_coverage <= self.maximum_coverage
            and state.semantic_criticality >= self.minimum_criticality
        )


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    policy_id: str
    bounded_loss: float
    measured_cost_ms: float
    sample_id: str
    group: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_id or not self.sample_id:
            raise ValueError("policy_id and sample_id are required")
        if not math.isfinite(float(self.bounded_loss)) or not 0 <= self.bounded_loss <= 1:
            raise ValueError("bounded_loss must be in [0, 1]")
        if not math.isfinite(float(self.measured_cost_ms)) or self.measured_cost_ms < 0:
            raise ValueError("measured_cost_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PolicyRiskBound:
    policy_id: str
    samples: int
    empirical_risk: float
    upper_risk: float
    mean_cost_ms: float
    confidence_delta: float
    correction_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class RiskControlProfile:
    policies: tuple[AdaptivePolicy, ...]
    bounds: tuple[PolicyRiskBound, ...]
    target_risk: float
    delta: float
    calibration_digest: str
    selected_policy_ids: tuple[str, ...]
    fallback_policy_id: str
    method: str = "hoeffding-bonferroni-v1"

    def bound(self, policy_id: str) -> PolicyRiskBound:
        try:
            return next(bound for bound in self.bounds if bound.policy_id == policy_id)
        except StopIteration as exc:
            raise KeyError(policy_id) from exc

    @property
    def digest(self) -> str:
        payload = {
            "policies": [asdict(policy) for policy in self.policies],
            "bounds": [asdict(bound) for bound in self.bounds],
            "targetRisk": self.target_risk,
            "delta": self.delta,
            "calibrationDigest": self.calibration_digest,
            "selectedPolicyIds": self.selected_policy_ids,
            "fallbackPolicyId": self.fallback_policy_id,
            "method": self.method,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicySelection:
    policy: AdaptivePolicy
    bound: PolicyRiskBound
    eligible_policy_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    profile_digest: str


def _calibration_digest(outcomes: Sequence[PolicyOutcome]) -> str:
    rows = [
        {
            "policyId": outcome.policy_id,
            "loss": outcome.bounded_loss,
            "costMs": outcome.measured_cost_ms,
            "sampleId": outcome.sample_id,
            "group": outcome.group,
        }
        for outcome in sorted(
            outcomes,
            key=lambda row: (row.sample_id, row.policy_id, row.group or ""),
        )
    ]
    return hashlib.sha256(
        json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _hoeffding_upper(
    empirical_risk: float,
    samples: int,
    *,
    delta: float,
    correction_count: int,
) -> float:
    if samples < 1:
        return 1.0
    corrected_delta = delta / max(1, correction_count)
    radius = math.sqrt(math.log(1.0 / corrected_delta) / (2.0 * samples))
    return min(1.0, empirical_risk + radius)


def fit_risk_control(
    policies: Sequence[AdaptivePolicy],
    outcomes: Sequence[PolicyOutcome],
    *,
    target_risk: float,
    delta: float = 0.05,
    minimum_samples: int = 30,
) -> RiskControlProfile:
    """Fit a Learn-Then-Test style policy filter on held-out bounded losses.

    The upper bounds use Hoeffding with Bonferroni correction across the complete
    policy family. This is intentionally conservative and dependency-free. More
    powerful conformal/LTT procedures can be compared without changing the runtime
    profile contract.
    """

    if not policies:
        raise ValueError("at least one policy is required")
    if len({policy.policy_id for policy in policies}) != len(policies):
        raise ValueError("policy IDs must be unique")
    if not 0 < target_risk < 1 or not 0 < delta < 1:
        raise ValueError("target_risk and delta must be in (0, 1)")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")

    by_policy: dict[str, list[PolicyOutcome]] = {policy.policy_id: [] for policy in policies}
    for outcome in outcomes:
        if outcome.policy_id not in by_policy:
            raise ValueError(f"outcome references unknown policy: {outcome.policy_id}")
        by_policy[outcome.policy_id].append(outcome)

    correction_count = len(policies)
    bounds: list[PolicyRiskBound] = []
    for policy in policies:
        rows = by_policy[policy.policy_id]
        samples = len(rows)
        empirical = (
            sum(row.bounded_loss for row in rows) / samples if samples else 1.0
        )
        mean_cost = (
            sum(row.measured_cost_ms for row in rows) / samples if samples else math.inf
        )
        upper = _hoeffding_upper(
            empirical,
            samples,
            delta=delta,
            correction_count=correction_count,
        )
        bounds.append(
            PolicyRiskBound(
                policy_id=policy.policy_id,
                samples=samples,
                empirical_risk=empirical,
                upper_risk=upper,
                mean_cost_ms=mean_cost,
                confidence_delta=delta,
                correction_count=correction_count,
                passed=samples >= minimum_samples and upper <= target_risk,
            )
        )

    fallback_candidates = [policy for policy in policies if policy.conservative_fallback]
    if len(fallback_candidates) != 1:
        raise ValueError("exactly one conservative fallback policy is required")
    fallback = fallback_candidates[0]

    bound_by_id = {bound.policy_id: bound for bound in bounds}
    passing = [policy for policy in policies if bound_by_id[policy.policy_id].passed]
    passing.sort(
        key=lambda policy: (
            bound_by_id[policy.policy_id].mean_cost_ms,
            policy.candidate_count,
            policy.policy_id,
        )
    )
    return RiskControlProfile(
        policies=tuple(policies),
        bounds=tuple(sorted(bounds, key=lambda bound: bound.policy_id)),
        target_risk=target_risk,
        delta=delta,
        calibration_digest=_calibration_digest(outcomes),
        selected_policy_ids=tuple(policy.policy_id for policy in passing),
        fallback_policy_id=fallback.policy_id,
    )


def select_policy(
    profile: RiskControlProfile,
    state: RuntimeRiskState,
    *,
    cost_budget_ms: float = math.inf,
) -> PolicySelection:
    if cost_budget_ms < 0 or (
        not math.isfinite(cost_budget_ms) and cost_budget_ms != math.inf
    ):
        raise ValueError("cost_budget_ms must be non-negative or infinity")
    policies = {policy.policy_id: policy for policy in profile.policies}
    eligible: list[AdaptivePolicy] = []
    reasons: list[str] = []
    for policy_id in profile.selected_policy_ids:
        policy = policies[policy_id]
        bound = profile.bound(policy_id)
        if not policy.applies(state):
            continue
        if policy.maximum_cost_ms > cost_budget_ms:
            continue
        if bound.mean_cost_ms > cost_budget_ms:
            continue
        eligible.append(policy)
    eligible.sort(
        key=lambda policy: (
            profile.bound(policy.policy_id).mean_cost_ms,
            policy.candidate_count,
            profile.bound(policy.policy_id).upper_risk,
            policy.policy_id,
        )
    )
    if eligible:
        selected = eligible[0]
        reasons.append("held-out-risk-bound-passed")
        reasons.append("minimum-measured-cost")
    else:
        selected = policies[profile.fallback_policy_id]
        reasons.append("no-calibrated-policy-satisfied-state-and-budget")
        reasons.append("conservative-fallback")
    return PolicySelection(
        policy=selected,
        bound=profile.bound(selected.policy_id),
        eligible_policy_ids=tuple(policy.policy_id for policy in eligible),
        reasons=tuple(reasons),
        profile_digest=profile.digest,
    )


def pareto_frontier(bounds: Iterable[PolicyRiskBound]) -> tuple[PolicyRiskBound, ...]:
    """Return policies not dominated in upper risk and measured cost."""

    rows = tuple(bounds)
    frontier: list[PolicyRiskBound] = []
    for candidate in rows:
        dominated = any(
            other.policy_id != candidate.policy_id
            and other.upper_risk <= candidate.upper_risk
            and other.mean_cost_ms <= candidate.mean_cost_ms
            and (
                other.upper_risk < candidate.upper_risk
                or other.mean_cost_ms < candidate.mean_cost_ms
            )
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(
        sorted(
            frontier,
            key=lambda row: (row.mean_cost_ms, row.upper_risk, row.policy_id),
        )
    )


def make_default_policy_family(candidate_counts: Sequence[int] = (1, 3, 5, 8, 16)) -> tuple[AdaptivePolicy, ...]:
    unique = tuple(sorted(set(int(value) for value in candidate_counts)))
    if not unique or unique[0] < 1:
        raise ValueError("candidate_counts must contain positive integers")
    policies: list[AdaptivePolicy] = []
    for count in unique:
        if count == 1:
            policies.append(
                AdaptivePolicy(
                    policy_id="k1-cheap",
                    candidate_count=1,
                    stages=("acoustic",),
                    maximum_cost_ms=1_500,
                    maximum_margin=1.0,
                    maximum_coverage=1.0,
                )
            )
            continue
        policies.append(
            AdaptivePolicy(
                policy_id=f"k{count}-rerank",
                candidate_count=count,
                stages=("acoustic", "ngram", "mbr", "compact-reranker"),
                maximum_cost_ms=4_000 + 250 * count,
                minimum_entropy=0.10,
                maximum_margin=0.90,
                maximum_coverage=1.0,
            )
        )
    maximum = unique[-1]
    policies.append(
        AdaptivePolicy(
            policy_id=f"k{maximum}-verified-fallback",
            candidate_count=maximum,
            stages=(
                "acoustic",
                "ngram",
                "mbr",
                "compact-reranker",
                "selective-relisten",
                "acoustic-verifier",
            ),
            maximum_cost_ms=math.inf,
            conservative_fallback=True,
        )
    )
    return tuple(policies)
