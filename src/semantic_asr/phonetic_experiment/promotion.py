"""Conjunctive promotion gate for phonetic proposal evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from .runner import PhoneticAblationReport


@dataclass(frozen=True, slots=True)
class PhoneticPromotionPolicy:
    target_arm: str
    baseline_arm: str
    minimum_exact_accuracy_gain: float = 0.0
    maximum_bootstrap_upper_error_delta: float = 0.0
    minimum_oracle_coverage: float = 0.95
    minimum_outside_first_pass_recovery_rate: float = 0.05
    maximum_false_correction_rate: float = 0.01
    maximum_introduced_error_characters: int = 0
    minimum_critical_exact_accuracy: float = 0.0
    minimum_accepted_coverage: float = 0.0
    maximum_mean_generation_latency_ms: float = 5_000.0
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.target_arm or not self.baseline_arm:
            raise ValueError("promotion policy requires target and baseline arms")
        for name in (
            "minimum_oracle_coverage",
            "minimum_outside_first_pass_recovery_rate",
            "maximum_false_correction_rate",
            "minimum_critical_exact_accuracy",
            "minimum_accepted_coverage",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_exact_accuracy_gain",
            "maximum_bootstrap_upper_error_delta",
            "maximum_mean_generation_latency_ms",
        ):
            value = float(getattr(self, name))
            if name == "maximum_mean_generation_latency_ms" and value < 0.0:
                raise ValueError("maximum_mean_generation_latency_ms must be non-negative")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.maximum_introduced_error_characters, bool)
            or not isinstance(self.maximum_introduced_error_characters, int)
            or self.maximum_introduced_error_characters < 0
        ):
            raise ValueError("maximum_introduced_error_characters must be non-negative")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    name: str
    passed: bool
    observed: float | int
    required: float | int
    relation: str


@dataclass(frozen=True, slots=True)
class PhoneticPromotionDecision:
    promote: bool
    target_arm: str
    baseline_arm: str
    policy_digest: str
    report_digest: str
    checks: tuple[PromotionCheck, ...]
    reasons: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "checks": [asdict(row) for row in self.checks],
            }
        )


def evaluate_phonetic_promotion(
    report: PhoneticAblationReport,
    policy: PhoneticPromotionPolicy,
) -> PhoneticPromotionDecision:
    aggregates = {row.arm_name: row for row in report.aggregates}
    if policy.target_arm not in aggregates or policy.baseline_arm not in aggregates:
        raise ValueError("promotion policy references an absent ablation arm")
    target = aggregates[policy.target_arm]
    baseline = aggregates[policy.baseline_arm]
    delta = next(
        (
            row
            for row in report.paired_deltas
            if row.target_arm == policy.target_arm and row.baseline_arm == policy.baseline_arm
        ),
        None,
    )
    if delta is None:
        raise ValueError("promotion report lacks the required paired bootstrap comparison")
    checks = (
        PromotionCheck(
            name="exact-accuracy-gain",
            passed=(
                target.exact_accuracy - baseline.exact_accuracy
                >= policy.minimum_exact_accuracy_gain
            ),
            observed=target.exact_accuracy - baseline.exact_accuracy,
            required=policy.minimum_exact_accuracy_gain,
            relation=">=",
        ),
        PromotionCheck(
            name="paired-bootstrap-upper-error-delta",
            passed=delta.upper_95 <= policy.maximum_bootstrap_upper_error_delta,
            observed=delta.upper_95,
            required=policy.maximum_bootstrap_upper_error_delta,
            relation="<=",
        ),
        PromotionCheck(
            name="oracle-coverage",
            passed=target.oracle_coverage >= policy.minimum_oracle_coverage,
            observed=target.oracle_coverage,
            required=policy.minimum_oracle_coverage,
            relation=">=",
        ),
        PromotionCheck(
            name="outside-first-pass-recovery",
            passed=(
                target.outside_first_pass_recovery_rate
                >= policy.minimum_outside_first_pass_recovery_rate
            ),
            observed=target.outside_first_pass_recovery_rate,
            required=policy.minimum_outside_first_pass_recovery_rate,
            relation=">=",
        ),
        PromotionCheck(
            name="false-correction-rate",
            passed=target.false_correction_rate <= policy.maximum_false_correction_rate,
            observed=target.false_correction_rate,
            required=policy.maximum_false_correction_rate,
            relation="<=",
        ),
        PromotionCheck(
            name="introduced-error-characters",
            passed=(
                target.total_introduced_error_characters
                <= policy.maximum_introduced_error_characters
            ),
            observed=target.total_introduced_error_characters,
            required=policy.maximum_introduced_error_characters,
            relation="<=",
        ),
        PromotionCheck(
            name="critical-exact-accuracy",
            passed=(target.critical_exact_accuracy >= policy.minimum_critical_exact_accuracy),
            observed=target.critical_exact_accuracy,
            required=policy.minimum_critical_exact_accuracy,
            relation=">=",
        ),
        PromotionCheck(
            name="accepted-coverage",
            passed=target.accepted_coverage >= policy.minimum_accepted_coverage,
            observed=target.accepted_coverage,
            required=policy.minimum_accepted_coverage,
            relation=">=",
        ),
        PromotionCheck(
            name="generation-latency",
            passed=(target.mean_generation_latency_ms <= policy.maximum_mean_generation_latency_ms),
            observed=target.mean_generation_latency_ms,
            required=policy.maximum_mean_generation_latency_ms,
            relation="<=",
        ),
    )
    failed = tuple(row.name for row in checks if not row.passed)
    return PhoneticPromotionDecision(
        promote=not failed,
        target_arm=policy.target_arm,
        baseline_arm=policy.baseline_arm,
        policy_digest=policy.digest,
        report_digest=report.digest,
        checks=checks,
        reasons=failed if failed else ("all-conjunctive-promotion-checks-passed",),
    )
