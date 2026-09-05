"""Fail-closed promotion gate for context × phonetic factorial results."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from .runner import ContextPhoneticFactorialReport


@dataclass(frozen=True, slots=True)
class ContextPhoneticPromotionPolicy:
    target_arm: str
    baseline_arm: str
    shuffled_control_arm: str
    minimum_exact_accuracy_gain: float = 0.0
    maximum_combined_vs_baseline_upper_error_delta: float = 0.0
    maximum_ordered_vs_shuffled_upper_error_delta: float = 0.0
    maximum_interaction_upper_error: float = 0.0
    minimum_oracle_coverage: float = 0.95
    minimum_outside_first_pass_recovery_rate: float = 0.05
    maximum_false_correction_rate: float = 0.01
    maximum_false_correction_rate_delta: float = 0.0
    maximum_introduced_error_characters: int = 0
    minimum_critical_exact_accuracy: float = 0.0
    minimum_accepted_coverage: float = 0.0
    maximum_total_runtime_ms: float = 5_000.0
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.target_arm or not self.baseline_arm or not self.shuffled_control_arm:
            raise ValueError("promotion policy requires target, baseline, and shuffled arms")
        bounded = (
            "minimum_oracle_coverage",
            "minimum_outside_first_pass_recovery_rate",
            "maximum_false_correction_rate",
            "minimum_critical_exact_accuracy",
            "minimum_accepted_coverage",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
            object.__setattr__(self, name, value)
        for name in (
            "minimum_exact_accuracy_gain",
            "maximum_combined_vs_baseline_upper_error_delta",
            "maximum_ordered_vs_shuffled_upper_error_delta",
            "maximum_interaction_upper_error",
            "maximum_false_correction_rate_delta",
            "maximum_total_runtime_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "maximum_total_runtime_ms" and value < 0.0:
                raise ValueError("maximum_total_runtime_ms must be non-negative")
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
class ContextPhoneticPromotionCheck:
    name: str
    passed: bool
    observed: float | int
    required: float | int
    relation: str


@dataclass(frozen=True, slots=True)
class ContextPhoneticPromotionDecision:
    promote: bool
    target_arm: str
    baseline_arm: str
    shuffled_control_arm: str
    policy_digest: str
    report_digest: str
    checks: tuple[ContextPhoneticPromotionCheck, ...]
    reasons: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "checks": [asdict(row) for row in self.checks],
            }
        )


def _contrast(report: ContextPhoneticFactorialReport, name: str):
    rows = [row for row in report.contrasts if row.name == name]
    if len(rows) != 1:
        raise ValueError(f"factorial report requires exactly one contrast named {name!r}")
    return rows[0]


def evaluate_context_phonetic_promotion(
    report: ContextPhoneticFactorialReport,
    policy: ContextPhoneticPromotionPolicy,
) -> ContextPhoneticPromotionDecision:
    aggregates = {row.arm_name: row for row in report.aggregates}
    for name in (policy.target_arm, policy.baseline_arm, policy.shuffled_control_arm):
        if name not in aggregates:
            raise ValueError(f"promotion policy references absent factorial arm: {name}")
    target = aggregates[policy.target_arm]
    baseline = aggregates[policy.baseline_arm]
    shuffled = aggregates[policy.shuffled_control_arm]
    combined = _contrast(report, "combined-vs-baseline")
    specificity = _contrast(report, "ordered-vs-shuffled")
    if combined.target_arm != policy.target_arm or combined.baseline_arm != policy.baseline_arm:
        raise ValueError("combined-vs-baseline contrast does not match promotion policy")
    if (
        specificity.target_arm != policy.target_arm
        or specificity.baseline_arm != policy.shuffled_control_arm
    ):
        raise ValueError("ordered-vs-shuffled contrast does not match promotion policy")
    checks = (
        ContextPhoneticPromotionCheck(
            name="exact-accuracy-gain",
            passed=(
                target.exact_accuracy - baseline.exact_accuracy
                >= policy.minimum_exact_accuracy_gain
            ),
            observed=target.exact_accuracy - baseline.exact_accuracy,
            required=policy.minimum_exact_accuracy_gain,
            relation=">=",
        ),
        ContextPhoneticPromotionCheck(
            name="combined-vs-baseline-bootstrap-upper",
            passed=(combined.upper_95 <= policy.maximum_combined_vs_baseline_upper_error_delta),
            observed=combined.upper_95,
            required=policy.maximum_combined_vs_baseline_upper_error_delta,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="ordered-vs-shuffled-bootstrap-upper",
            passed=(specificity.upper_95 <= policy.maximum_ordered_vs_shuffled_upper_error_delta),
            observed=specificity.upper_95,
            required=policy.maximum_ordered_vs_shuffled_upper_error_delta,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="context-phonetic-interaction-upper",
            passed=(report.interaction.upper_95 <= policy.maximum_interaction_upper_error),
            observed=report.interaction.upper_95,
            required=policy.maximum_interaction_upper_error,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="oracle-coverage",
            passed=target.oracle_coverage >= policy.minimum_oracle_coverage,
            observed=target.oracle_coverage,
            required=policy.minimum_oracle_coverage,
            relation=">=",
        ),
        ContextPhoneticPromotionCheck(
            name="outside-first-pass-recovery",
            passed=(
                target.outside_first_pass_recovery_rate
                >= policy.minimum_outside_first_pass_recovery_rate
            ),
            observed=target.outside_first_pass_recovery_rate,
            required=policy.minimum_outside_first_pass_recovery_rate,
            relation=">=",
        ),
        ContextPhoneticPromotionCheck(
            name="false-correction-rate",
            passed=target.false_correction_rate <= policy.maximum_false_correction_rate,
            observed=target.false_correction_rate,
            required=policy.maximum_false_correction_rate,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="false-correction-rate-delta",
            passed=(
                target.false_correction_rate - baseline.false_correction_rate
                <= policy.maximum_false_correction_rate_delta
            ),
            observed=target.false_correction_rate - baseline.false_correction_rate,
            required=policy.maximum_false_correction_rate_delta,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="introduced-error-characters",
            passed=(
                target.total_introduced_error_characters
                <= policy.maximum_introduced_error_characters
            ),
            observed=target.total_introduced_error_characters,
            required=policy.maximum_introduced_error_characters,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="critical-exact-accuracy",
            passed=(target.critical_exact_accuracy >= policy.minimum_critical_exact_accuracy),
            observed=target.critical_exact_accuracy,
            required=policy.minimum_critical_exact_accuracy,
            relation=">=",
        ),
        ContextPhoneticPromotionCheck(
            name="accepted-coverage",
            passed=target.accepted_coverage >= policy.minimum_accepted_coverage,
            observed=target.accepted_coverage,
            required=policy.minimum_accepted_coverage,
            relation=">=",
        ),
        ContextPhoneticPromotionCheck(
            name="total-runtime",
            passed=target.total_runtime_ms <= policy.maximum_total_runtime_ms,
            observed=target.total_runtime_ms,
            required=policy.maximum_total_runtime_ms,
            relation="<=",
        ),
        ContextPhoneticPromotionCheck(
            name="shuffled-control-not-better-than-ordered",
            passed=shuffled.character_error_rate >= target.character_error_rate,
            observed=shuffled.character_error_rate - target.character_error_rate,
            required=0.0,
            relation=">=",
        ),
    )
    failed = tuple(row.name for row in checks if not row.passed)
    return ContextPhoneticPromotionDecision(
        promote=not failed,
        target_arm=policy.target_arm,
        baseline_arm=policy.baseline_arm,
        shuffled_control_arm=policy.shuffled_control_arm,
        policy_digest=policy.digest,
        report_digest=report.digest,
        checks=checks,
        reasons=failed if failed else ("all-conjunctive-promotion-checks-passed",),
    )
