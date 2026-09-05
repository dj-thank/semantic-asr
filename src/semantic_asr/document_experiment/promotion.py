"""Fail-closed promotion gate for document-context experiment reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from .metrics import ArmAggregateMetrics, PairedBootstrapInterval
from .runner import DocumentContextExperimentReport


@dataclass(frozen=True, slots=True)
class DocumentContextPromotionPolicy:
    target_arm: str
    baseline_arm: str
    shuffled_control_arm: str
    minimum_absolute_strict_cer_reduction: float = 0.0
    maximum_bootstrap_upper_delta: float = 0.0
    minimum_ordered_advantage_over_shuffled: float = 0.0
    maximum_critical_error_regression: int = 0
    maximum_false_correction_regression: int = 0
    maximum_introduced_error_regression: int = 0
    minimum_coverage: float = 0.5
    maximum_mean_latency_ms: float = 60_000.0
    require_no_case_failures: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("target_arm", "baseline_arm", "shuffled_control_arm"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if len({self.target_arm, self.baseline_arm, self.shuffled_control_arm}) != 3:
            raise ValueError("target, baseline, and shuffled-control arms must be distinct")
        for name in (
            "minimum_absolute_strict_cer_reduction",
            "minimum_ordered_advantage_over_shuffled",
            "minimum_coverage",
            "maximum_mean_latency_ms",
        ):
            value = _strict_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        upper = _strict_float(
            self.maximum_bootstrap_upper_delta,
            name="maximum_bootstrap_upper_delta",
        )
        object.__setattr__(self, "maximum_bootstrap_upper_delta", upper)
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in [0, 1]")
        for name in (
            "maximum_critical_error_regression",
            "maximum_false_correction_regression",
            "maximum_introduced_error_regression",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.require_no_case_failures, bool):
            raise TypeError("require_no_case_failures must be a boolean")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PromotionCheck:
    name: str
    passed: bool
    observed: float | int | str
    required: float | int | str
    detail: str

    def __post_init__(self) -> None:
        if not self.name or not self.detail:
            raise ValueError("promotion check requires name and detail")
        if not isinstance(self.passed, bool):
            raise TypeError("promotion check passed must be a boolean")


@dataclass(frozen=True, slots=True)
class DocumentContextPromotionDecision:
    passed: bool
    policy_digest: str
    report_digest: str
    target_arm: str
    baseline_arm: str
    shuffled_control_arm: str
    checks: tuple[PromotionCheck, ...]

    def __post_init__(self) -> None:
        if not _is_sha256(self.policy_digest) or not _is_sha256(self.report_digest):
            raise ValueError("promotion decision digests must be SHA-256 values")
        if not self.checks:
            raise ValueError("promotion decision requires checks")
        if self.passed != all(check.passed for check in self.checks):
            raise ValueError("promotion decision does not match its checks")

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(check.detail for check in self.checks if not check.passed)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "passed": self.passed,
                "policyDigest": self.policy_digest,
                "reportDigest": self.report_digest,
                "targetArm": self.target_arm,
                "baselineArm": self.baseline_arm,
                "shuffledControlArm": self.shuffled_control_arm,
                "checks": [asdict(check) for check in self.checks],
            }
        )


def _aggregate(report: DocumentContextExperimentReport, name: str) -> ArmAggregateMetrics:
    for row in report.aggregates:
        if row.arm_name == name:
            return row
    raise ValueError(f"experiment report is missing aggregate arm {name!r}")


def _interval(
    report: DocumentContextExperimentReport,
    target: str,
    baseline: str,
) -> PairedBootstrapInterval:
    for row in report.paired_intervals:
        if row.arm_name == target and row.baseline_arm == baseline:
            return row
    raise ValueError("experiment report is missing the target/baseline bootstrap interval")


def evaluate_document_context_promotion(
    report: DocumentContextExperimentReport,
    policy: DocumentContextPromotionPolicy,
) -> DocumentContextPromotionDecision:
    target = _aggregate(report, policy.target_arm)
    baseline = _aggregate(report, policy.baseline_arm)
    shuffled = _aggregate(report, policy.shuffled_control_arm)
    interval = _interval(report, policy.target_arm, policy.baseline_arm)
    strict_reduction = baseline.strict_cer - target.strict_cer
    ordered_advantage = shuffled.strict_cer - target.strict_cer
    critical_regression = target.critical_token_errors - baseline.critical_token_errors
    false_correction_regression = (
        target.false_correction_windows - baseline.false_correction_windows
    )
    introduced_regression = (
        target.introduced_error_characters - baseline.introduced_error_characters
    )
    mean_latency = target.total_latency_ms / target.case_count
    checks = (
        PromotionCheck(
            name="strict-cer-reduction",
            passed=strict_reduction >= policy.minimum_absolute_strict_cer_reduction,
            observed=strict_reduction,
            required=policy.minimum_absolute_strict_cer_reduction,
            detail="target arm did not achieve the preregistered strict CER reduction",
        ),
        PromotionCheck(
            name="paired-bootstrap-upper",
            passed=interval.upper <= policy.maximum_bootstrap_upper_delta,
            observed=interval.upper,
            required=policy.maximum_bootstrap_upper_delta,
            detail="paired bootstrap upper bound does not exclude the allowed regression",
        ),
        PromotionCheck(
            name="ordered-vs-shuffled",
            passed=ordered_advantage >= policy.minimum_ordered_advantage_over_shuffled,
            observed=ordered_advantage,
            required=policy.minimum_ordered_advantage_over_shuffled,
            detail="ordered context does not outperform the shuffled-context control",
        ),
        PromotionCheck(
            name="critical-error-regression",
            passed=critical_regression <= policy.maximum_critical_error_regression,
            observed=critical_regression,
            required=policy.maximum_critical_error_regression,
            detail="semantic-critical token errors regressed",
        ),
        PromotionCheck(
            name="false-correction-regression",
            passed=(
                false_correction_regression
                <= policy.maximum_false_correction_regression
            ),
            observed=false_correction_regression,
            required=policy.maximum_false_correction_regression,
            detail="context-induced false-correction windows regressed",
        ),
        PromotionCheck(
            name="introduced-error-regression",
            passed=introduced_regression <= policy.maximum_introduced_error_regression,
            observed=introduced_regression,
            required=policy.maximum_introduced_error_regression,
            detail="introduced error characters regressed",
        ),
        PromotionCheck(
            name="coverage",
            passed=target.coverage >= policy.minimum_coverage,
            observed=target.coverage,
            required=policy.minimum_coverage,
            detail="accepted coverage is below the preregistered minimum",
        ),
        PromotionCheck(
            name="latency",
            passed=mean_latency <= policy.maximum_mean_latency_ms,
            observed=mean_latency,
            required=policy.maximum_mean_latency_ms,
            detail="mean target-arm latency exceeds the preregistered limit",
        ),
        PromotionCheck(
            name="case-failures",
            passed=(not report.failures or not policy.require_no_case_failures),
            observed=len(report.failures),
            required=0 if policy.require_no_case_failures else "not enforced",
            detail="one or more case/arm evaluations failed",
        ),
    )
    return DocumentContextPromotionDecision(
        passed=all(check.passed for check in checks),
        policy_digest=policy.digest,
        report_digest=report.digest,
        target_arm=policy.target_arm,
        baseline_arm=policy.baseline_arm,
        shuffled_control_arm=policy.shuffled_control_arm,
        checks=checks,
    )
