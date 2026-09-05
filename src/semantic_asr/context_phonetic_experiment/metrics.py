"""Reference-opened metrics and grouped factorial contrasts."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_experiment.metrics import edit_distance
from .planner import PreparedContextPhoneticCase
from .protocol import ContextPhoneticArm, ContextPhoneticCase, ContextPhoneticProtocol
from .selection import ContextPhoneticDecision


@dataclass(frozen=True, slots=True)
class ContextPhoneticCaseMetrics:
    case_id: str
    group_id: str
    arm_name: str
    phonetic_arm_name: str
    context_condition: str
    context_donor_case_id: str | None
    reference_text_sha256: str
    first_pass_text_sha256: str
    proposed_text_sha256: str
    effective_text_sha256: str
    reference_characters: int
    first_pass_edits: int
    proposed_edits: int
    effective_edits: int
    pool_oracle: bool
    reference_outside_first_pass: bool
    proposed_exact: bool
    effective_exact: bool
    recovered_outside_first_pass: bool
    false_correction: bool
    corrected_first_pass: bool
    introduced_error_characters: int
    corrected_error_characters: int
    accepted: bool
    changed_proposal: bool
    changed_effective: bool
    critical: bool
    margin: float
    pool_generation_latency_ms: float
    context_scoring_latency_ms: float
    selection_latency_ms: float

    def __post_init__(self) -> None:
        if not self.case_id or not self.group_id or not self.arm_name:
            raise ValueError("factorial case metrics require case, group, and arm identities")
        if self.context_condition not in {"none", "ordered", "shuffled"}:
            raise ValueError("factorial case metric context condition is invalid")
        if self.context_condition == "none" and self.context_donor_case_id is not None:
            raise ValueError("none-context metrics must not name a context donor")
        if self.context_condition != "none" and not self.context_donor_case_id:
            raise ValueError("context metrics require a donor case ID")
        for digest in (
            self.reference_text_sha256,
            self.first_pass_text_sha256,
            self.proposed_text_sha256,
            self.effective_text_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("factorial case text digests must be SHA-256 values")
        if self.reference_characters < 1:
            raise ValueError("reference_characters must be positive")
        for name in (
            "first_pass_edits",
            "proposed_edits",
            "effective_edits",
            "introduced_error_characters",
            "corrected_error_characters",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in (
            "margin",
            "pool_generation_latency_ms",
            "context_scoring_latency_ms",
            "selection_latency_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def first_pass_exact(self) -> bool:
        return self.first_pass_edits == 0

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "pool_generation_latency_ms": None,
                "context_scoring_latency_ms": None,
                "selection_latency_ms": None,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPhoneticArmAggregate:
    arm_name: str
    phonetic_arm_name: str
    context_condition: str
    case_count: int
    exact_count: int
    proposed_exact_count: int
    first_pass_exact_count: int
    oracle_count: int
    outside_first_pass_case_count: int
    outside_first_pass_recovery_count: int
    false_correction_count: int
    corrected_first_pass_count: int
    critical_case_count: int
    critical_exact_count: int
    accepted_count: int
    changed_effective_count: int
    total_reference_characters: int
    total_first_pass_edits: int
    total_effective_edits: int
    total_introduced_error_characters: int
    total_corrected_error_characters: int
    mean_margin: float
    mean_pool_generation_latency_ms: float
    mean_context_scoring_latency_ms: float
    mean_selection_latency_ms: float

    def __post_init__(self) -> None:
        if not self.arm_name or not self.phonetic_arm_name:
            raise ValueError("factorial aggregate requires arm identities")
        if self.context_condition not in {"none", "ordered", "shuffled"}:
            raise ValueError("factorial aggregate context condition is invalid")
        count_fields = (
            "case_count",
            "exact_count",
            "proposed_exact_count",
            "first_pass_exact_count",
            "oracle_count",
            "outside_first_pass_case_count",
            "outside_first_pass_recovery_count",
            "false_correction_count",
            "corrected_first_pass_count",
            "critical_case_count",
            "critical_exact_count",
            "accepted_count",
            "changed_effective_count",
            "total_reference_characters",
            "total_first_pass_edits",
            "total_effective_edits",
            "total_introduced_error_characters",
            "total_corrected_error_characters",
        )
        for name in count_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.case_count < 1 or self.total_reference_characters < 1:
            raise ValueError("factorial aggregate requires cases and reference characters")
        bounded_counts = (
            self.exact_count,
            self.proposed_exact_count,
            self.first_pass_exact_count,
            self.oracle_count,
            self.false_correction_count,
            self.corrected_first_pass_count,
            self.critical_case_count,
            self.critical_exact_count,
            self.accepted_count,
            self.changed_effective_count,
        )
        if any(value > self.case_count for value in bounded_counts):
            raise ValueError("factorial aggregate count exceeds case_count")
        if self.outside_first_pass_recovery_count > self.outside_first_pass_case_count:
            raise ValueError("outside-first-pass recovery exceeds eligible cases")
        if self.false_correction_count > self.first_pass_exact_count:
            raise ValueError("false corrections exceed first-pass exact cases")
        if self.critical_exact_count > self.critical_case_count:
            raise ValueError("critical exact count exceeds critical cases")
        for name in (
            "mean_margin",
            "mean_pool_generation_latency_ms",
            "mean_context_scoring_latency_ms",
            "mean_selection_latency_ms",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def exact_accuracy(self) -> float:
        return self.exact_count / self.case_count

    @property
    def proposed_exact_accuracy(self) -> float:
        return self.proposed_exact_count / self.case_count

    @property
    def oracle_coverage(self) -> float:
        return self.oracle_count / self.case_count

    @property
    def outside_first_pass_recovery_rate(self) -> float:
        if not self.outside_first_pass_case_count:
            return 0.0
        return self.outside_first_pass_recovery_count / self.outside_first_pass_case_count

    @property
    def false_correction_rate(self) -> float:
        if not self.first_pass_exact_count:
            return 0.0
        return self.false_correction_count / self.first_pass_exact_count

    @property
    def critical_exact_accuracy(self) -> float:
        if not self.critical_case_count:
            return 0.0
        return self.critical_exact_count / self.critical_case_count

    @property
    def accepted_coverage(self) -> float:
        return self.accepted_count / self.case_count

    @property
    def character_error_rate(self) -> float:
        return self.total_effective_edits / self.total_reference_characters

    @property
    def first_pass_character_error_rate(self) -> float:
        return self.total_first_pass_edits / self.total_reference_characters

    @property
    def total_runtime_ms(self) -> float:
        return (
            self.mean_pool_generation_latency_ms
            + self.mean_context_scoring_latency_ms
            + self.mean_selection_latency_ms
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "exactAccuracy": self.exact_accuracy,
                "proposedExactAccuracy": self.proposed_exact_accuracy,
                "oracleCoverage": self.oracle_coverage,
                "outsideFirstPassRecoveryRate": self.outside_first_pass_recovery_rate,
                "falseCorrectionRate": self.false_correction_rate,
                "criticalExactAccuracy": self.critical_exact_accuracy,
                "acceptedCoverage": self.accepted_coverage,
                "characterErrorRate": self.character_error_rate,
                "firstPassCharacterErrorRate": self.first_pass_character_error_rate,
                "totalRuntimeMs": self.total_runtime_ms,
            }
        )


@dataclass(frozen=True, slots=True)
class GroupedPairedContrast:
    name: str
    target_arm: str
    baseline_arm: str
    mean_character_error_delta: float
    lower_95: float
    upper_95: float
    exact_accuracy_delta: float
    false_correction_rate_delta: float
    group_count: int
    resamples: int
    seed: str

    def __post_init__(self) -> None:
        if not self.name or not self.target_arm or not self.baseline_arm:
            raise ValueError("paired contrast requires name and arm identities")
        if self.group_count < 1 or self.resamples < 1 or not self.seed:
            raise ValueError("paired contrast group/resample/seed values are invalid")
        for name in (
            "mean_character_error_delta",
            "lower_95",
            "upper_95",
            "exact_accuracy_delta",
            "false_correction_rate_delta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not self.lower_95 <= self.mean_character_error_delta <= self.upper_95:
            raise ValueError("paired contrast point estimate lies outside its interval")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class FactorialInteractionContrast:
    name: str
    combined_ordered_arm: str
    combined_none_arm: str
    baseline_ordered_arm: str
    baseline_none_arm: str
    mean_error_interaction: float
    lower_95: float
    upper_95: float
    group_count: int
    resamples: int
    seed: str

    def __post_init__(self) -> None:
        if not self.name or self.group_count < 1 or self.resamples < 1:
            raise ValueError("factorial interaction identity/counts are invalid")
        for name in ("mean_error_interaction", "lower_95", "upper_95"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not self.lower_95 <= self.mean_error_interaction <= self.upper_95:
            raise ValueError("factorial interaction point estimate lies outside its interval")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def _group_id(case: ContextPhoneticCase, protocol: ContextPhoneticProtocol) -> str:
    source = case.phonetic_case
    if protocol.bootstrap_group == "speaker":
        return source.speaker_id
    if protocol.bootstrap_group == "session":
        return source.session_id
    if protocol.bootstrap_group == "source":
        return source.source_id
    raise ValueError("unknown factorial bootstrap group")


def evaluate_factorial_case_arm(
    prepared: PreparedContextPhoneticCase,
    decision: ContextPhoneticDecision,
    case: ContextPhoneticCase,
    arm: ContextPhoneticArm,
    protocol: ContextPhoneticProtocol,
) -> ContextPhoneticCaseMetrics:
    if decision.prepared_case_digest != prepared.digest:
        raise ValueError("factorial decision belongs to a different prepared case")
    if decision.arm_name != arm.name or decision.arm_digest != arm.digest:
        raise ValueError("factorial decision belongs to a different arm")
    pool = prepared.pool
    proposed = pool.candidate(decision.proposed_candidate_id)
    effective = pool.candidate(decision.effective_candidate_id)
    first_pass = pool.candidate(pool.first_pass_selected_candidate_id)
    reference = case.phonetic_case.reference
    first_pass_surfaces = {row.text for row in pool.candidates if row.is_first_pass}
    pool_surfaces = {row.text for row in pool.candidates}
    first_edits = edit_distance(first_pass.text, reference.text)
    proposed_edits = edit_distance(proposed.text, reference.text)
    effective_edits = edit_distance(effective.text, reference.text)
    if arm.context_condition == "none":
        donor = None
        context_latency = 0.0
    elif arm.context_condition == "ordered":
        donor = prepared.ordered.donor_case_id
        context_latency = prepared.ordered.scoring_latency_ms
    else:
        donor = prepared.shuffled.donor_case_id
        context_latency = prepared.shuffled.scoring_latency_ms
    return ContextPhoneticCaseMetrics(
        case_id=case.case_id,
        group_id=_group_id(case, protocol),
        arm_name=arm.name,
        phonetic_arm_name=arm.phonetic_arm_name,
        context_condition=arm.context_condition,
        context_donor_case_id=donor,
        reference_text_sha256=reference.text_sha256,
        first_pass_text_sha256=first_pass.text_sha256,
        proposed_text_sha256=proposed.text_sha256,
        effective_text_sha256=effective.text_sha256,
        reference_characters=len(reference.text),
        first_pass_edits=first_edits,
        proposed_edits=proposed_edits,
        effective_edits=effective_edits,
        pool_oracle=reference.text in pool_surfaces,
        reference_outside_first_pass=reference.text not in first_pass_surfaces,
        proposed_exact=proposed.text == reference.text,
        effective_exact=effective.text == reference.text,
        recovered_outside_first_pass=(
            reference.text not in first_pass_surfaces and effective.text == reference.text
        ),
        false_correction=(first_pass.text == reference.text and effective.text != reference.text),
        corrected_first_pass=(
            first_pass.text != reference.text and effective.text == reference.text
        ),
        introduced_error_characters=max(0, effective_edits - first_edits),
        corrected_error_characters=max(0, first_edits - effective_edits),
        accepted=decision.status == "accepted",
        changed_proposal=decision.changed_proposal,
        changed_effective=decision.changed_effective,
        critical=reference.critical,
        margin=decision.margin,
        pool_generation_latency_ms=pool.generation_latency_ms,
        context_scoring_latency_ms=context_latency,
        selection_latency_ms=decision.selection_latency_ms,
    )


def aggregate_factorial_arm(
    rows: tuple[ContextPhoneticCaseMetrics, ...],
) -> ContextPhoneticArmAggregate:
    if not rows:
        raise ValueError("cannot aggregate an empty factorial arm")
    identities = {(row.arm_name, row.phonetic_arm_name, row.context_condition) for row in rows}
    if len(identities) != 1:
        raise ValueError("factorial aggregate cannot mix arm identities")
    arm_name, phonetic_arm_name, context_condition = next(iter(identities))
    return ContextPhoneticArmAggregate(
        arm_name=arm_name,
        phonetic_arm_name=phonetic_arm_name,
        context_condition=context_condition,
        case_count=len(rows),
        exact_count=sum(row.effective_exact for row in rows),
        proposed_exact_count=sum(row.proposed_exact for row in rows),
        first_pass_exact_count=sum(row.first_pass_exact for row in rows),
        oracle_count=sum(row.pool_oracle for row in rows),
        outside_first_pass_case_count=sum(row.reference_outside_first_pass for row in rows),
        outside_first_pass_recovery_count=sum(row.recovered_outside_first_pass for row in rows),
        false_correction_count=sum(row.false_correction for row in rows),
        corrected_first_pass_count=sum(row.corrected_first_pass for row in rows),
        critical_case_count=sum(row.critical for row in rows),
        critical_exact_count=sum(row.critical and row.effective_exact for row in rows),
        accepted_count=sum(row.accepted for row in rows),
        changed_effective_count=sum(row.changed_effective for row in rows),
        total_reference_characters=sum(row.reference_characters for row in rows),
        total_first_pass_edits=sum(row.first_pass_edits for row in rows),
        total_effective_edits=sum(row.effective_edits for row in rows),
        total_introduced_error_characters=sum(row.introduced_error_characters for row in rows),
        total_corrected_error_characters=sum(row.corrected_error_characters for row in rows),
        mean_margin=sum(row.margin for row in rows) / len(rows),
        mean_pool_generation_latency_ms=sum(row.pool_generation_latency_ms for row in rows)
        / len(rows),
        mean_context_scoring_latency_ms=sum(row.context_scoring_latency_ms for row in rows)
        / len(rows),
        mean_selection_latency_ms=sum(row.selection_latency_ms for row in rows) / len(rows),
    )


def _grouped_bootstrap(
    arms: tuple[tuple[ContextPhoneticCaseMetrics, ...], ...],
    *,
    statistic: Callable[[tuple[tuple[ContextPhoneticCaseMetrics, ...], ...]], float],
    resamples: int,
    seed: str,
) -> tuple[float, float, float, int]:
    if resamples < 1 or not seed:
        raise ValueError("grouped bootstrap requires resamples and seed")
    maps = tuple({row.case_id: row for row in rows} for rows in arms)
    if any(len(mapping) != len(rows) for mapping, rows in zip(maps, arms, strict=True)):
        raise ValueError("factorial bootstrap case IDs must be unique")
    case_ids = set(maps[0])
    if any(set(mapping) != case_ids for mapping in maps[1:]):
        raise ValueError("factorial bootstrap arms have different case IDs")
    groups: dict[str, tuple[str, ...]] = {}
    temporary: dict[str, list[str]] = {}
    for case_id in sorted(case_ids):
        group_ids = {mapping[case_id].group_id for mapping in maps}
        if len(group_ids) != 1:
            raise ValueError("factorial bootstrap group identity differs across arms")
        temporary.setdefault(next(iter(group_ids)), []).append(case_id)
    groups = {key: tuple(value) for key, value in temporary.items()}
    group_ids = tuple(sorted(groups))

    def sampled_rows(sampled_groups: tuple[str, ...]):
        sampled_cases = tuple(
            case_id for group_id in sampled_groups for case_id in groups[group_id]
        )
        return tuple(tuple(mapping[case_id] for case_id in sampled_cases) for mapping in maps)

    point = statistic(sampled_rows(group_ids))
    randomizer = random.Random(seed)
    values = [
        statistic(sampled_rows(tuple(randomizer.choice(group_ids) for _ in group_ids)))
        for _ in range(resamples)
    ]
    values.sort()
    lower_index = max(0, math.floor(0.025 * (len(values) - 1)))
    upper_index = min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))
    lower = min(point, values[lower_index])
    upper = max(point, values[upper_index])
    return point, lower, upper, len(group_ids)


def grouped_paired_contrast(
    name: str,
    target: tuple[ContextPhoneticCaseMetrics, ...],
    baseline: tuple[ContextPhoneticCaseMetrics, ...],
    *,
    resamples: int,
    seed: str,
) -> GroupedPairedContrast:
    def error_delta(rows) -> float:
        target_rows, baseline_rows = rows
        target_edits = sum(row.effective_edits for row in target_rows)
        baseline_edits = sum(row.effective_edits for row in baseline_rows)
        characters = sum(row.reference_characters for row in target_rows)
        return (target_edits - baseline_edits) / characters

    point, lower, upper, group_count = _grouped_bootstrap(
        (target, baseline),
        statistic=error_delta,
        resamples=resamples,
        seed=seed,
    )
    target_aggregate = aggregate_factorial_arm(target)
    baseline_aggregate = aggregate_factorial_arm(baseline)
    return GroupedPairedContrast(
        name=name,
        target_arm=target[0].arm_name,
        baseline_arm=baseline[0].arm_name,
        mean_character_error_delta=point,
        lower_95=lower,
        upper_95=upper,
        exact_accuracy_delta=(target_aggregate.exact_accuracy - baseline_aggregate.exact_accuracy),
        false_correction_rate_delta=(
            target_aggregate.false_correction_rate - baseline_aggregate.false_correction_rate
        ),
        group_count=group_count,
        resamples=resamples,
        seed=seed,
    )


def grouped_factorial_interaction(
    name: str,
    combined_ordered: tuple[ContextPhoneticCaseMetrics, ...],
    combined_none: tuple[ContextPhoneticCaseMetrics, ...],
    baseline_ordered: tuple[ContextPhoneticCaseMetrics, ...],
    baseline_none: tuple[ContextPhoneticCaseMetrics, ...],
    *,
    resamples: int,
    seed: str,
) -> FactorialInteractionContrast:
    def interaction(rows) -> float:
        co, cn, bo, bn = rows

        def cer(values):
            return sum(row.effective_edits for row in values) / sum(
                row.reference_characters for row in values
            )

        return (cer(co) - cer(cn)) - (cer(bo) - cer(bn))

    point, lower, upper, group_count = _grouped_bootstrap(
        (combined_ordered, combined_none, baseline_ordered, baseline_none),
        statistic=interaction,
        resamples=resamples,
        seed=seed,
    )
    return FactorialInteractionContrast(
        name=name,
        combined_ordered_arm=combined_ordered[0].arm_name,
        combined_none_arm=combined_none[0].arm_name,
        baseline_ordered_arm=baseline_ordered[0].arm_name,
        baseline_none_arm=baseline_none[0].arm_name,
        mean_error_interaction=point,
        lower_95=lower,
        upper_95=upper,
        group_count=group_count,
        resamples=resamples,
        seed=seed,
    )
