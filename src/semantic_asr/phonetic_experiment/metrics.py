"""Reference-opened metrics for frozen phonetic candidate/arm decisions."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .planner import FrozenPhoneticCandidatePool
from .protocol import FrozenSpanReference
from .selection import PhoneticAblationDecision


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class PhoneticCaseArmMetrics:
    case_id: str
    group_id: str
    arm_name: str
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
    generation_latency_ms: float
    selection_latency_ms: float

    def __post_init__(self) -> None:
        if not self.case_id or not self.group_id or not self.arm_name:
            raise ValueError("case arm metrics require case, group, and arm names")
        for digest in (
            self.reference_text_sha256,
            self.first_pass_text_sha256,
            self.proposed_text_sha256,
            self.effective_text_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("case arm text digests must be SHA-256 values")
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
        for name in ("margin", "generation_latency_ms", "selection_latency_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def effective_error_rate(self) -> float:
        return self.effective_edits / self.reference_characters

    @property
    def first_pass_error_rate(self) -> float:
        return self.first_pass_edits / self.reference_characters

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticArmAggregate:
    arm_name: str
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
    mean_generation_latency_ms: float
    mean_selection_latency_ms: float

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
            }
        )


def evaluate_case_arm(
    pool: FrozenPhoneticCandidatePool,
    decision: PhoneticAblationDecision,
    reference: FrozenSpanReference,
    *,
    group_id: str,
) -> PhoneticCaseArmMetrics:
    if decision.pool_digest != pool.digest:
        raise ValueError("decision belongs to a different frozen candidate pool")
    proposed = pool.candidate(decision.proposed_candidate_id)
    effective = pool.candidate(decision.effective_candidate_id)
    first_pass = pool.candidate(pool.first_pass_selected_candidate_id)
    first_pass_surfaces = {
        candidate.text for candidate in pool.candidates if candidate.is_first_pass
    }
    pool_surfaces = {candidate.text for candidate in pool.candidates}
    first_edits = edit_distance(first_pass.text, reference.text)
    proposed_edits = edit_distance(proposed.text, reference.text)
    effective_edits = edit_distance(effective.text, reference.text)
    return PhoneticCaseArmMetrics(
        case_id=pool.case_id,
        group_id=group_id,
        arm_name=decision.arm_name,
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
        generation_latency_ms=pool.generation_latency_ms,
        selection_latency_ms=decision.selection_latency_ms,
    )


def aggregate_arm(rows: tuple[PhoneticCaseArmMetrics, ...]) -> PhoneticArmAggregate:
    if not rows:
        raise ValueError("cannot aggregate an empty phonetic arm")
    arm_names = {row.arm_name for row in rows}
    if len(arm_names) != 1:
        raise ValueError("phonetic aggregate cannot mix arm names")
    return PhoneticArmAggregate(
        arm_name=next(iter(arm_names)),
        case_count=len(rows),
        exact_count=sum(row.effective_exact for row in rows),
        proposed_exact_count=sum(row.proposed_exact for row in rows),
        first_pass_exact_count=sum(row.first_pass_edits == 0 for row in rows),
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
        mean_generation_latency_ms=sum(row.generation_latency_ms for row in rows) / len(rows),
        mean_selection_latency_ms=sum(row.selection_latency_ms for row in rows) / len(rows),
    )


@dataclass(frozen=True, slots=True)
class PairedErrorDelta:
    target_arm: str
    baseline_arm: str
    mean_character_error_delta: float
    lower_95: float
    upper_95: float
    resamples: int
    seed: str
    group_count: int

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def paired_bootstrap_error_delta(
    target: tuple[PhoneticCaseArmMetrics, ...],
    baseline: tuple[PhoneticCaseArmMetrics, ...],
    *,
    resamples: int,
    seed: str,
) -> PairedErrorDelta:
    if not target or len(target) != len(baseline):
        raise ValueError("paired bootstrap requires equal non-empty arms")
    target_by_case = {row.case_id: row for row in target}
    baseline_by_case = {row.case_id: row for row in baseline}
    if len(target_by_case) != len(target) or len(baseline_by_case) != len(baseline):
        raise ValueError("paired bootstrap case IDs must be unique")
    if set(target_by_case) != set(baseline_by_case):
        raise ValueError("paired bootstrap case IDs differ")
    if resamples < 1 or not seed:
        raise ValueError("paired bootstrap requires resamples and seed")
    groups: dict[str, list[str]] = {}
    for case_id in sorted(target_by_case):
        target_row = target_by_case[case_id]
        baseline_row = baseline_by_case[case_id]
        if target_row.group_id != baseline_row.group_id:
            raise ValueError("paired bootstrap group identities differ between arms")
        groups.setdefault(target_row.group_id, []).append(case_id)
    group_ids = tuple(sorted(groups))

    def delta(sampled_groups: tuple[str, ...]) -> float:
        sampled_cases = tuple(
            case_id for group_id in sampled_groups for case_id in groups[group_id]
        )
        target_edits = sum(target_by_case[case_id].effective_edits for case_id in sampled_cases)
        baseline_edits = sum(baseline_by_case[case_id].effective_edits for case_id in sampled_cases)
        characters = sum(target_by_case[case_id].reference_characters for case_id in sampled_cases)
        return (target_edits - baseline_edits) / characters

    point = delta(group_ids)
    randomizer = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = tuple(randomizer.choice(group_ids) for _ in group_ids)
        values.append(delta(sample))
    values.sort()
    lower_index = max(0, math.floor(0.025 * (len(values) - 1)))
    upper_index = min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))
    return PairedErrorDelta(
        target_arm=target[0].arm_name,
        baseline_arm=baseline[0].arm_name,
        mean_character_error_delta=point,
        lower_95=values[lower_index],
        upper_95=values[upper_index],
        resamples=resamples,
        seed=seed,
        group_count=len(group_ids),
    )
