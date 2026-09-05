from __future__ import annotations

from semantic_asr.phonetic_experiment.metrics import (
    PhoneticCaseArmMetrics,
    aggregate_arm,
    paired_bootstrap_error_delta,
)


def metric(
    case_id: str,
    group_id: str,
    arm: str,
    *,
    first_pass_edits: int,
    effective_edits: int,
    false_correction: bool = False,
) -> PhoneticCaseArmMetrics:
    return PhoneticCaseArmMetrics(
        case_id=case_id,
        group_id=group_id,
        arm_name=arm,
        reference_text_sha256="a" * 64,
        first_pass_text_sha256="b" * 64,
        proposed_text_sha256="c" * 64,
        effective_text_sha256="d" * 64,
        reference_characters=10,
        first_pass_edits=first_pass_edits,
        proposed_edits=effective_edits,
        effective_edits=effective_edits,
        pool_oracle=True,
        reference_outside_first_pass=False,
        proposed_exact=effective_edits == 0,
        effective_exact=effective_edits == 0,
        recovered_outside_first_pass=False,
        false_correction=false_correction,
        corrected_first_pass=first_pass_edits > 0 and effective_edits == 0,
        introduced_error_characters=max(0, effective_edits - first_pass_edits),
        corrected_error_characters=max(0, first_pass_edits - effective_edits),
        accepted=True,
        changed_proposal=first_pass_edits != effective_edits,
        changed_effective=first_pass_edits != effective_edits,
        critical=False,
        margin=0.5,
        generation_latency_ms=1.0,
        selection_latency_ms=0.1,
    )


def test_false_correction_rate_is_conditioned_on_correct_first_pass_cases() -> None:
    rows = (
        metric("a", "speaker-a", "target", first_pass_edits=0, effective_edits=1, false_correction=True),
        metric("b", "speaker-b", "target", first_pass_edits=0, effective_edits=0),
        metric("c", "speaker-c", "target", first_pass_edits=2, effective_edits=0),
    )

    aggregate = aggregate_arm(rows)

    assert aggregate.first_pass_exact_count == 2
    assert aggregate.false_correction_count == 1
    assert aggregate.false_correction_rate == 0.5


def test_paired_bootstrap_resamples_speaker_groups_not_individual_rows() -> None:
    baseline = (
        metric("a1", "speaker-a", "baseline", first_pass_edits=1, effective_edits=1),
        metric("a2", "speaker-a", "baseline", first_pass_edits=1, effective_edits=1),
        metric("b1", "speaker-b", "baseline", first_pass_edits=1, effective_edits=1),
    )
    target = (
        metric("a1", "speaker-a", "target", first_pass_edits=1, effective_edits=0),
        metric("a2", "speaker-a", "target", first_pass_edits=1, effective_edits=0),
        metric("b1", "speaker-b", "target", first_pass_edits=1, effective_edits=2),
    )

    delta = paired_bootstrap_error_delta(
        target,
        baseline,
        resamples=200,
        seed="grouped-bootstrap",
    )

    assert delta.group_count == 2
    assert delta.resamples == 200
    assert delta.lower_95 <= delta.mean_character_error_delta <= delta.upper_95
