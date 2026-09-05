from __future__ import annotations

from dataclasses import replace

from semantic_asr.context_phonetic_experiment.promotion import (
    ContextPhoneticPromotionPolicy,
    evaluate_context_phonetic_promotion,
)
from semantic_asr.context_phonetic_experiment.runner import (
    run_context_phonetic_experiment,
)
from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner

from _context_phonetic_factorial_fixture import (
    factorial_manifest,
    factorial_protocol,
    utility_artifact,
)


def report(tmp_path):
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    return run_context_phonetic_experiment(
        manifest,
        factorial_protocol(),
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
        scorer,
    )


def passing_policy() -> ContextPhoneticPromotionPolicy:
    return ContextPhoneticPromotionPolicy(
        target_arm="phone+mora:ordered",
        baseline_arm="first-pass:none",
        shuffled_control_arm="phone+mora:shuffled",
        minimum_exact_accuracy_gain=0.5,
        maximum_combined_vs_baseline_upper_error_delta=0.0,
        maximum_ordered_vs_shuffled_upper_error_delta=0.0,
        maximum_interaction_upper_error=0.0,
        minimum_oracle_coverage=1.0,
        minimum_outside_first_pass_recovery_rate=1.0,
        maximum_false_correction_rate=0.0,
        maximum_false_correction_rate_delta=0.0,
        maximum_introduced_error_characters=0,
        minimum_critical_exact_accuracy=1.0,
        minimum_accepted_coverage=1.0,
        maximum_total_runtime_ms=10_000.0,
    )


def test_factorial_promotion_passes_only_when_every_control_passes(tmp_path) -> None:
    decision = evaluate_context_phonetic_promotion(report(tmp_path), passing_policy())

    assert decision.promote
    assert decision.reasons == ("all-conjunctive-promotion-checks-passed",)
    assert all(check.passed for check in decision.checks)


def test_factorial_promotion_fails_on_any_single_stricter_threshold(tmp_path) -> None:
    strict = replace(passing_policy(), minimum_exact_accuracy_gain=0.75)

    decision = evaluate_context_phonetic_promotion(report(tmp_path), strict)

    assert not decision.promote
    assert "exact-accuracy-gain" in decision.reasons
