from __future__ import annotations

from _phonetic_experiment_fixture import manifest, protocol, utility_artifact

from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner
from semantic_asr.phonetic_experiment.promotion import (
    PhoneticPromotionPolicy,
    evaluate_phonetic_promotion,
)
from semantic_asr.phonetic_experiment.runner import run_phonetic_ablation


def report(tmp_path):
    experiment, runtime = manifest(tmp_path)
    return run_phonetic_ablation(
        experiment,
        protocol(),
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
    )


def test_promotion_fails_when_false_corrections_exceed_policy(tmp_path) -> None:
    decision = evaluate_phonetic_promotion(
        report(tmp_path),
        PhoneticPromotionPolicy(
            target_arm="phone+mora",
            baseline_arm="first-pass",
            minimum_exact_accuracy_gain=-1.0,
            maximum_bootstrap_upper_error_delta=1.0,
            minimum_oracle_coverage=0.0,
            minimum_outside_first_pass_recovery_rate=0.0,
            maximum_false_correction_rate=0.0,
            maximum_introduced_error_characters=10,
            minimum_critical_exact_accuracy=0.0,
            minimum_accepted_coverage=0.0,
            maximum_mean_generation_latency_ms=10_000.0,
        ),
    )

    assert not decision.promote
    assert "false-correction-rate" in decision.reasons


def test_promotion_passes_only_when_every_threshold_is_relaxed(tmp_path) -> None:
    decision = evaluate_phonetic_promotion(
        report(tmp_path),
        PhoneticPromotionPolicy(
            target_arm="phone+mora",
            baseline_arm="first-pass",
            minimum_exact_accuracy_gain=-1.0,
            maximum_bootstrap_upper_error_delta=1.0,
            minimum_oracle_coverage=0.0,
            minimum_outside_first_pass_recovery_rate=0.0,
            maximum_false_correction_rate=1.0,
            maximum_introduced_error_characters=10,
            minimum_critical_exact_accuracy=0.0,
            minimum_accepted_coverage=0.0,
            maximum_mean_generation_latency_ms=10_000.0,
        ),
    )

    assert decision.promote
    assert decision.reasons == ("all-conjunctive-promotion-checks-passed",)
