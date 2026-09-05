from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner
from semantic_asr.phonetic_experiment.promotion import PhoneticPromotionPolicy
from semantic_asr.phonetic_experiment.registration import (
    PhoneticExperimentRegistration,
    run_registered_phonetic_experiment,
)

from _phonetic_experiment_fixture import manifest, protocol, utility_artifact


def permissive_policy() -> PhoneticPromotionPolicy:
    return PhoneticPromotionPolicy(
        target_arm="phone+mora",
        baseline_arm="first-pass",
        minimum_exact_accuracy_gain=-1.0,
        maximum_bootstrap_upper_error_delta=1.0,
        minimum_oracle_coverage=0.0,
        minimum_outside_first_pass_recovery_rate=0.0,
        maximum_false_correction_rate=1.0,
        maximum_introduced_error_characters=100,
        minimum_critical_exact_accuracy=0.0,
        minimum_accepted_coverage=0.0,
        maximum_mean_generation_latency_ms=10_000.0,
    )


def test_registered_experiment_runs_only_exact_frozen_components(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    plan = protocol()
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())
    promotion = permissive_policy()
    registration = PhoneticExperimentRegistration.create(
        name="fixture",
        revision="r1",
        manifest=experiment,
        protocol=plan,
        planner=planner,
        promotion_policy=promotion,
    )

    result = run_registered_phonetic_experiment(
        registration,
        manifest=experiment,
        protocol=plan,
        planner=planner,
        promotion_policy=promotion,
    )

    assert result.registration_digest == registration.digest
    assert result.promotion.report_digest == result.report.digest


def test_changed_thresholds_fail_registration_before_planning(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    plan = protocol()
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())
    promotion = permissive_policy()
    registration = PhoneticExperimentRegistration.create(
        name="fixture",
        revision="r1",
        manifest=experiment,
        protocol=plan,
        planner=planner,
        promotion_policy=promotion,
    )
    changed = replace(promotion, maximum_false_correction_rate=0.0)

    with pytest.raises(ValueError, match="promotion_policy_digest"):
        run_registered_phonetic_experiment(
            registration,
            manifest=experiment,
            protocol=plan,
            planner=planner,
            promotion_policy=changed,
        )
    assert runtime.calls == []
