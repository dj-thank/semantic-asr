from __future__ import annotations

from dataclasses import replace

import pytest
from _context_phonetic_factorial_fixture import (
    factorial_manifest,
    factorial_protocol,
    utility_artifact,
)

from semantic_asr.context_phonetic_experiment.promotion import (
    ContextPhoneticPromotionPolicy,
)
from semantic_asr.context_phonetic_experiment.registration import (
    ContextPhoneticExperimentRegistration,
    run_registered_context_phonetic_experiment,
)
from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner


def policy() -> ContextPhoneticPromotionPolicy:
    return ContextPhoneticPromotionPolicy(
        target_arm="phone+mora:ordered",
        baseline_arm="first-pass:none",
        shuffled_control_arm="phone+mora:shuffled",
        minimum_exact_accuracy_gain=-1.0,
        maximum_combined_vs_baseline_upper_error_delta=1.0,
        maximum_ordered_vs_shuffled_upper_error_delta=1.0,
        maximum_interaction_upper_error=1.0,
        minimum_oracle_coverage=0.0,
        minimum_outside_first_pass_recovery_rate=0.0,
        maximum_false_correction_rate=1.0,
        maximum_false_correction_rate_delta=1.0,
        maximum_introduced_error_characters=100,
        minimum_critical_exact_accuracy=0.0,
        minimum_accepted_coverage=0.0,
        maximum_total_runtime_ms=10_000.0,
    )


def components(tmp_path):
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())
    promotion = policy()
    registration = ContextPhoneticExperimentRegistration.create(
        name="factorial-fixture",
        revision="r1",
        manifest=manifest,
        protocol=protocol,
        phonetic_planner=planner,
        context_scorer=scorer,
        promotion_policy=promotion,
    )
    return manifest, runtime, scorer, protocol, planner, promotion, registration


def test_registered_factorial_experiment_runs_exact_frozen_components(tmp_path) -> None:
    manifest, _runtime, scorer, protocol, planner, promotion, registration = components(tmp_path)

    result = run_registered_context_phonetic_experiment(
        registration,
        manifest=manifest,
        protocol=protocol,
        phonetic_planner=planner,
        context_scorer=scorer,
        promotion_policy=promotion,
    )

    assert result.registration_digest == registration.digest
    assert result.promotion.report_digest == result.report.digest


def test_changed_reference_is_rejected_before_audio_or_context_scoring(tmp_path) -> None:
    manifest, runtime, scorer, protocol, planner, promotion, registration = components(tmp_path)
    first = manifest.cases[0]
    changed_manifest = replace(
        manifest,
        cases=(
            replace(
                first,
                phonetic_case=replace(
                    first.phonetic_case,
                    reference=replace(first.phonetic_case.reference, text="ただ"),
                ),
            ),
            *manifest.cases[1:],
        ),
    )

    with pytest.raises(ValueError, match="manifest_digest"):
        run_registered_context_phonetic_experiment(
            registration,
            manifest=changed_manifest,
            protocol=protocol,
            phonetic_planner=planner,
            context_scorer=scorer,
            promotion_policy=promotion,
        )
    assert runtime.calls == []
    assert scorer.calls == []


def test_changed_threshold_or_scorer_identity_is_rejected_before_planning(tmp_path) -> None:
    manifest, runtime, scorer, protocol, planner, promotion, registration = components(tmp_path)
    changed_policy = replace(promotion, maximum_false_correction_rate=0.0)

    with pytest.raises(ValueError, match="promotion_policy_digest"):
        run_registered_context_phonetic_experiment(
            registration,
            manifest=manifest,
            protocol=protocol,
            phonetic_planner=planner,
            context_scorer=scorer,
            promotion_policy=changed_policy,
        )
    assert runtime.calls == []
    assert scorer.calls == []

    scorer.profile_digest = "9" * 64
    with pytest.raises(ValueError, match="context_scorer_profile_digest"):
        run_registered_context_phonetic_experiment(
            registration,
            manifest=manifest,
            protocol=protocol,
            phonetic_planner=planner,
            context_scorer=scorer,
            promotion_policy=promotion,
        )
    assert runtime.calls == []
