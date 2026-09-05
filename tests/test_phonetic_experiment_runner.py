from __future__ import annotations

import json
from dataclasses import replace

import pytest
from _phonetic_experiment_fixture import manifest, protocol, utility_artifact

from semantic_asr.phonetic_experiment.planner import (
    FrozenPhoneticCandidatePlanner,
    PlanningCaseView,
)
from semantic_asr.phonetic_experiment.runner import (
    evaluate_prepared_phonetic_ablation,
    prepare_phonetic_ablation,
    run_phonetic_ablation,
)


def test_planning_view_contains_no_reference(tmp_path) -> None:
    experiment, _runtime = manifest(tmp_path)
    view = PlanningCaseView.from_case(experiment.cases[0])

    assert not hasattr(view, "reference")


def test_all_arms_reuse_one_frozen_candidate_pool_per_case(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    plan = protocol()
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())

    report = run_phonetic_ablation(experiment, plan, planner)

    assert len(runtime.calls) == len(experiment.cases)
    assert len(report.case_results) == len(experiment.cases)
    for result in report.case_results:
        assert {decision.pool_digest for decision in result.decisions} == {result.pool_digest}
        assert {decision.arm_name for decision in result.decisions} == {
            arm.name for arm in plan.arms
        }


def test_reference_change_after_preparation_invalidates_manifest_binding(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    plan = protocol()
    prepared = prepare_phonetic_ablation(
        experiment,
        plan,
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
    )
    case = experiment.cases[0]
    changed_case = replace(
        case,
        reference=replace(case.reference, text="ただ"),
    )
    changed_manifest = experiment.__class__(
        name=experiment.name,
        revision=experiment.revision,
        cases=(changed_case, *experiment.cases[1:]),
        runtime_profile_digest=experiment.runtime_profile_digest,
        utility_artifact_digest=experiment.utility_artifact_digest,
        rights_registry_sha256=experiment.rights_registry_sha256,
        split_manifest_sha256=experiment.split_manifest_sha256,
    )

    with pytest.raises(ValueError, match="different manifest"):
        evaluate_prepared_phonetic_ablation(changed_manifest, plan, prepared)


def test_report_is_text_hash_only_by_default(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    report = run_phonetic_ablation(
        experiment,
        protocol(),
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
    )
    destination = report.write(tmp_path / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "まだ" not in serialized
    assert "また" not in serialized
    assert payload["rawReferenceTextIncluded"] is False
    assert payload["rawCandidateTextIncluded"] is False


def test_outside_recovery_and_false_correction_are_both_visible(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    report = run_phonetic_ablation(
        experiment,
        protocol(),
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
    )
    aggregates = {row.arm_name: row for row in report.aggregates}

    assert aggregates["phone+mora"].outside_first_pass_recovery_count == 1
    assert aggregates["phone+mora"].false_correction_count == 1
    assert aggregates["first-pass"].false_correction_count == 0
