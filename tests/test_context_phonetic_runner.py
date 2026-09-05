from __future__ import annotations

import json

from semantic_asr.context_phonetic_experiment.planner import (
    prepare_context_phonetic_experiment,
)
from semantic_asr.context_phonetic_experiment.runner import (
    evaluate_prepared_context_phonetic_experiment,
    run_context_phonetic_experiment,
)
from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner

from _context_phonetic_factorial_fixture import (
    factorial_manifest,
    factorial_protocol,
    utility_artifact,
)


def run(tmp_path):
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    report = run_context_phonetic_experiment(
        manifest,
        protocol,
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
        scorer,
    )
    return manifest, protocol, runtime, scorer, report


def test_factorial_report_contains_main_effects_specificity_and_interaction(tmp_path) -> None:
    _manifest, _protocol, _runtime, _scorer, report = run(tmp_path)
    contrasts = {row.name: row for row in report.contrasts}
    aggregates = {row.arm_name: row for row in report.aggregates}

    assert set(contrasts) == {
        "combined-vs-baseline",
        "ordered-vs-shuffled",
        "phonetic-main-effect",
        "context-main-effect",
    }
    assert report.interaction.name == "context-by-phonetic-error-interaction"
    assert aggregates["phone+mora:ordered"].exact_count == 4
    assert aggregates["first-pass:none"].exact_count == 2
    assert aggregates["phone+mora:none"].false_correction_count == 1
    assert aggregates["phone+mora:ordered"].false_correction_count == 0
    assert aggregates["phone+mora:ordered"].outside_first_pass_recovery_count == 2
    assert aggregates["phone+mora:shuffled"].exact_accuracy < aggregates[
        "phone+mora:ordered"
    ].exact_accuracy


def test_every_arm_uses_the_same_prepared_pool_and_context_scores(tmp_path) -> None:
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    prepared = prepare_context_phonetic_experiment(
        manifest,
        protocol,
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
        scorer,
    )

    report = evaluate_prepared_context_phonetic_experiment(
        manifest,
        protocol,
        prepared,
    )

    for case_result in report.case_results:
        assert {
            decision.prepared_case_digest for decision in case_result.decisions
        } == {case_result.prepared_case_digest}
        assert len(case_result.decisions) == len(protocol.arms)


def test_report_omits_raw_candidate_reference_and_context_text(tmp_path) -> None:
    _manifest, _protocol, _runtime, _scorer, report = run(tmp_path)
    destination = report.write(tmp_path / "factorial-report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False)

    for raw in (
        "まだ",
        "また",
        "ただ",
        "たま",
        "left context",
        "right context",
        "prefer:",
    ):
        assert raw not in serialized
    assert payload["rawCandidateTextIncluded"] is False
    assert payload["rawReferenceTextIncluded"] is False
    assert payload["rawContextIncluded"] is False


def test_changed_reference_cannot_reuse_a_registered_prepared_result(tmp_path) -> None:
    from dataclasses import replace

    manifest, runtime, scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    prepared = prepare_context_phonetic_experiment(
        manifest,
        protocol,
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
        scorer,
    )
    first = manifest.cases[0]
    changed = replace(
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

    # Planning evidence is intentionally reusable, but the final report is bound to the changed
    # full manifest digest. Preregistered execution (tested separately) rejects the mutation.
    report = evaluate_prepared_context_phonetic_experiment(changed, protocol, prepared)
    assert report.manifest_digest == changed.digest
    assert report.manifest_digest != manifest.digest
