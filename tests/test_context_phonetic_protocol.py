from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_asr.context_phonetic_experiment.planner import (
    deterministic_context_derangement,
)
from semantic_asr.context_phonetic_experiment.protocol import ContextPhoneticManifest

from _context_phonetic_factorial_fixture import factorial_manifest, factorial_protocol


def test_shuffle_is_deterministic_and_avoids_registered_identity_leakage(tmp_path) -> None:
    manifest, _runtime, _scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()

    first = deterministic_context_derangement(manifest, protocol)
    second = deterministic_context_derangement(manifest, protocol)

    assert {key: value.case_id for key, value in first.items()} == {
        key: value.case_id for key, value in second.items()
    }
    by_id = {case.case_id: case for case in manifest.cases}
    for case_id, donor in first.items():
        receiver = by_id[case_id]
        assert donor.case_id != receiver.case_id
        assert donor.phonetic_case.speaker_id != receiver.phonetic_case.speaker_id
        assert donor.phonetic_case.session_id != receiver.phonetic_case.session_id
        assert donor.phonetic_case.source_id != receiver.phonetic_case.source_id
        assert donor.context_group_id != receiver.context_group_id


def test_impossible_shuffle_constraints_fail_closed(tmp_path) -> None:
    manifest, _runtime, _scorer = factorial_manifest(tmp_path)
    first = manifest.cases[0]
    second = replace(
        manifest.cases[1],
        phonetic_case=replace(
            manifest.cases[1].phonetic_case,
            speaker_id=first.phonetic_case.speaker_id,
            session_id=first.phonetic_case.session_id,
            source_id=first.phonetic_case.source_id,
        ),
    )
    tiny = ContextPhoneticManifest(
        name="impossible",
        revision="r1",
        cases=(first, second),
        phonetic_manifest_digest=manifest.phonetic_manifest_digest,
        context_source_digest=manifest.context_source_digest,
        rights_registry_sha256=manifest.rights_registry_sha256,
    )

    with pytest.raises(ValueError, match="no deterministic context derangement"):
        deterministic_context_derangement(tiny, factorial_protocol())


def test_protocol_requires_factorial_counterparts(tmp_path) -> None:
    manifest, _runtime, _scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    assert manifest.planning_digest != manifest.digest
    assert protocol.arm("phone+mora:ordered").context_condition == "ordered"
    assert protocol.arm("phone+mora:shuffled").phonetic_arm_name == "phone+mora"
