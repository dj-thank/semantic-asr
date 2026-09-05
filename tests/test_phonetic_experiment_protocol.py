from __future__ import annotations

from dataclasses import replace

from _phonetic_experiment_fixture import (
    RIGHTS_DIGEST,
    SPLIT_DIGEST,
    case,
    lexicon,
    utility_artifact,
)

from semantic_asr.phonetic_experiment.protocol import (
    FrozenSpanReference,
    PhoneticAblationManifest,
)


def test_reference_may_be_outside_exogenous_lexicon(tmp_path) -> None:
    value = case(
        tmp_path,
        case_id="outside-recovery",
        first_pass=(("また", 0.6, True), ("ただ", 0.4, False)),
        reference="まだ",
        critical=True,
    )
    reduced_lexicon = replace(
        lexicon(),
        entries=tuple(entry for entry in lexicon().entries if entry.text != "まだ"),
    )

    changed = replace(
        value,
        lexicon=reduced_lexicon,
        reference=FrozenSpanReference(reference_id="missing", text="未知"),
    )

    assert changed.reference.text == "未知"
    assert all(entry.text != changed.reference.text for entry in changed.lexicon.entries)


def test_multiple_cases_may_share_speaker_and_are_grouped_statistically(tmp_path) -> None:
    first = case(
        tmp_path,
        case_id="outside-recovery",
        first_pass=(("また", 0.6, True), ("ただ", 0.4, False)),
        reference="まだ",
        critical=True,
    )
    second = replace(
        case(
            tmp_path,
            case_id="correct-retention",
            first_pass=(("まだ", 0.7, True), ("また", 0.3, False)),
            reference="まだ",
            critical=False,
        ),
        speaker_id=first.speaker_id,
    )

    manifest = PhoneticAblationManifest(
        name="shared-speaker",
        revision="r1",
        cases=(first, second),
        runtime_profile_digest="c" * 64,
        utility_artifact_digest=utility_artifact().digest,
        rights_registry_sha256=RIGHTS_DIGEST,
        split_manifest_sha256=SPLIT_DIGEST,
    )

    assert manifest.cases[0].speaker_id == manifest.cases[1].speaker_id
