from __future__ import annotations

import pytest
from _document_experiment_fixture import AUDIO, RIGHTS, SPLIT, first_pass

from semantic_asr.document_experiment.protocol import (
    CriticalReferenceToken,
    DocumentExperimentArm,
    DocumentExperimentCase,
    DocumentExperimentManifest,
    DocumentExperimentProtocol,
    FrozenExternalContext,
    FrozenReference,
)
from semantic_asr.multilevel_lattice import DocumentContext


def reference() -> FrozenReference:
    windows = (
        "レビュー完了まではまだマージしません。",
        "承認後に統合します。",
    )
    return FrozenReference(
        reference_id="reference-1",
        source_audio_sha256=AUDIO,
        text="".join(windows),
        window_texts=windows,
        critical_tokens=(
            CriticalReferenceToken(kind="negation", text="ません", count=1),
            CriticalReferenceToken(kind="entity", text="マージ", count=1),
        ),
    )


def case(*, contexts=()) -> DocumentExperimentCase:
    return DocumentExperimentCase(
        case_id="case-1",
        first_pass=first_pass(),
        reference=reference(),
        rights_decision="allow",
        license_id="fixture-license",
        source_id="source-1",
        speaker_id="speaker-test-1",
        session_id="session-test-1",
        dataset_revision="fixture-r1",
        split_manifest_sha256=SPLIT,
        external_contexts=contexts,
    )


def test_reference_is_not_part_of_planning_digest() -> None:
    row = case()
    changed_reference = FrozenReference(
        reference_id=row.reference.reference_id,
        source_audio_sha256=AUDIO,
        text="別の参照です。承認後に統合します。",
        window_texts=("別の参照です。", "承認後に統合します。"),
    )
    changed = DocumentExperimentCase(
        case_id=row.case_id,
        first_pass=row.first_pass,
        reference=changed_reference,
        rights_decision=row.rights_decision,
        license_id=row.license_id,
        source_id=row.source_id,
        speaker_id=row.speaker_id,
        session_id=row.session_id,
        dataset_revision=row.dataset_revision,
        split_manifest_sha256=row.split_manifest_sha256,
    )

    assert row.planning_digest == changed.planning_digest
    assert row.digest != changed.digest


def test_reference_bearing_case_fails_closed_without_allow_rights() -> None:
    with pytest.raises(ValueError, match="rights_decision='allow'"):
        DocumentExperimentCase(
            case_id="case-1",
            first_pass=first_pass(),
            reference=reference(),
            rights_decision="review",
            license_id="fixture-license",
            source_id="source-1",
            speaker_id="speaker-test-1",
            session_id="session-test-1",
            dataset_revision="fixture-r1",
            split_manifest_sha256=SPLIT,
        )


def test_complete_reference_in_external_context_is_rejected() -> None:
    leaked = FrozenExternalContext(
        name="leaked",
        context=DocumentContext(left_context=reference().text),
        provenance_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="complete evaluation reference"):
        case(contexts=(leaked,))


def test_manifest_rejects_speaker_or_session_leakage() -> None:
    row = case()

    with pytest.raises(ValueError, match="test speakers overlap"):
        DocumentExperimentManifest(
            name="fixture",
            revision="r1",
            cases=(row,),
            rights_registry_sha256=RIGHTS,
            split_manifest_sha256=SPLIT,
            training_speaker_ids=(row.speaker_id,),
        )

    with pytest.raises(ValueError, match="test sessions overlap"):
        DocumentExperimentManifest(
            name="fixture",
            revision="r1",
            cases=(row,),
            rights_registry_sha256=RIGHTS,
            split_manifest_sha256=SPLIT,
            calibration_session_ids=(row.session_id,),
        )


def test_protocol_requires_an_acoustic_only_baseline() -> None:
    baseline = DocumentExperimentArm(
        name="acoustic",
        candidate_view="acoustic-only",
        direction="none",
        is_control=True,
    )
    ordered = DocumentExperimentArm(
        name="ordered",
        candidate_view="ordered-document",
        direction="bidirectional",
        scorer_key="ngram",
    )
    protocol = DocumentExperimentProtocol(
        name="fixture-protocol",
        revision="r1",
        arms=(baseline, ordered),
        baseline_arm="acoustic",
        bootstrap_resamples=200,
    )

    assert protocol.digest

    with pytest.raises(ValueError, match="baseline arm must be acoustic-only"):
        DocumentExperimentProtocol(
            name="invalid",
            revision="r1",
            arms=(baseline, ordered),
            baseline_arm="ordered",
            bootstrap_resamples=200,
        )
