from __future__ import annotations

import pytest
from _document_experiment_fixture import AUDIO, RIGHTS, SPLIT, fake_plan, first_pass

from semantic_asr.document_experiment.ngram_scorer import DocumentLanguageScore
from semantic_asr.document_experiment.protocol import (
    DocumentExperimentArm,
    DocumentExperimentCase,
    DocumentExperimentManifest,
    DocumentExperimentProtocol,
    FrozenReference,
)
from semantic_asr.document_experiment.registration import (
    DocumentExperimentRegistration,
    FrozenScorerRegistry,
    run_registered_document_context_experiment,
)
from semantic_asr.document_experiment.runner import prepare_document_experiment


class RegisteredScorer:
    def __init__(self, profile_digest: str = "d" * 64) -> None:
        self.profile_digest = profile_digest

    def score_path(
        self,
        path,
        arm,
        *,
        case_id,
        left_context="",
        right_context="",
        maximum_characters,
    ):
        del case_id, left_context, right_context, maximum_characters
        value = 0.5 if "まだ" in path.text else -0.5
        return DocumentLanguageScore(
            value=value,
            raw_average_log_likelihood=value,
            forward_average_log_likelihood=value,
            backward_average_log_likelihood=value,
            source="registered-fixture",
            profile_digest=self.profile_digest,
            path_digest=path.digest,
            arm_digest=arm.digest,
            scored_characters=len(path.text) * 2,
            scorer_calls=2,
        )


def inputs():
    windows = (
        "レビュー完了まではまだマージしません。",
        "承認後に統合します。",
    )
    case = DocumentExperimentCase(
        case_id="case",
        first_pass=first_pass(),
        reference=FrozenReference(
            reference_id="reference",
            source_audio_sha256=AUDIO,
            text="".join(windows),
            window_texts=windows,
        ),
        rights_decision="allow",
        license_id="fixture-license",
        source_id="source",
        speaker_id="speaker",
        session_id="session",
        dataset_revision="r1",
        split_manifest_sha256=SPLIT,
    )
    manifest = DocumentExperimentManifest(
        name="fixture",
        revision="r1",
        cases=(case,),
        rights_registry_sha256=RIGHTS,
        split_manifest_sha256=SPLIT,
    )
    protocol = DocumentExperimentProtocol(
        name="fixture",
        revision="r1",
        arms=(
            DocumentExperimentArm(
                name="base",
                candidate_view="acoustic-only",
                direction="none",
            ),
            DocumentExperimentArm(
                name="context",
                candidate_view="ordered-document",
                direction="bidirectional",
                scorer_key="model",
            ),
        ),
        baseline_arm="base",
        maximum_candidate_documents=3,
        maximum_scored_characters=10_000,
        bootstrap_resamples=200,
    )
    return manifest, protocol


def test_registered_run_binds_protocol_manifest_and_model_profile() -> None:
    manifest, protocol = inputs()
    scorer = RegisteredScorer()
    scorers = {"model": scorer}
    registry = FrozenScorerRegistry.from_scorers(scorers, revision="registry-r1")
    registration = DocumentExperimentRegistration.create(
        protocol,
        manifest,
        registry,
        registration_id="registration-1",
    )
    prepared = prepare_document_experiment(manifest, protocol, lambda view: fake_plan())

    result = run_registered_document_context_experiment(
        prepared,
        manifest,
        protocol,
        registry=registry,
        registration=registration,
        scorers=scorers,
    )

    assert result.registration.digest == registration.digest
    assert result.scorer_registry.digest == registry.digest
    assert result.digest
    assert "selectedText" not in str(result.as_dict(include_text=False))


def test_registry_rejects_runtime_model_replacement() -> None:
    manifest, protocol = inputs()
    registered = {"model": RegisteredScorer("d" * 64)}
    runtime = {"model": RegisteredScorer("e" * 64)}
    registry = FrozenScorerRegistry.from_scorers(registered, revision="registry-r1")

    with pytest.raises(ValueError, match="profile changed"):
        registry.validate(protocol, runtime)


def test_registration_rejects_protocol_change_after_freeze() -> None:
    manifest, protocol = inputs()
    scorers = {"model": RegisteredScorer()}
    registry = FrozenScorerRegistry.from_scorers(scorers, revision="registry-r1")
    registration = DocumentExperimentRegistration.create(
        protocol,
        manifest,
        registry,
        registration_id="registration-1",
    )
    changed_protocol = DocumentExperimentProtocol(
        name=protocol.name,
        revision="r2",
        arms=protocol.arms,
        baseline_arm=protocol.baseline_arm,
        maximum_candidate_documents=protocol.maximum_candidate_documents,
        maximum_scored_characters=protocol.maximum_scored_characters,
        bootstrap_resamples=protocol.bootstrap_resamples,
    )
    prepared = prepare_document_experiment(manifest, protocol, lambda view: fake_plan())

    with pytest.raises(ValueError, match="runtime protocol differs"):
        run_registered_document_context_experiment(
            prepared,
            manifest,
            changed_protocol,
            registry=registry,
            registration=registration,
            scorers=scorers,
        )
