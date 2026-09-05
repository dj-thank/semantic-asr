from __future__ import annotations

from _document_experiment_fixture import AUDIO, RIGHTS, SPLIT, fake_plan, first_pass

from semantic_asr.document_experiment.ngram_scorer import DocumentLanguageScore
from semantic_asr.document_experiment.promotion import (
    DocumentContextPromotionPolicy,
    evaluate_document_context_promotion,
)
from semantic_asr.document_experiment.protocol import (
    CriticalReferenceToken,
    DocumentExperimentArm,
    DocumentExperimentCase,
    DocumentExperimentManifest,
    DocumentExperimentProtocol,
    FrozenReference,
)
from semantic_asr.document_experiment.runner import (
    prepare_document_experiment,
    run_document_context_experiment,
)


class ArmAwareScorer:
    profile_digest = "d" * 64

    def __init__(self, *, shuffled_also_prefers_correction: bool) -> None:
        self.shuffled_also_prefers_correction = shuffled_also_prefers_correction

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
        corrected = "まだ" in path.text
        if arm.candidate_view == "shuffled-document" and not self.shuffled_also_prefers_correction:
            value = 0.8 if "また" in path.text else -0.8
        else:
            value = 0.8 if corrected else -0.8
        return DocumentLanguageScore(
            value=value,
            raw_average_log_likelihood=value,
            forward_average_log_likelihood=value,
            backward_average_log_likelihood=value,
            source="arm-aware-fixture",
            profile_digest=self.profile_digest,
            path_digest=path.digest,
            arm_digest=arm.digest,
            scored_characters=len(path.text) * 2,
            scorer_calls=2,
        )


def fixture_manifest() -> DocumentExperimentManifest:
    windows = (
        "レビュー完了まではまだマージしません。",
        "承認後に統合します。",
    )
    reference = FrozenReference(
        reference_id="reference",
        source_audio_sha256=AUDIO,
        text="".join(windows),
        window_texts=windows,
        critical_tokens=(CriticalReferenceToken(kind="negation", text="ません"),),
    )
    case = DocumentExperimentCase(
        case_id="case",
        first_pass=first_pass(),
        reference=reference,
        rights_decision="allow",
        license_id="fixture-license",
        source_id="source",
        speaker_id="test-speaker",
        session_id="test-session",
        dataset_revision="r1",
        split_manifest_sha256=SPLIT,
    )
    return DocumentExperimentManifest(
        name="fixture",
        revision="r1",
        cases=(case,),
        rights_registry_sha256=RIGHTS,
        split_manifest_sha256=SPLIT,
    )


def fixture_protocol() -> DocumentExperimentProtocol:
    return DocumentExperimentProtocol(
        name="fixture",
        revision="r1",
        arms=(
            DocumentExperimentArm(
                name="acoustic",
                candidate_view="acoustic-only",
                direction="none",
                is_control=True,
            ),
            DocumentExperimentArm(
                name="ordered",
                candidate_view="ordered-document",
                direction="bidirectional",
                scorer_key="fixture",
            ),
            DocumentExperimentArm(
                name="shuffled",
                candidate_view="shuffled-document",
                direction="bidirectional",
                scorer_key="fixture",
                is_control=True,
            ),
        ),
        baseline_arm="acoustic",
        maximum_candidate_documents=3,
        maximum_scored_characters=10_000,
        bootstrap_resamples=200,
    )


def report(*, shuffled_also_prefers_correction: bool):
    manifest = fixture_manifest()
    protocol = fixture_protocol()
    prepared = prepare_document_experiment(manifest, protocol, lambda view: fake_plan())
    return run_document_context_experiment(
        prepared,
        manifest,
        protocol,
        scorers={
            "fixture": ArmAwareScorer(
                shuffled_also_prefers_correction=shuffled_also_prefers_correction
            )
        },
    )


def policy() -> DocumentContextPromotionPolicy:
    return DocumentContextPromotionPolicy(
        target_arm="ordered",
        baseline_arm="acoustic",
        shuffled_control_arm="shuffled",
        minimum_absolute_strict_cer_reduction=0.001,
        maximum_bootstrap_upper_delta=-0.001,
        minimum_ordered_advantage_over_shuffled=0.001,
        minimum_coverage=1.0,
        maximum_mean_latency_ms=60_000.0,
    )


def test_promotion_rejects_gain_shared_by_shuffled_control() -> None:
    decision = evaluate_document_context_promotion(
        report(shuffled_also_prefers_correction=True),
        policy(),
    )

    assert not decision.passed
    assert any(
        check.name == "ordered-vs-shuffled" and not check.passed for check in decision.checks
    )


def test_promotion_passes_when_ordered_arm_uniquely_improves_without_regression() -> None:
    decision = evaluate_document_context_promotion(
        report(shuffled_also_prefers_correction=False),
        policy(),
    )

    assert decision.passed
    assert not decision.reasons
    assert decision.digest
