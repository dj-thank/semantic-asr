from __future__ import annotations

import pytest
from _document_experiment_fixture import fake_paths

from semantic_asr.document_experiment.ngram_scorer import (
    BidirectionalCharacterNgramScorer,
    FrozenCharacterNgramModel,
    NgramCalibrationSequence,
    fit_character_ngram_model,
    fit_ngram_normalization,
)
from semantic_asr.document_experiment.protocol import DocumentExperimentArm

TRAIN = "1" * 64
CALIBRATION = "2" * 64


def scorer() -> BidirectionalCharacterNgramScorer:
    texts = (
        "レビュー完了まではまだマージしません。承認後に統合します。",
        "確認が終わるまではまだ公開しません。完了後に公開します。",
        "承認されるまではまだ反映しません。承認後に反映します。",
        "検証中なのでまだ変更しません。検証完了後に変更します。",
    )
    forward = fit_character_ngram_model(
        texts,
        order=4,
        alpha=0.2,
        training_manifest_sha256=TRAIN,
        revision="forward-r1",
    )
    backward = fit_character_ngram_model(
        texts,
        order=4,
        alpha=0.2,
        training_manifest_sha256=TRAIN,
        revision="backward-r1",
        reversed_text=True,
    )
    normalization = fit_ngram_normalization(
        forward,
        backward,
        (
            NgramCalibrationSequence(text=texts[0]),
            NgramCalibrationSequence(text=texts[1]),
            NgramCalibrationSequence(text=texts[2]),
        ),
        calibration_manifest_sha256=CALIBRATION,
        revision="normalization-r1",
    )
    return BidirectionalCharacterNgramScorer(forward, backward, normalization)


def arm(*, view="ordered-document", direction="bidirectional") -> DocumentExperimentArm:
    return DocumentExperimentArm(
        name=f"{view}-{direction}",
        candidate_view=view,
        direction=direction,
        scorer_key="ngram",
    )


def test_bidirectional_ngram_prefers_trained_contextual_sequence() -> None:
    model = scorer()
    retained, corrected, _harmful = fake_paths()
    experiment_arm = arm()

    retained_score = model.score_path(
        retained,
        experiment_arm,
        case_id="case",
        maximum_characters=10_000,
    )
    corrected_score = model.score_path(
        corrected,
        experiment_arm,
        case_id="case",
        maximum_characters=10_000,
    )

    assert corrected_score.raw_average_log_likelihood > (retained_score.raw_average_log_likelihood)
    assert corrected_score.value > retained_score.value
    assert corrected_score.scorer_calls == 2
    assert corrected_score.profile_digest == model.profile_digest


def test_shuffled_control_is_deterministic_and_arm_bound() -> None:
    model = scorer()
    retained, _corrected, _harmful = fake_paths()
    shuffled = arm(view="shuffled-document")

    first = model.score_path(
        retained,
        shuffled,
        case_id="case-1",
        maximum_characters=10_000,
    )
    second = model.score_path(
        retained,
        shuffled,
        case_id="case-1",
        maximum_characters=10_000,
    )

    assert first == second
    assert first.arm_digest == shuffled.digest


def test_forward_arm_uses_one_scorer_pass() -> None:
    model = scorer()
    retained, _corrected, _harmful = fake_paths()
    forward = arm(direction="forward")

    result = model.score_path(
        retained,
        forward,
        case_id="case",
        left_context="前の議論です。",
        maximum_characters=10_000,
    )

    assert result.scorer_calls == 1
    assert result.backward_average_log_likelihood is None


def test_ngram_artifact_rejects_duplicate_context_rows() -> None:
    valid = scorer().forward
    duplicated = (*valid.rows, valid.rows[0])

    with pytest.raises(ValueError, match="duplicate n-gram context row"):
        FrozenCharacterNgramModel(
            order=valid.order,
            alpha=valid.alpha,
            vocabulary=valid.vocabulary,
            rows=duplicated,
            training_manifest_sha256=valid.training_manifest_sha256,
            revision=valid.revision,
        )


def test_shuffled_control_is_not_identity_for_multi_window_document() -> None:
    model = scorer()
    retained, _corrected, _harmful = fake_paths()
    shuffled = arm(view="shuffled-document")
    ordered = arm(view="ordered-document")

    shuffled_score = model.score_path(
        retained,
        shuffled,
        case_id="identity-prone-case",
        maximum_characters=10_000,
    )
    ordered_score = model.score_path(
        retained,
        ordered,
        case_id="identity-prone-case",
        maximum_characters=10_000,
    )

    assert shuffled_score.arm_digest != ordered_score.arm_digest
    assert shuffled_score.raw_average_log_likelihood != ordered_score.raw_average_log_likelihood
