from __future__ import annotations

from pathlib import Path

import pytest

from semantic_asr.deliberation_lattice import DocumentContext, LatticeArc
from semantic_asr.document_ranker import (
    DocumentFeatureConfig,
    DocumentRankExample,
    DocumentRankInput,
    DocumentRankTrainingConfig,
    DocumentRankerArtifact,
    DocumentRankerGlobalScorer,
    fit_document_ranker_calibration,
    group_top1_accuracy,
    pairwise_accuracy,
    train_document_ranker,
)


def example(group: str, candidate: str, text: str, cer: float, *, retained: bool):
    return DocumentRankExample(
        group_id=group,
        candidate_id=candidate,
        rank_input=DocumentRankInput(
            text=text,
            left_context="レビュー中です。",
            right_context="承認後に統合します。",
            topic_summary="変更のマージ判断",
            local_score=0.4 if retained else 0.35,
            overlap_score=0.1,
            mean_audio_support=0.7 if retained else 0.68,
            changed_window_count=0 if retained else 1,
            window_count=2,
            retained_path=retained,
        ),
        character_error_rate=cer,
        critical_error_count=0 if "まだ" in text else 1,
        first_pass_exact=retained and cer == 0.0,
    )


def training_examples():
    return (
        example("g1", "g1-good", "この変更はまだマージしません。", 0.0, retained=False),
        example("g1", "g1-bad", "この変更はまたマージしません。", 0.15, retained=True),
        example("g2", "g2-good", "承認後に実行します。", 0.0, retained=True),
        example("g2", "g2-bad", "承認後に中止します。", 0.20, retained=False),
        example("g3", "g3-good", "三千円です。", 0.0, retained=True),
        example("g3", "g3-bad", "三万円です。", 0.20, retained=False),
    )


def train():
    config = DocumentRankTrainingConfig(
        epochs=40,
        learning_rate=0.1,
        random_seed=7,
    )
    model = train_document_ranker(
        training_examples(),
        training_manifest_sha256="a" * 64,
        revision="ranker-r1",
        feature_config=DocumentFeatureConfig(hash_dimension=2_048),
        training_config=config,
    )
    calibration = fit_document_ranker_calibration(
        model,
        training_examples(),
        calibration_manifest_sha256="b" * 64,
        revision="cal-r1",
    )
    return model, calibration, config


def test_pairwise_trainer_learns_document_preferences() -> None:
    model, _calibration, config = train()

    assert model.pairwise_accuracy >= 0.99
    assert pairwise_accuracy(model, training_examples(), config) >= 0.99
    assert group_top1_accuracy(model, training_examples(), config) >= 0.99
    assert len(model.epoch_losses) == config.epochs
    assert model.epoch_losses[-1] < model.epoch_losses[0]


def test_artifact_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    model, calibration, config = train()
    artifact = DocumentRankerArtifact(
        model=model,
        calibration=calibration,
        test_manifest_sha256="c" * 64,
        test_pairwise_accuracy=pairwise_accuracy(model, training_examples(), config),
        test_group_top1_accuracy=group_top1_accuracy(model, training_examples(), config),
    )
    path = artifact.save(tmp_path / "ranker.json")

    loaded = DocumentRankerArtifact.load(path)

    assert loaded.digest == artifact.digest
    assert loaded.model.score_input(training_examples()[0].rank_input) == pytest.approx(
        artifact.model.score_input(training_examples()[0].rank_input)
    )

    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"bias": 0.0', '"bias": 1.0'), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        DocumentRankerArtifact.load(path)


def test_global_scorer_reads_complete_path_and_dense_evidence() -> None:
    model, calibration, config = train()
    artifact = DocumentRankerArtifact(
        model=model,
        calibration=calibration,
        test_manifest_sha256="c" * 64,
        test_pairwise_accuracy=pairwise_accuracy(model, training_examples(), config),
        test_group_top1_accuracy=group_top1_accuracy(model, training_examples(), config),
    )
    scorer = DocumentRankerGlobalScorer(artifact)
    context = DocumentContext(
        left_context="レビュー中です。",
        right_context="承認後に統合します。",
        topic_summary="変更のマージ判断",
    )
    good = (
        LatticeArc(
            arc_id="good",
            span_id="document",
            text="この変更はまだマージしません。",
            origin="first-pass",
            utilities=(),
            observed_eligible=False,
            metadata={
                "localScore": 0.35,
                "overlapScore": 0.1,
                "meanAudioSupport": 0.68,
                "changedWindowCount": 1,
                "windowCount": 2,
            },
        ),
    )
    bad = (
        LatticeArc(
            arc_id="bad",
            span_id="document",
            text="この変更はまたマージしません。",
            origin="first-pass",
            utilities=(),
            observed_eligible=False,
            metadata={
                "localScore": 0.4,
                "overlapScore": 0.1,
                "meanAudioSupport": 0.7,
                "changedWindowCount": 0,
                "windowCount": 2,
                "retainedPath": True,
            },
        ),
    )

    scores = scorer.score_many((good, bad), context=context)

    assert scores[0].value > scores[1].value
    assert all(score.profile_digest == artifact.digest for score in scores)
    assert all(score.context_digest == context.digest for score in scores)


def test_duplicate_example_identity_is_rejected() -> None:
    rows = training_examples()
    with pytest.raises(ValueError, match="unique"):
        train_document_ranker(
            (*rows, rows[0]),
            training_manifest_sha256="a" * 64,
            revision="bad",
        )
