#!/usr/bin/env python3
"""Train, calibrate, and evaluate the dependency-free document path ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_asr.document_ranker import (
    DocumentFeatureConfig,
    DocumentRankExample,
    DocumentRankInput,
    DocumentRankTrainingConfig,
    DocumentRankerArtifact,
    fit_document_ranker_calibration,
    group_top1_accuracy,
    manifest_sha256,
    pairwise_accuracy,
    train_document_ranker,
)


def load_examples(path: Path) -> tuple[DocumentRankExample, ...]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                rank_input = DocumentRankInput(
                    text=str(row["text"]),
                    left_context=str(row.get("leftContext", "")),
                    right_context=str(row.get("rightContext", "")),
                    topic_summary=str(row.get("topicSummary", "")),
                    entity_ids=tuple(str(value) for value in row.get("entityIds", ())),
                    local_score=float(row.get("localScore", 0.0)),
                    overlap_score=float(row.get("overlapScore", 0.0)),
                    mean_audio_support=float(row.get("meanAudioSupport", 0.0)),
                    changed_window_count=int(row.get("changedWindowCount", 0)),
                    generated_window_count=int(row.get("generatedWindowCount", 0)),
                    ambiguous_overlap_count=int(row.get("ambiguousOverlapCount", 0)),
                    window_count=int(row.get("windowCount", 1)),
                    retained_path=bool(row.get("retainedPath", False)),
                    metadata=dict(row.get("metadata", {})),
                )
                output.append(
                    DocumentRankExample(
                        group_id=str(row["groupId"]),
                        candidate_id=str(row["candidateId"]),
                        rank_input=rank_input,
                        character_error_rate=float(row["characterErrorRate"]),
                        critical_error_count=int(row.get("criticalErrorCount", 0)),
                        first_pass_exact=bool(row.get("firstPassExact", False)),
                        metadata=dict(row.get("exampleMetadata", {})),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid document ranker row at line {line_number}: {exc}") from exc
    if not output:
        raise ValueError(f"document ranker manifest is empty: {path}")
    return tuple(output)


def group_ids(examples: tuple[DocumentRankExample, ...]) -> set[str]:
    return {example.group_id for example in examples}


def require_disjoint_groups(
    train: tuple[DocumentRankExample, ...],
    calibration: tuple[DocumentRankExample, ...],
    test: tuple[DocumentRankExample, ...],
) -> None:
    train_groups = group_ids(train)
    calibration_groups = group_ids(calibration)
    test_groups = group_ids(test)
    overlaps = {
        "train-calibration": train_groups.intersection(calibration_groups),
        "train-test": train_groups.intersection(test_groups),
        "calibration-test": calibration_groups.intersection(test_groups),
    }
    contaminated = {name: sorted(values) for name, values in overlaps.items() if values}
    if contaminated:
        raise ValueError(f"document ranker group leakage detected: {contaminated}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--hash-dimension", type=int, default=32_768)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--critical-error-weight", type=float, default=2.0)
    parser.add_argument("--false-correction-weight", type=float, default=4.0)
    parser.add_argument("--maximum-pairs-per-group", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = load_examples(args.train)
    calibration = load_examples(args.calibration)
    test = load_examples(args.test)
    require_disjoint_groups(train, calibration, test)

    feature_config = DocumentFeatureConfig(
        hash_dimension=args.hash_dimension,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )
    training_config = DocumentRankTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        critical_error_weight=args.critical_error_weight,
        false_correction_weight=args.false_correction_weight,
        maximum_pairs_per_group=args.maximum_pairs_per_group,
        random_seed=args.seed,
    )
    training_digest = manifest_sha256(args.train)
    calibration_digest = manifest_sha256(args.calibration)
    test_digest = manifest_sha256(args.test)
    model = train_document_ranker(
        train,
        training_manifest_sha256=training_digest,
        revision=args.revision,
        feature_config=feature_config,
        training_config=training_config,
    )
    calibration_profile = fit_document_ranker_calibration(
        model,
        calibration,
        calibration_manifest_sha256=calibration_digest,
        revision=f"{args.revision}-calibration",
    )
    test_pairwise = pairwise_accuracy(model, test, training_config)
    test_top1 = group_top1_accuracy(model, test, training_config)
    artifact = DocumentRankerArtifact(
        model=model,
        calibration=calibration_profile,
        test_manifest_sha256=test_digest,
        test_pairwise_accuracy=test_pairwise,
        test_group_top1_accuracy=test_top1,
    )
    artifact.save(args.output)
    report = {
        "schemaVersion": "1",
        "artifactDigest": artifact.digest,
        "modelDigest": model.digest,
        "calibrationDigest": calibration_profile.digest,
        "featureConfigDigest": feature_config.digest,
        "trainingConfigDigest": training_config.digest,
        "trainingManifestSha256": training_digest,
        "calibrationManifestSha256": calibration_digest,
        "testManifestSha256": test_digest,
        "trainGroups": len(group_ids(train)),
        "calibrationGroups": len(group_ids(calibration)),
        "testGroups": len(group_ids(test)),
        "trainExamples": len(train),
        "calibrationExamples": len(calibration),
        "testExamples": len(test),
        "trainingPairwiseAccuracy": model.pairwise_accuracy,
        "testPairwiseAccuracy": test_pairwise,
        "testGroupTop1Accuracy": test_top1,
        "epochLosses": model.epoch_losses,
        "claimBoundary": (
            "ranker software evaluation only; deployment requires locked Japanese audio, "
            "first-pass-exact false-correction arms, and document promotion gates"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
