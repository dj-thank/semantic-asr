#!/usr/bin/env python3
"""Train and evaluate the shared phone/mora CTC head from frozen `.npy` features."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from semantic_asr.contracts import sha256_json
from semantic_asr.phonetic_dataset import (
    PhoneticDatasetResourcePolicy,
    load_phonetic_feature_manifest,
    validate_phonetic_split_disjointness,
)
from semantic_asr.phonetic_heads_optional import JointPhoneMoraCTCHead
from semantic_asr.phonetic_trainer_optional import (
    PhoneticOptimizationConfig,
    build_joint_phonetic_artifact,
    evaluate_joint_phonetic_head,
    save_joint_phonetic_weights,
    train_joint_phonetic_head,
)
from semantic_asr.phonetic_training import (
    JointPhoneticHeadConfig,
    PhoneticLabelInventory,
    PhoneticTrainingManifest,
)


def _exact_config(path: Path) -> JointPhoneticHeadConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schemaVersion",
        "inputDimension",
        "hiddenDimension",
        "encoderId",
        "encoderRevision",
        "encoderArtifactSha256",
        "dropout",
        "phoneLossWeight",
        "moraLossWeight",
        "blankRegularizationWeight",
        "phoneInventory",
        "moraInventory",
    }
    if set(payload) != expected:
        raise ValueError(
            "joint phonetic config schema is not exact; "
            f"missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )

    def inventory(value: object, kind: str) -> PhoneticLabelInventory:
        if not isinstance(value, dict):
            raise ValueError(f"{kind} inventory must be an object")
        inventory_expected = {
            "schemaVersion",
            "labels",
            "blankSymbol",
            "revision",
            "sourceManifestSha256",
        }
        if set(value) != inventory_expected:
            raise ValueError(f"{kind} inventory schema is not exact")
        return PhoneticLabelInventory(
            kind=kind,
            labels=tuple(str(row) for row in value["labels"]),
            blank_symbol=str(value["blankSymbol"]),
            revision=str(value["revision"]),
            source_manifest_sha256=str(value["sourceManifestSha256"]),
            schema_version=str(value["schemaVersion"]),
        )

    return JointPhoneticHeadConfig(
        input_dimension=int(payload["inputDimension"]),
        hidden_dimension=int(payload["hiddenDimension"]),
        phone_inventory=inventory(payload["phoneInventory"], "phone"),
        mora_inventory=inventory(payload["moraInventory"], "mora"),
        encoder_id=str(payload["encoderId"]),
        encoder_revision=str(payload["encoderRevision"]),
        encoder_artifact_sha256=(
            None
            if payload["encoderArtifactSha256"] is None
            else str(payload["encoderArtifactSha256"])
        ),
        dropout=float(payload["dropout"]),
        phone_loss_weight=float(payload["phoneLossWeight"]),
        mora_loss_weight=float(payload["moraLossWeight"]),
        blank_regularization_weight=float(payload["blankRegularizationWeight"]),
        schema_version=str(payload["schemaVersion"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--rights-registry-sha256", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-items", type=int, default=2_000_000)
    parser.add_argument("--maximum-frames-per-item", type=int, default=60_000)
    parser.add_argument("--maximum-total-feature-cells", type=int, default=10_000_000_000)
    parser.add_argument("--target-true-accept-rate", type=float, default=0.95)
    args = parser.parse_args()

    head_config = _exact_config(args.config)
    resources = PhoneticDatasetResourcePolicy(
        maximum_items=args.maximum_items,
        maximum_frames_per_item=args.maximum_frames_per_item,
        maximum_feature_dimension=max(1, head_config.input_dimension),
        maximum_total_feature_cells=args.maximum_total_feature_cells,
    )
    train = load_phonetic_feature_manifest(
        args.train,
        split="train",
        phone_inventory=head_config.phone_inventory,
        mora_inventory=head_config.mora_inventory,
        resources=resources,
    )
    calibration = load_phonetic_feature_manifest(
        args.calibration,
        split="calibration",
        phone_inventory=head_config.phone_inventory,
        mora_inventory=head_config.mora_inventory,
        resources=resources,
    )
    test = load_phonetic_feature_manifest(
        args.test,
        split="test",
        phone_inventory=head_config.phone_inventory,
        mora_inventory=head_config.mora_inventory,
        resources=resources,
    )
    validate_phonetic_split_disjointness(train, calibration, test)
    training_manifest = PhoneticTrainingManifest(
        training_manifest_sha256=train.manifest_sha256,
        calibration_manifest_sha256=calibration.manifest_sha256,
        test_manifest_sha256=test.manifest_sha256,
        speaker_disjoint=True,
        source_disjoint=True,
        rights_registry_sha256=args.rights_registry_sha256,
        feature_revision=train.feature_revision,
        random_seed=args.seed,
    )
    optimization = PhoneticOptimizationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        random_seed=args.seed,
        device=args.device,
    )
    model = JointPhoneMoraCTCHead(head_config)
    history = train_joint_phonetic_head(model, train, optimization=optimization)
    calibration_result = evaluate_joint_phonetic_head(
        model,
        calibration,
        device=args.device,
        fit_calibration=True,
        target_true_accept_rate=args.target_true_accept_rate,
        calibration_revision=f"{args.revision}-sequence-calibration",
    )
    test_result = evaluate_joint_phonetic_head(
        model,
        test,
        device=args.device,
        fit_calibration=False,
        target_true_accept_rate=args.target_true_accept_rate,
    )
    weights_sha256 = save_joint_phonetic_weights(
        model,
        args.weights,
        config_digest=head_config.digest,
    )

    import torch

    artifact = build_joint_phonetic_artifact(
        head_config=head_config,
        training_manifest=training_manifest,
        weights_sha256=weights_sha256,
        test_evaluation=test_result,
        framework_version=torch.__version__,
        revision=args.revision,
    )
    artifact_payload = {
        "schemaVersion": "1",
        "headConfig": asdict(head_config),
        "headConfigDigest": head_config.digest,
        "trainingManifest": asdict(training_manifest),
        "trainingManifestDigest": training_manifest.digest,
        "optimizationConfig": asdict(optimization),
        "optimizationConfigDigest": optimization.digest,
        "history": asdict(history),
        "historyDigest": history.digest,
        "sequenceCalibration": asdict(calibration_result.calibration),
        "sequenceCalibrationDigest": calibration_result.calibration.digest,
        "testEvaluation": {
            "metrics": asdict(test_result.metrics),
            "digest": test_result.digest,
        },
        "artifact": asdict(artifact),
        "artifactDigest": artifact.digest,
        "weightsFile": args.weights.name,
        "weightsSha256": weights_sha256,
        "claimBoundary": (
            "trained head artifact only; runtime or ASR promotion requires locked end-to-end "
            "document evaluation and target-device profiling"
        ),
    }
    artifact_payload["envelopeDigest"] = sha256_json(artifact_payload)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schemaVersion": "1",
        "artifactDigest": artifact.digest,
        "envelopeDigest": artifact_payload["envelopeDigest"],
        "weightsSha256": weights_sha256,
        "trainManifestDigest": train.digest,
        "calibrationManifestDigest": calibration.digest,
        "testManifestDigest": test.digest,
        "history": asdict(history),
        "calibrationMetrics": asdict(calibration_result.metrics),
        "calibrationProfile": asdict(calibration_result.calibration),
        "testMetrics": asdict(test_result.metrics),
        "resourcePolicyDigest": resources.digest,
        "claimBoundary": artifact_payload["claimBoundary"],
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
