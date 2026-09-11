from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("safetensors")

from semantic_asr.phonetic_dataset import file_sha256  # noqa: E402
from semantic_asr.phonetic_training import PhoneticLabelInventory  # noqa: E402


PHONE = PhoneticLabelInventory(
    kind="phone",
    labels=("<blk>", "a", "k", "m"),
    blank_symbol="<blk>",
    revision="phone-r1",
    source_manifest_sha256="a" * 64,
)
MORA = PhoneticLabelInventory(
    kind="mora",
    labels=("<blk>", "ア", "カ", "マ"),
    blank_symbol="<blk>",
    revision="mora-r1",
    source_manifest_sha256="b" * 64,
)


def write_config(path: Path) -> None:
    payload = {
        "schemaVersion": "1",
        "inputDimension": 6,
        "hiddenDimension": 8,
        "encoderId": "frozen-fixture-encoder",
        "encoderRevision": "1" * 40,
        "encoderArtifactSha256": "c" * 64,
        "dropout": 0.0,
        "phoneLossWeight": 1.0,
        "moraLossWeight": 1.0,
        "blankRegularizationWeight": 0.0,
        "phoneInventory": {
            "schemaVersion": "1",
            "labels": PHONE.labels,
            "blankSymbol": PHONE.blank_symbol,
            "revision": PHONE.revision,
            "sourceManifestSha256": PHONE.source_manifest_sha256,
        },
        "moraInventory": {
            "schemaVersion": "1",
            "labels": MORA.labels,
            "blankSymbol": MORA.blank_symbol,
            "revision": MORA.revision,
            "sourceManifestSha256": MORA.source_manifest_sha256,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split(root: Path, split: str, prefix: str) -> Path:
    feature_dir = root / "features"
    feature_dir.mkdir(exist_ok=True)
    rows = []
    for index in range(2):
        randomizer = numpy.random.default_rng(index + len(prefix))
        values = randomizer.normal(size=(10, 6)).astype(numpy.float32)
        feature = feature_dir / f"{prefix}-{index}.npy"
        numpy.save(feature, values, allow_pickle=False)
        rows.append(
            {
                "schemaVersion": "1",
                "utteranceId": f"{prefix}-utt-{index}",
                "split": split,
                "featurePath": f"features/{feature.name}",
                "featureSha256": file_sha256(feature),
                "frameCount": 10,
                "featureDimension": 6,
                "featureDtype": "float32",
                "phoneTargets": [1, 2, 3],
                "moraTargets": [1, 2, 3],
                "phoneInventoryDigest": PHONE.digest,
                "moraInventoryDigest": MORA.digest,
                "speakerId": f"{prefix}-speaker-{index}",
                "sourceId": f"{prefix}-source-{index}",
                "sourceAudioSha256": hashlib.sha256(
                    f"{prefix}-audio-{index}".encode()
                ).hexdigest(),
                "featureRevision": "fixture-features-r1",
                "rightsDecision": "allow",
                "licenseId": "fixture-license",
            }
        )
    manifest = root / f"{split}.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_training_cli_emits_safetensors_artifact_and_locked_test_report(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    write_config(config)
    train = write_split(tmp_path, "train", "train")
    calibration = write_split(tmp_path, "calibration", "cal")
    test = write_split(tmp_path, "test", "test")
    weights = tmp_path / "joint.safetensors"
    artifact = tmp_path / "artifact.json"
    report = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_joint_phonetic_head.py",
            "--config",
            str(config),
            "--train",
            str(train),
            "--calibration",
            str(calibration),
            "--test",
            str(test),
            "--rights-registry-sha256",
            "d" * 64,
            "--revision",
            "fixture-joint-r1",
            "--weights",
            str(weights),
            "--artifact",
            str(artifact),
            "--report",
            str(report),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert weights.exists()
    assert file_sha256(weights) == artifact_payload["weightsSha256"]
    assert artifact_payload["artifact"]["serialization"] == "safetensors"
    assert artifact_payload["trainingManifest"]["speaker_disjoint"]
    assert artifact_payload["trainingManifest"]["source_disjoint"]
    assert report_payload["testMetrics"]["validation_sample_count"] == 2
    assert report_payload["calibrationProfile"]["calibration_manifest_sha256"] == (
        file_sha256(calibration)
    )
