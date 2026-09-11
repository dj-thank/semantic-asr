from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_asr.phonetic_dataset import (
    PhoneticDatasetResourcePolicy,
    file_sha256,
    load_feature_array,
    load_phonetic_feature_manifest,
    validate_phonetic_split_disjointness,
)
from semantic_asr.phonetic_training import PhoneticLabelInventory


def inventory(kind: str, labels: tuple[str, ...], marker: str) -> PhoneticLabelInventory:
    return PhoneticLabelInventory(
        kind=kind,
        labels=labels,
        blank_symbol="<blk>",
        revision=f"{kind}-r1",
        source_manifest_sha256=marker * 64,
    )


PHONE = inventory("phone", ("<blk>", "a", "k"), "a")
MORA = inventory("mora", ("<blk>", "ア", "カ"), "b")


def row(split: str, prefix: str, *, rights: str = "allow") -> dict[str, object]:
    return {
        "schemaVersion": "1",
        "utteranceId": f"{prefix}-utterance",
        "split": split,
        "featurePath": f"features/{prefix}.npy",
        "featureSha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "frameCount": 8,
        "featureDimension": 4,
        "featureDtype": "float32",
        "phoneTargets": [1, 2],
        "moraTargets": [1, 2],
        "phoneInventoryDigest": PHONE.digest,
        "moraInventoryDigest": MORA.digest,
        "speakerId": f"{prefix}-speaker",
        "sourceId": f"{prefix}-source",
        "sourceAudioSha256": hashlib.sha256(f"audio-{prefix}".encode()).hexdigest(),
        "featureRevision": "frozen-features-r1",
        "rightsDecision": rights,
        "licenseId": "test-license",
    }


def write_manifest(path: Path, values) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def load(path: Path, split: str):
    return load_phonetic_feature_manifest(
        path,
        split=split,  # type: ignore[arg-type]
        phone_inventory=PHONE,
        mora_inventory=MORA,
        resources=PhoneticDatasetResourcePolicy(maximum_items=10),
    )


def test_manifest_requires_exact_rights_and_inventory_binding(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    write_manifest(manifest, (row("train", "train-1"),))

    loaded = load(manifest, "train")

    assert loaded.manifest_sha256 == file_sha256(manifest)
    assert loaded.items[0].phone_inventory_digest == PHONE.digest
    assert loaded.items[0].mora_inventory_digest == MORA.digest

    denied = tmp_path / "denied.jsonl"
    write_manifest(denied, (row("train", "denied", rights="review"),))
    with pytest.raises(ValueError, match="rights_decision='allow'"):
        load(denied, "train")


def test_split_validation_rejects_speaker_source_and_audio_leakage(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    test_path = tmp_path / "test.jsonl"
    train_row = row("train", "shared")
    calibration_row = row("calibration", "cal")
    calibration_row["speakerId"] = train_row["speakerId"]
    write_manifest(train_path, (train_row,))
    write_manifest(calibration_path, (calibration_row,))
    write_manifest(test_path, (row("test", "test"),))

    with pytest.raises(ValueError, match="speaker"):
        validate_phonetic_split_disjointness(
            load(train_path, "train"),
            load(calibration_path, "calibration"),
            load(test_path, "test"),
        )


def test_manifest_rejects_schema_extensions_and_path_traversal(tmp_path: Path) -> None:
    unknown = row("train", "unknown")
    unknown["unexpected"] = True
    path = tmp_path / "unknown.jsonl"
    write_manifest(path, (unknown,))
    with pytest.raises(ValueError, match="non-exact schema"):
        load(path, "train")

    traversal = row("train", "traversal")
    traversal["featurePath"] = "../escape.npy"
    path = tmp_path / "traversal.jsonl"
    write_manifest(path, (traversal,))
    with pytest.raises(ValueError, match="traverse"):
        load(path, "train")


def test_digest_verified_numpy_loading(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    array = numpy.arange(32, dtype=numpy.float32).reshape(8, 4)
    feature = feature_dir / "array.npy"
    numpy.save(feature, array, allow_pickle=False)
    value = row("train", "array")
    value["featurePath"] = "features/array.npy"
    value["featureSha256"] = file_sha256(feature)
    manifest_path = tmp_path / "train.jsonl"
    write_manifest(manifest_path, (value,))
    manifest = load(manifest_path, "train")

    loaded = load_feature_array(manifest, manifest.items[0])

    assert loaded.shape == (8, 4)
    feature.write_bytes(feature.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_feature_array(manifest, manifest.items[0])
