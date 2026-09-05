from __future__ import annotations

import json
from pathlib import Path

import pytest
from _phonetic_runtime_fixture import write_wav

from semantic_asr.phonetic_runtime.manifest import (
    PhoneticManifestRow,
    PhoneticSplitManifest,
    load_phonetic_manifest,
    validate_split_isolation,
)


def row(tmp_path: Path, split: str, index: int) -> PhoneticManifestRow:
    path = tmp_path / f"{split}-{index}.wav"
    digest = write_wav(path, frequency=300.0 + index * 50)
    return PhoneticManifestRow(
        utterance_id=f"{split}-{index}",
        audio_path=path.resolve(),
        source_audio_sha256=digest,
        sample_rate=16_000,
        phone_symbols=("k", "a"),
        mora_symbols=("カ",),
        speaker_id=f"{split}-speaker-{index}",
        session_id=f"{split}-session-{index}",
        source_id=f"{split}-source-{index}",
        license_id="fixture-license",
        rights_decision="allow",
        split=split,  # type: ignore[arg-type]
    )


def manifest(tmp_path: Path) -> PhoneticSplitManifest:
    return PhoneticSplitManifest(
        name="fixture",
        revision="r1",
        rows=(
            row(tmp_path, "train", 1),
            row(tmp_path, "validation", 2),
            row(tmp_path, "calibration", 3),
            row(tmp_path, "test", 4),
        ),
        source_manifest_sha256="a" * 64,
    )


def test_split_manifest_is_speaker_session_and_source_disjoint(tmp_path) -> None:
    value = manifest(tmp_path)

    validate_split_isolation(value)
    assert len(value.rows_for("train")) == 1
    assert value.digest


def test_split_manifest_rejects_speaker_leakage(tmp_path) -> None:
    train = row(tmp_path, "train", 1)
    calibration = row(tmp_path, "calibration", 2)
    calibration = PhoneticManifestRow(
        utterance_id=calibration.utterance_id,
        audio_path=calibration.audio_path,
        source_audio_sha256=calibration.source_audio_sha256,
        sample_rate=calibration.sample_rate,
        phone_symbols=calibration.phone_symbols,
        mora_symbols=calibration.mora_symbols,
        speaker_id=train.speaker_id,
        session_id=calibration.session_id,
        source_id=calibration.source_id,
        license_id=calibration.license_id,
        rights_decision=calibration.rights_decision,
        split=calibration.split,
    )
    value = PhoneticSplitManifest(
        name="leaked",
        revision="r1",
        rows=(train, calibration),
        source_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="speaker leakage"):
        validate_split_isolation(value)


def test_manifest_row_requires_explicit_allow_rights(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    digest = write_wav(path)

    with pytest.raises(ValueError, match="rights_decision='allow'"):
        PhoneticManifestRow(
            utterance_id="x",
            audio_path=path.resolve(),
            source_audio_sha256=digest,
            sample_rate=16_000,
            phone_symbols=("a",),
            mora_symbols=("ア",),
            speaker_id="speaker",
            session_id="session",
            source_id="source",
            license_id="license",
            rights_decision="review",
            split="train",
        )


def test_jsonl_loader_requires_sidecar_and_absolute_audio_paths(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    digest = write_wav(path)
    manifest_path = tmp_path / "manifest.jsonl"
    payload = {
        "utteranceId": "train-1",
        "audioPath": str(path.resolve()),
        "sourceAudioSha256": digest,
        "sampleRate": 16_000,
        "phoneSymbols": ["k", "a"],
        "moraSymbols": ["カ"],
        "speakerId": "speaker-1",
        "sessionId": "session-1",
        "sourceId": "source-1",
        "licenseId": "fixture-license",
        "rightsDecision": "allow",
        "split": "train",
    }
    validation = dict(payload)
    validation.update(
        {
            "utteranceId": "validation-1",
            "audioPath": str((tmp_path / "validation.wav").resolve()),
            "sourceAudioSha256": write_wav(tmp_path / "validation.wav", frequency=500.0),
            "speakerId": "speaker-validation",
            "sessionId": "session-validation",
            "sourceId": "source-validation",
            "split": "validation",
        }
    )
    calibration = dict(payload)
    calibration.update(
        {
            "utteranceId": "cal-1",
            "audioPath": str((tmp_path / "cal.wav").resolve()),
            "sourceAudioSha256": write_wav(tmp_path / "cal.wav", frequency=600.0),
            "speakerId": "speaker-2",
            "sessionId": "session-2",
            "sourceId": "source-2",
            "split": "calibration",
        }
    )
    manifest_path.write_text(
        json.dumps(payload) + "\n" + json.dumps(validation) + "\n" + json.dumps(calibration) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata sidecar"):
        load_phonetic_manifest(manifest_path)

    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps({"name": "fixture", "revision": "r1"}),
        encoding="utf-8",
    )
    loaded = load_phonetic_manifest(manifest_path)
    assert len(loaded.rows) == 3
