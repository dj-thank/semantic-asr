from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from semantic_asr.phonetic_runtime.manifest import (
    PhoneticManifestRow,
    PhoneticSplitManifest,
    validate_split_isolation,
)

from _phonetic_runtime_fixture import write_wav


def row(tmp_path: Path, split: str, index: int) -> PhoneticManifestRow:
    path = tmp_path / f"{split}-{index}.wav"
    digest = write_wav(path, frequency=220.0 + index * 30.0)
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


def four_way_manifest(tmp_path: Path) -> PhoneticSplitManifest:
    return PhoneticSplitManifest(
        name="four-way",
        revision="r1",
        rows=(
            row(tmp_path, "train", 1),
            row(tmp_path, "validation", 2),
            row(tmp_path, "calibration", 3),
            row(tmp_path, "test", 4),
        ),
        source_manifest_sha256="a" * 64,
    )


def test_four_way_split_isolation_passes() -> None:
    value = four_way_manifest(Path(pytest.ensuretemp("four-way-split")))

    validate_split_isolation(value)

    assert len(value.rows_for("validation")) == 1
    assert len(value.rows_for("calibration")) == 1
    assert len(value.rows_for("test")) == 1


def test_validation_and_calibration_may_not_share_speakers_sessions_or_sources(tmp_path: Path) -> None:
    value = four_way_manifest(tmp_path)
    rows = list(value.rows)
    validation = next(row for row in rows if row.split == "validation")
    calibration_index = next(index for index, row in enumerate(rows) if row.split == "calibration")
    rows[calibration_index] = replace(
        rows[calibration_index],
        speaker_id=validation.speaker_id,
    )
    leaked = replace(value, rows=tuple(rows))

    with pytest.raises(ValueError, match="speaker leakage between validation and calibration"):
        validate_split_isolation(leaked)


def test_model_selection_requires_validation_not_score_calibration(tmp_path: Path) -> None:
    value = PhoneticSplitManifest(
        name="missing-validation",
        revision="r1",
        rows=(
            row(tmp_path, "train", 1),
            row(tmp_path, "calibration", 2),
            row(tmp_path, "test", 3),
        ),
        source_manifest_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="validation"):
        validate_split_isolation(value)
