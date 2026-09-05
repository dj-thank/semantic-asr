from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_asr.phonetic_runtime.manifest import load_phonetic_manifest
from semantic_asr.phonetic_runtime.materialize import materialize_japanese_phonetic_manifest

from _phonetic_runtime_fixture import write_wav


def source_rows(tmp_path: Path) -> list[dict[str, object]]:
    rows = []
    for index, split in enumerate(("train", "validation", "calibration", "test"), 1):
        audio = tmp_path / f"{split}.wav"
        write_wav(audio, frequency=240.0 + index * 70.0)
        rows.append(
            {
                "utteranceId": f"{split}-1",
                "audioPath": str(audio.resolve()),
                "reading": "がっこうへいく",
                "transcript": "学校へ行く",
                "speakerId": f"{split}-speaker",
                "sessionId": f"{split}-session",
                "sourceId": f"{split}-source",
                "licenseId": "fixture-license",
                "rightsDecision": "allow",
                "split": split,
            }
        )
    return rows


def write_input(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "input.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_materializer_outputs_loadable_manifest_without_raw_transcript(tmp_path: Path) -> None:
    source = write_input(tmp_path, source_rows(tmp_path))
    output = tmp_path / "outside" / "materialized"

    result = materialize_japanese_phonetic_manifest(
        source,
        output,
        name="fixture",
        revision="r1",
    )
    loaded = load_phonetic_manifest(result.manifest_path)
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    provenance = json.loads(result.provenance_path.read_text(encoding="utf-8"))

    assert len(loaded.rows) == 4
    assert "学校へ行く" not in manifest_text
    assert "transcriptSha256" in manifest_text
    assert provenance["rawTranscriptStored"] is False
    assert provenance["labelProfileDigest"] == result.label_profile_digest
    assert result.digest


def test_materializer_requires_allow_rights_and_explicit_reading(tmp_path: Path) -> None:
    rows = source_rows(tmp_path)
    rows[0]["rightsDecision"] = "review"
    source = write_input(tmp_path, rows)

    with pytest.raises(ValueError, match="rightsDecision='allow'"):
        materialize_japanese_phonetic_manifest(
            source,
            tmp_path / "rights-output",
            name="fixture",
            revision="r1",
        )

    rows = source_rows(tmp_path)
    rows[0]["reading"] = "学校"
    source = write_input(tmp_path, rows)
    with pytest.raises(ValueError, match="Kanji readings must be supplied explicitly"):
        materialize_japanese_phonetic_manifest(
            source,
            tmp_path / "reading-output",
            name="fixture",
            revision="r1",
        )


def test_materializer_rejects_split_leakage_and_existing_destination(tmp_path: Path) -> None:
    rows = source_rows(tmp_path)
    rows[1]["speakerId"] = rows[0]["speakerId"]
    source = write_input(tmp_path, rows)

    with pytest.raises(ValueError, match="speaker leakage"):
        materialize_japanese_phonetic_manifest(
            source,
            tmp_path / "leaked-output",
            name="fixture",
            revision="r1",
        )

    source = write_input(tmp_path, source_rows(tmp_path))
    destination = tmp_path / "already-exists"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_japanese_phonetic_manifest(
            source,
            destination,
            name="fixture",
            revision="r1",
        )


def test_materializer_is_byte_reproducible_across_fresh_destinations(tmp_path: Path) -> None:
    source = write_input(tmp_path, source_rows(tmp_path))
    first = materialize_japanese_phonetic_manifest(
        source,
        tmp_path / "first",
        name="fixture",
        revision="r1",
    )
    second = materialize_japanese_phonetic_manifest(
        source,
        tmp_path / "second",
        name="fixture",
        revision="r1",
    )

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.phone_inventory_path.read_bytes() == second.phone_inventory_path.read_bytes()
    assert first.mora_inventory_path.read_bytes() == second.mora_inventory_path.read_bytes()
    assert first.provenance_path.read_bytes() == second.provenance_path.read_bytes()
    assert first.digest == second.digest
