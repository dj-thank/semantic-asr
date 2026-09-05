from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from semantic_asr.phonetic_runtime.manifest import load_phonetic_manifest

from _phonetic_runtime_fixture import write_wav

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_japanese_phonetic_manifest.py"


def write_source(tmp_path: Path) -> Path:
    rows = []
    for index, split in enumerate(("train", "validation", "calibration", "test"), 1):
        audio = tmp_path / f"{split}.wav"
        write_wav(audio, frequency=300.0 + index * 50.0)
        rows.append(
            {
                "utteranceId": f"{split}-1",
                "audioPath": str(audio.resolve()),
                "reading": "まだまーじしません",
                "transcript": "まだマージしません",
                "speakerId": f"{split}-speaker",
                "sessionId": f"{split}-session",
                "sourceId": f"{split}-source",
                "licenseId": "fixture-license",
                "rightsDecision": "allow",
                "split": split,
            }
        )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return source


def test_cli_requires_explicit_derived_label_acknowledgement(tmp_path: Path) -> None:
    source = write_source(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
            "--name",
            "fixture",
            "--revision",
            "r1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert result.returncode != 0
    assert "--allow-derived-phonetic-labels is required" in result.stdout


def test_cli_materializes_loadable_four_way_manifest(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    output = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output-dir",
            str(output),
            "--name",
            "fixture",
            "--revision",
            "r1",
            "--allow-derived-phonetic-labels",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    loaded = load_phonetic_manifest(output / "manifest.jsonl")
    assert len(loaded.rows) == 4
    assert payload["materializationDigest"]
    assert (output / "materialization.json").is_file()
