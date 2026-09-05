from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def row():
    return {
        "schemaVersion": "1",
        "utteranceId": "utt-1",
        "split": "train",
        "audioPath": "audio/utt-1.wav",
        "audioSha256": sha("audio"),
        "sampleRate": 16000,
        "segmentStartMs": 0,
        "segmentEndMs": 1000,
        "transcript": "学校へ行く",
        "explicitReading": "ガッコウヘイク",
        "speakerId": "speaker-1",
        "sourceId": "source-1",
        "rightsDecision": "allow",
        "licenseId": "fixture-license",
    }


def write_input(path: Path) -> None:
    path.write_text(json.dumps(row(), ensure_ascii=False) + "\n", encoding="utf-8")


def command(input_path: Path, output: Path):
    return [
        sys.executable,
        "scripts/prepare_phonetic_readings.py",
        "--input",
        str(input_path),
        "--split",
        "train",
        "--output",
        str(output),
    ]


def test_cli_prepares_human_reading_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output = tmp_path / "prepared" / "train.jsonl"
    write_input(input_path)

    completed = subprocess.run(
        [*command(input_path, output), "--allow-output"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert output.exists()
    assert output.with_suffix(".jsonl.reading-receipts.jsonl").exists()
    assert report["originCounts"] == {"human-explicit": 1}
    assert report["itemCount"] == 1
    assert len(report["resultDigest"]) == 64


def test_cli_requires_explicit_output_authorization(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output = tmp_path / "train.jsonl"
    write_input(input_path)

    completed = subprocess.run(
        command(input_path, output),
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--allow-output is required" in completed.stderr
    assert not output.exists()


def test_cli_requires_complete_review_metadata_group(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output = tmp_path / "train.jsonl"
    write_input(input_path)
    review = tmp_path / "reviews.jsonl"
    review.write_text("", encoding="utf-8")

    completed = subprocess.run(
        [
            *command(input_path, output),
            "--allow-output",
            "--review-ledger",
            str(review),
            "--review-ledger-revision",
            "r1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--review-ledger requires" in completed.stderr
    assert not output.exists()
