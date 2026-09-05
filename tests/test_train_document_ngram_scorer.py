from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from semantic_asr.document_experiment.artifacts import BidirectionalNgramArtifact

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_document_ngram_scorer.py"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def manifests(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    write_manifest(
        train,
        [
            {
                "text": "レビュー完了まではまだマージしません。",
                "speakerId": "train-speaker-1",
                "sessionId": "train-session-1",
                "licenseId": "fixture-license",
                "rightsDecision": "allow",
            },
            {
                "text": "確認中なのでまだ公開しません。",
                "speakerId": "train-speaker-2",
                "sessionId": "train-session-2",
                "licenseId": "fixture-license",
                "rightsDecision": "allow",
            },
        ],
    )
    write_manifest(
        calibration,
        [
            {
                "text": "承認後に統合します。",
                "speakerId": "cal-speaker-1",
                "sessionId": "cal-session-1",
                "licenseId": "fixture-license",
                "rightsDecision": "allow",
            },
            {
                "text": "検証完了後に公開します。",
                "speakerId": "cal-speaker-2",
                "sessionId": "cal-session-2",
                "licenseId": "fixture-license",
                "rightsDecision": "allow",
            },
        ],
    )
    return train, calibration


def test_training_cli_writes_tamper_checked_artifact(tmp_path: Path) -> None:
    train, calibration = manifests(tmp_path)
    output = tmp_path / "artifact.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train),
            "--calibration",
            str(calibration),
            "--output",
            str(output),
            "--revision",
            "fixture-r1",
            "--allow-derived-artifact",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout
    artifact = BidirectionalNgramArtifact.read(output)
    assert artifact.revision == "fixture-r1"
    assert artifact.scorer().profile_digest


def test_training_cli_requires_derived_artifact_acknowledgement(tmp_path: Path) -> None:
    train, calibration = manifests(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train),
            "--calibration",
            str(calibration),
            "--output",
            str(tmp_path / "artifact.json"),
            "--revision",
            "fixture-r1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert result.returncode != 0
    assert "--allow-derived-artifact is required" in result.stdout


def test_training_cli_rejects_speaker_overlap(tmp_path: Path) -> None:
    train, calibration = manifests(tmp_path)
    rows = [json.loads(line) for line in calibration.read_text(encoding="utf-8").splitlines()]
    rows[0]["speakerId"] = "train-speaker-1"
    write_manifest(calibration, rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train),
            "--calibration",
            str(calibration),
            "--output",
            str(tmp_path / "artifact.json"),
            "--revision",
            "fixture-r1",
            "--allow-derived-artifact",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert result.returncode != 0
    assert "training and calibration speakers overlap" in result.stdout
