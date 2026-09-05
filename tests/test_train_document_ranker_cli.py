from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from semantic_asr.document_ranker import DocumentRankerArtifact


def rows(prefix: str):
    return (
        {
            "groupId": f"{prefix}-1",
            "candidateId": "good",
            "text": "この変更はまだマージしません。",
            "leftContext": "レビュー中です。",
            "rightContext": "承認後に統合します。",
            "localScore": 0.35,
            "meanAudioSupport": 0.68,
            "changedWindowCount": 1,
            "windowCount": 2,
            "retainedPath": False,
            "characterErrorRate": 0.0,
            "criticalErrorCount": 0,
        },
        {
            "groupId": f"{prefix}-1",
            "candidateId": "bad",
            "text": "この変更はまたマージしません。",
            "leftContext": "レビュー中です。",
            "rightContext": "承認後に統合します。",
            "localScore": 0.40,
            "meanAudioSupport": 0.70,
            "changedWindowCount": 0,
            "windowCount": 2,
            "retainedPath": True,
            "characterErrorRate": 0.15,
            "criticalErrorCount": 1,
        },
        {
            "groupId": f"{prefix}-2",
            "candidateId": "good",
            "text": "三千円です。",
            "leftContext": "費用を確認します。",
            "rightContext": "予算内です。",
            "localScore": 0.50,
            "meanAudioSupport": 0.80,
            "changedWindowCount": 0,
            "windowCount": 1,
            "retainedPath": True,
            "characterErrorRate": 0.0,
            "criticalErrorCount": 0,
            "firstPassExact": True,
        },
        {
            "groupId": f"{prefix}-2",
            "candidateId": "bad",
            "text": "三万円です。",
            "leftContext": "費用を確認します。",
            "rightContext": "予算内です。",
            "localScore": 0.48,
            "meanAudioSupport": 0.78,
            "changedWindowCount": 1,
            "windowCount": 1,
            "retainedPath": False,
            "characterErrorRate": 0.20,
            "criticalErrorCount": 1,
            "firstPassExact": True,
        },
    )


def write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def test_cli_trains_calibrates_tests_and_writes_digest_checked_artifact(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    artifact = tmp_path / "ranker.json"
    report = tmp_path / "report.json"
    write_jsonl(train, rows("train"))
    write_jsonl(calibration, rows("cal"))
    write_jsonl(test, rows("test"))

    subprocess.run(
        [
            sys.executable,
            "scripts/train_document_ranker.py",
            "--train",
            str(train),
            "--calibration",
            str(calibration),
            "--test",
            str(test),
            "--output",
            str(artifact),
            "--report",
            str(report),
            "--revision",
            "test-ranker-r1",
            "--epochs",
            "20",
            "--hash-dimension",
            "2048",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    loaded = DocumentRankerArtifact.load(artifact)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert loaded.digest == payload["artifactDigest"]
    assert payload["trainGroups"] == 2
    assert payload["calibrationGroups"] == 2
    assert payload["testGroups"] == 2
    assert 0.0 <= payload["testPairwiseAccuracy"] <= 1.0


def test_cli_rejects_group_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, rows("same"))
    write_jsonl(calibration, rows("same"))
    write_jsonl(test, rows("test"))

    process = subprocess.run(
        [
            sys.executable,
            "scripts/train_document_ranker.py",
            "--train",
            str(train),
            "--calibration",
            str(calibration),
            "--test",
            str(test),
            "--output",
            str(tmp_path / "ranker.json"),
            "--report",
            str(tmp_path / "report.json"),
            "--revision",
            "bad",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert process.returncode != 0
    assert "group leakage" in process.stderr
