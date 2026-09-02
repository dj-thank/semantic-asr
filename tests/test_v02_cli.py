from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from semantic_asr.cli_v2 import build_advanced_parser, main


def test_ranker_calibration_cli_writes_runtime_profile() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "calibration.jsonl"
        rows = []
        for group in range(4):
            rows.extend(
                [
                    {
                        "sampleId": f"g{group}-negative",
                        "groupId": f"speaker-{group}",
                        "score": 0.2,
                        "correct": False,
                        "split": "calibration",
                    },
                    {
                        "sampleId": f"g{group}-positive",
                        "groupId": f"speaker-{group}",
                        "score": 1.8,
                        "correct": True,
                        "split": "calibration",
                    },
                ]
            )
        source.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        target = root / "profile.json"
        assert (
            main(
                [
                    "calibrate-ranker",
                    str(source),
                    "--output",
                    str(target),
                    "--source-ranker",
                    "fixture-ranker",
                ]
            )
            == 0
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["schemaVersion"] == "ranker-calibration-v1"
        assert payload["profile"]["sample_count"] == 8
        assert payload["profile"]["group_count"] == 4
        assert (
            payload["after"]["negative_log_likelihood"]
            < payload["before"]["negative_log_likelihood"]
        )


def test_benchmark_cli_writes_group_bootstrap_report() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "benchmark.jsonl"
        rows = []
        for index, (reference, wrong) in enumerate(
            [
                ("料金は3000円です", "料金は30000円です"),
                ("明日は行きません", "明日は行きます"),
            ],
            1,
        ):
            rows.append(
                {
                    "sampleId": f"sample-{index}",
                    "groupId": f"speaker-{index}",
                    "sourceId": f"source-{index}",
                    "split": "test",
                    "reference": reference,
                    "candidates": [
                        {
                            "candidateId": "wrong",
                            "text": wrong,
                            "acoustic": 0.6,
                            "mora": 0.6,
                            "rank": 1,
                            "hypothesisCount": 2,
                        },
                        {
                            "candidateId": "correct",
                            "text": reference,
                            "acoustic": 0.59,
                            "mora": 0.59,
                            "rank": 2,
                            "hypothesisCount": 2,
                        },
                    ],
                }
            )
        source.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        target = root / "report.json"
        assert (
            main(
                [
                    "benchmark",
                    str(source),
                    "--output",
                    str(target),
                    "--ks",
                    "1,2",
                    "--bootstrap-iterations",
                    "20",
                ]
            )
            == 0
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["sample_count"] == 2
        assert payload["group_count"] == 2
        assert float(payload["oracle_cer_at_k"]["2"]) <= float(payload["oracle_cer_at_k"]["1"])


def test_benchmark_cli_serializes_undefined_bootstrap_for_unscorable_annotations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "benchmark-uncertain.jsonl"
        source.write_text(
            json.dumps(
                {
                    "sampleId": "uncertain",
                    "groupId": "speaker",
                    "sourceId": "source",
                    "split": "test",
                    "reference": "明日は舞い上がる",
                    "annotatedReference": "明日は(? 舞い)上がる",
                    "candidates": [
                        {
                            "candidateId": "candidate",
                            "text": "明日は舞い上がる",
                            "rank": 1,
                            "hypothesisCount": 1,
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        target = root / "report.json"
        assert (
            main(
                [
                    "benchmark",
                    str(source),
                    "--output",
                    str(target),
                    "--ks",
                    "1",
                    "--bootstrap-iterations",
                    "2",
                ]
            )
            == 0
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["rows"][0]["baseline_cer"] is None
        assert payload["cascade_improvement"] is None
        stdout_payload = json.loads(capsys.readouterr().out)
        assert stdout_payload["cascadeImprovement"] is None


def test_teacher_distillation_cli_preserves_candidate_set() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "teachers.jsonl"
        source.write_text(
            json.dumps(
                {
                    "exampleId": "example",
                    "candidates": [
                        {"candidateId": "a", "text": "料金は3000円です"},
                        {"candidateId": "b", "text": "料金は30000円です"},
                    ],
                    "judgments": [
                        {
                            "teacher": "teacher-a",
                            "scoreKind": "logit",
                            "scores": {"a": 2.0, "b": -1.0},
                        },
                        {
                            "teacher": "teacher-b",
                            "scoreKind": "preference",
                            "scores": {"a": 0.8, "b": 0.2},
                        },
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        target = root / "distilled.jsonl"
        rejected = root / "rejected.jsonl"
        assert (
            main(
                [
                    "distill-teachers",
                    str(source),
                    "--output",
                    str(target),
                    "--rejected-output",
                    str(rejected),
                ]
            )
            == 0
        )
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert {candidate["candidate_id"] for candidate in payload["candidates"]} == {
            "a",
            "b",
        }
        assert payload["losses"]["a"] < payload["losses"]["b"]
        assert rejected.read_text(encoding="utf-8") == ""


def test_advanced_cli_discovers_effort_profiles() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["transcribe-v2", "--help"])
    assert exc_info.value.code == 0


def test_advanced_transcribe_cli_carries_model_provenance_options() -> None:
    args = build_advanced_parser().parse_args(
        [
            "transcribe-v2",
            "audio.wav",
            "--model-revision",
            "1" * 40,
            "--runtime-revision",
            "runtime-r2",
            "--qwen-model-revision",
            "2" * 40,
            "--qwen-aligner-revision",
            "3" * 40,
        ]
    )
    assert args.model_revision == "1" * 40
    assert args.runtime_revision == "runtime-r2"
    assert args.qwen_model_revision == "2" * 40
    assert args.qwen_aligner_revision == "3" * 40
