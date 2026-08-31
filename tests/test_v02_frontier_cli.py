from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_asr.cli_root import main


def test_frontier_train_ngram_and_throttle_policy(capsys) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.txt"
        corpus.write_text(
            "料金は3000円です\n料金は3000円です\n料金は30000円です\n",
            encoding="utf-8",
        )
        model = root / "ngram.json"
        assert (
            main(
                [
                    "train-ngram",
                    str(corpus),
                    "--output",
                    str(model),
                    "--mode",
                    "character",
                    "--order",
                    "4",
                ]
            )
            == 0
        )
        payload = json.loads(model.read_text(encoding="utf-8"))
        assert payload["schemaVersion"] == "ngram-v1"
        assert payload["documentCount"] == 3
        assert main(["throttle-policy", "--effort", "edge-gpu"]) == 0
        stdout = capsys.readouterr().out
        assert '"source_profile": "edge-gpu"' in stdout


def test_frontier_listwise_and_fusion_training_write_profiles() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        ranker_data = root / "ranker.jsonl"
        rows = []
        for index in range(6):
            rows.append(
                {
                    "exampleId": f"example-{index}",
                    "candidates": [
                        {
                            "candidateId": "wrong",
                            "text": "料金は30000円です",
                            "acoustic": 0.20,
                            "mora": 0.20,
                            "lexical": 0.80,
                            "rank": 1,
                            "hypothesisCount": 2,
                        },
                        {
                            "candidateId": "correct",
                            "text": "料金は3000円です",
                            "acoustic": 0.90,
                            "mora": 0.90,
                            "lexical": 0.20,
                            "rank": 2,
                            "hypothesisCount": 2,
                        },
                    ],
                    "losses": {"wrong": 1.0, "correct": 0.0},
                }
            )
        ranker_data.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        ranker_output = root / "listwise.json"
        assert (
            main(
                [
                    "train-listwise-ranker",
                    str(ranker_data),
                    "--output",
                    str(ranker_output),
                    "--epochs",
                    "80",
                ]
            )
            == 0
        )
        ranker_profile = json.loads(ranker_output.read_text(encoding="utf-8"))
        assert ranker_profile["schemaVersion"] == "listwise-semantic-mwer-v1"
        assert ranker_profile["after"]["mean_expected_loss"] < ranker_profile["before"][
            "mean_expected_loss"
        ]

        fusion_data = root / "fusion.jsonl"
        fusion_rows = []
        for index in range(10):
            fusion_rows.append(
                {
                    "exampleId": f"fusion-{index}",
                    "groupId": f"speaker-{index % 3}",
                    "split": "train",
                    "candidates": [
                        {
                            "candidateId": "correct",
                            "text": "実際の発話",
                            "acoustic": 0.92,
                            "mora": 0.90,
                            "lexical": 0.20,
                            "preservation": 0.85,
                            "crossModel": 0.82,
                        },
                        {
                            "candidateId": "fluent",
                            "text": "自然な捏造",
                            "acoustic": 0.18,
                            "mora": 0.20,
                            "lexical": 0.98,
                            "preservation": 0.30,
                            "crossModel": 0.22,
                        },
                    ],
                    "targetDistribution": {"correct": 1.0, "fluent": 0.0},
                }
            )
        fusion_data.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in fusion_rows) + "\n",
            encoding="utf-8",
        )
        fusion_output = root / "fusion.json"
        assert (
            main(
                [
                    "train-fusion",
                    str(fusion_data),
                    "--output",
                    str(fusion_output),
                    "--epochs",
                    "80",
                ]
            )
            == 0
        )
        fusion_profile = json.loads(fusion_output.read_text(encoding="utf-8"))
        assert fusion_profile["schemaVersion"] == "learned-fusion-v1"
        weights = fusion_profile["profile"]["weights"]
        acoustic_family = weights["acoustic"] + weights["mora"] + weights["cross_model"]
        assert acoustic_family >= 0.72


def test_frontier_deployment_gate_returns_nonzero_for_regression() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        common_artifact = {
            "source_model": "fixture/reranker",
            "source_revision": "revision",
            "tokenizer_sha256": "c" * 64,
            "runtime": "fixture",
            "runtime_version": "1",
            "calibration_digest": "d" * 64,
            "build_manifest_sha256": "e" * 64,
            "quantization": {},
        }
        common_evaluation = {
            "test_manifest_sha256": "f" * 64,
            "sample_count": 100,
            "group_count": 10,
            "runtime_hardware": "fixture-cpu",
            "repeated_runs": 2,
        }
        baseline = {
            "artifact": {
                **common_artifact,
                "name": "baseline",
                "artifact_format": "pytorch",
                "artifact_sha256": "a" * 64,
            },
            "metrics": {
                "candidate_top1_accuracy": 0.90,
                "pairwise_accuracy": 0.94,
                "semantic_loss": 0.10,
                "critical_error_rate": 0.02,
                "calibration_error": 0.03,
                "aurc": 0.07,
                "real_time_factor": 0.20,
                "peak_memory_mb": 1000.0,
                "artifact_size_mb": 1200.0,
                "deterministic_replay_rate": 1.0,
            },
            **common_evaluation,
        }
        candidate = {
            "artifact": {
                **common_artifact,
                "name": "candidate",
                "artifact_format": "torchao-int4",
                "artifact_sha256": "b" * 64,
            },
            "metrics": {
                "candidate_top1_accuracy": 0.90,
                "pairwise_accuracy": 0.94,
                "semantic_loss": 0.10,
                "critical_error_rate": 0.03,
                "calibration_error": 0.03,
                "aurc": 0.07,
                "real_time_factor": 0.10,
                "peak_memory_mb": 400.0,
                "artifact_size_mb": 350.0,
                "deterministic_replay_rate": 1.0,
            },
            **common_evaluation,
        }
        baseline_path = root / "baseline.json"
        candidate_path = root / "candidate.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        assert main(["deployment-gate", str(baseline_path), str(candidate_path)]) == 2
