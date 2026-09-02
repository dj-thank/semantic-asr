from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_asr.benchmark import load_benchmark_jsonl, run_benchmark
from semantic_asr.cli_root import main


def _candidate(candidate_id: str, text: str, rank: int) -> dict[str, object]:
    return {
        "candidateId": candidate_id,
        "text": text,
        "acoustic": 0.8 if rank == 1 else 0.7,
        "mora": 0.8 if rank == 1 else 0.7,
        "rank": rank,
        "hypothesisCount": 2,
    }


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, index in (("train", 1), ("calibration", 2), ("test", 3)):
        rows.append(
            {
                "sampleId": f"sample-{split}",
                "groupId": f"speaker-{split}",
                "sourceId": f"source-{split}",
                "split": split,
                "reference": "料金は3000円です",
                "candidates": [
                    _candidate("wrong", "料金は30000円です", 1),
                    _candidate("correct", "料金は3000円です", 2),
                ],
                "domain": f"domain-{index}",
            }
        )
    return rows


def _ranker_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "profile": {
                    "name": "fixture-ranker",
                    "weights": {
                        "acoustic": 0.0,
                        "mora": 0.0,
                        "lexical": 0.0,
                        "preservation": 0.0,
                        "cross_model": 0.0,
                        "rank_fraction": -2.0,
                        "logprob": 0.0,
                        "length": 0.0,
                        "context_overlap": 0.0,
                        "critical_count": -2.0,
                        "source_diversity": 0.0,
                    },
                    "bias": 0.0,
                    "feature_mean": {},
                    "feature_scale": {},
                    "training_manifest_sha256": "a" * 64,
                    "version": "fixture",
                }
            }
        ),
        encoding="utf-8",
    )


def test_experiment_cli_partition_score_calibrate_and_apply() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "all.jsonl"
        manifest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in _manifest_rows()) + "\n",
            encoding="utf-8",
        )
        split_dir = root / "split"
        assert (
            main(
                [
                    "partition-manifest",
                    str(manifest),
                    "--output-dir",
                    str(split_dir),
                ]
            )
            == 0
        )
        partition = json.loads((split_dir / "partition.json").read_text(encoding="utf-8"))
        assert partition["counts"] == {"train": 1, "calibration": 1, "test": 1}

        ranker = root / "ranker.json"
        _ranker_profile(ranker)
        calibration_samples = root / "calibration-samples.jsonl"
        assert (
            main(
                [
                    "score-ranker-calibration",
                    str(split_dir / "calibration.jsonl"),
                    "--ranker-profile",
                    str(ranker),
                    "--output",
                    str(calibration_samples),
                ]
            )
            == 0
        )
        sample_rows = [
            json.loads(line)
            for line in calibration_samples.read_text(encoding="utf-8").splitlines()
        ]
        assert len(sample_rows) == 2
        assert {row["correct"] for row in sample_rows} == {False, True}

        calibration_rows = []
        for group_index in range(4):
            calibration_rows.extend(
                [
                    {
                        "sampleId": f"g{group_index}-wrong",
                        "groupId": f"g{group_index}",
                        "score": -2.0,
                        "correct": False,
                        "split": "calibration",
                    },
                    {
                        "sampleId": f"g{group_index}-correct",
                        "groupId": f"g{group_index}",
                        "score": 2.0,
                        "correct": True,
                        "split": "calibration",
                    },
                ]
            )
        calibration_fit_input = root / "calibration-fit.jsonl"
        calibration_fit_input.write_text(
            "\n".join(json.dumps(row) for row in calibration_rows) + "\n",
            encoding="utf-8",
        )
        calibration_profile = root / "calibration.json"
        assert (
            main(
                [
                    "calibrate-ranker",
                    str(calibration_fit_input),
                    "--output",
                    str(calibration_profile),
                    "--source-ranker",
                    "fixture-ranker",
                ]
            )
            == 0
        )

        reranked = root / "reranked-test.jsonl"
        assert (
            main(
                [
                    "apply-ranker",
                    str(split_dir / "test.jsonl"),
                    "--ranker-profile",
                    str(ranker),
                    "--calibration",
                    str(calibration_profile),
                    "--output",
                    str(reranked),
                ]
            )
            == 0
        )
        row = json.loads(reranked.read_text(encoding="utf-8"))
        by_id = {candidate["candidate_id"]: candidate for candidate in row["candidates"]}
        assert by_id["wrong"]["rank"] == 1
        assert by_id["correct"]["rank"] == 2
        assert by_id["correct"]["metadata"]["offlineRerankerRank"] == 1
        assert by_id["correct"]["metadata"]["offlineRerankerEvidenceInjected"] is True


def test_partition_and_rerank_round_trip_preserves_unsafe_annotation_metadata() -> None:
    rows = _manifest_rows()
    annotations = {
        "sample-train": "(F えー)料金は3000円です",
        "sample-calibration": "料金は(? 3000円)です",
        "sample-test": "[PERSON_01]",
        "sample-test-masked": "[MASK]",
    }
    for row in rows:
        row["annotatedReference"] = annotations[str(row["sampleId"])]
    rows.append(
        {
            "sampleId": "sample-test-masked",
            "groupId": "speaker-test-masked",
            "sourceId": "source-test-masked",
            "split": "test",
            "reference": "匿名の発話",
            "annotatedReference": annotations["sample-test-masked"],
            "candidates": [
                _candidate("wrong", "別の発話", 1),
                _candidate("correct", "匿名の発話", 2),
            ],
            "domain": "domain-masked",
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "annotated.jsonl"
        manifest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        split_dir = root / "split"
        assert (
            main(
                [
                    "partition-manifest",
                    str(manifest),
                    "--output-dir",
                    str(split_dir),
                ]
            )
            == 0
        )

        for split in ("train", "calibration", "test"):
            split_rows = [
                json.loads(line)
                for line in (split_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            assert {
                row["sampleId"]: row["annotatedReference"] for row in split_rows
            } == {
                sample_id: annotation
                for sample_id, annotation in annotations.items()
                if sample_id.startswith(f"sample-{split}")
            }

        ranker = root / "ranker.json"
        _ranker_profile(ranker)
        reranked = root / "reranked-test.jsonl"
        assert (
            main(
                [
                    "apply-ranker",
                    str(split_dir / "test.jsonl"),
                    "--ranker-profile",
                    str(ranker),
                    "--output",
                    str(reranked),
                ]
            )
            == 0
        )
        reranked_rows = [
            json.loads(line) for line in reranked.read_text(encoding="utf-8").splitlines()
        ]
        assert {
            row["sampleId"]: row["annotatedReference"] for row in reranked_rows
        } == {
            "sample-test": annotations["sample-test"],
            "sample-test-masked": annotations["sample-test-masked"],
        }

        report = run_benchmark(load_benchmark_jsonl(reranked), ks=(1, 2), bootstrap_iterations=2)
        assert {
            row.sample_id: row.annotated_reference for row in report.rows
        } == {
            "sample-test": annotations["sample-test"],
            "sample-test-masked": annotations["sample-test-masked"],
        }
        assert all(row.baseline_cer is None for row in report.rows)
        assert all(row.cascade_cer is None for row in report.rows)
        assert all(row.mbr_cer is None for row in report.rows)
        assert report.oracle_cer_at_k == {1: None, 2: None}
        assert report.cascade_improvement is None
