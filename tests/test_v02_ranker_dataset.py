from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from semantic_asr.cli_root import main
from semantic_asr.ranker_dataset import ranker_example_from_row


def _row(*, split: str = "train") -> dict[str, object]:
    return {
        "exampleId": f"example-{split}",
        "groupId": f"speaker-{split}",
        "split": split,
        "reference": "料金は3000円です",
        "candidates": [
            {
                "candidateId": "wrong",
                "text": "料金は30000円です",
                "acoustic": 0.60,
                "mora": 0.60,
                "rank": 1,
                "hypothesisCount": 2,
            },
            {
                "candidateId": "correct",
                "text": "料金は3000円です",
                "acoustic": 0.59,
                "mora": 0.59,
                "rank": 2,
                "hypothesisCount": 2,
            },
        ],
    }


def test_reference_manifest_derives_semantic_losses() -> None:
    example = ranker_example_from_row(_row(), line_number=1)
    assert example.losses["correct"] == pytest.approx(0.0)
    assert example.losses["wrong"] > example.losses["correct"]


def test_ranker_dataset_rejects_calibration_and_test_rows() -> None:
    with pytest.raises(ValueError, match="forbidden split"):
        ranker_example_from_row(_row(split="calibration"), line_number=1)
    with pytest.raises(ValueError, match="forbidden split"):
        ranker_example_from_row(_row(split="test"), line_number=1)


def test_pairwise_and_listwise_cli_accept_reference_only_train_manifest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "train.jsonl"
        rows = []
        for index in range(8):
            row = _row()
            row["exampleId"] = f"example-{index}"
            row["groupId"] = f"speaker-{index % 4}"
            rows.append(row)
        source.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        pairwise = root / "pairwise.json"
        listwise = root / "listwise.json"
        assert (
            main(
                [
                    "train-ranker",
                    str(source),
                    "--output",
                    str(pairwise),
                    "--epochs",
                    "40",
                ]
            )
            == 0
        )
        assert (
            main(
                [
                    "train-listwise-ranker",
                    str(source),
                    "--output",
                    str(listwise),
                    "--epochs",
                    "60",
                ]
            )
            == 0
        )
        assert json.loads(pairwise.read_text(encoding="utf-8"))["profile"]["name"]
        assert (
            json.loads(listwise.read_text(encoding="utf-8"))["schemaVersion"]
            == "listwise-semantic-mwer-v1"
        )


def test_load_jsonl_examples_skips_single_candidate_rows(tmp_path) -> None:
    import json

    from semantic_asr.ranker_training import load_jsonl_examples

    rows = [
        {
            "exampleId": "single",
            "split": "train",
            "reference": "はい",
            "candidates": [{"candidate_id": "a", "text": "はい", "acoustic": -0.1}],
        },
        {
            "exampleId": "pair",
            "split": "train",
            "reference": "はい",
            "candidates": [
                {"candidate_id": "a", "text": "はい", "acoustic": -0.1},
                {"candidate_id": "b", "text": "いいえ", "acoustic": -0.3},
            ],
        },
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("
".join(json.dumps(row, ensure_ascii=False) for row in rows), "utf-8")
    examples = load_jsonl_examples(path)
    assert [example.example_id for example in examples] == ["pair"]
