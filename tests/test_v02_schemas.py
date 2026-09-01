from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"


def _schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def test_typed_evidence_score_schema_requires_calibration_for_probability() -> None:
    schema = _schema("v02-evidence-score.schema.json")
    jsonschema.validate(
        {
            "value": 0.82,
            "semantics": "probability",
            "calibrated": True,
            "provenance": {
                "scorer": "compact-reranker",
                "calibrationDigest": "calibration-v1",
            },
        },
        schema,
    )
    try:
        jsonschema.validate(
            {
                "value": 0.82,
                "semantics": "probability",
                "calibrated": False,
                "provenance": {"scorer": "chat-self-report"},
            },
            schema,
        )
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("uncalibrated probability unexpectedly passed schema")


def test_ranking_example_matches_schema() -> None:
    schema = _schema("v02-ranking-group.schema.json")
    lines = (
        (ROOT / "examples" / "v02-ranking-groups.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert lines
    for line in lines:
        if line.strip():
            jsonschema.validate(json.loads(line), schema)


def test_experiment_manifest_schema() -> None:
    schema = _schema("v02-experiment-manifest.schema.json")
    jsonschema.validate(
        {
            "datasetName": "fixture",
            "datasetRevision": "1",
            "rightsRegistryDigest": None,
            "records": [
                {
                    "sampleId": "sample-1",
                    "split": "train",
                    "audioSha256": "a" * 64,
                    "reference": "今日は東京に行きます",
                    "speakerId": "speaker-1",
                    "sourceRecordingId": "recording-1",
                    "durationSeconds": 2.5,
                    "domain": "meeting",
                    "metadata": {},
                }
            ],
        },
        schema,
    )
