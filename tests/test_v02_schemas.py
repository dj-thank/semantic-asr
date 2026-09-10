from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

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
    payload = json.loads(
        (ROOT / "examples" / "v02-experiment-manifest.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(payload, schema)


def _experiment_manifest_example() -> dict[str, object]:
    return json.loads(
        (ROOT / "examples" / "v02-experiment-manifest.json").read_text(encoding="utf-8")
    )


def test_experiment_manifest_schema_rejects_noncanonical_audio_digest() -> None:
    schema = _schema("v02-experiment-manifest.schema.json")
    payload = _experiment_manifest_example()
    records = payload["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["audioSha256"] = "AB" * 32
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)


def test_experiment_manifest_schema_rejects_unknown_split_role() -> None:
    schema = _schema("v02-experiment-manifest.schema.json")
    payload = _experiment_manifest_example()
    records = payload["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["split"] = "validation"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, schema)
