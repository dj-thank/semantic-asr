from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jsonschema

from semantic_asr.adapters import DecodeRequest
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.longform import SemanticASRTranscriber
from semantic_asr.outputs import write_outputs

ROOT = Path(__file__).resolve().parents[1]


class SchemaFakeAdapter:
    name = "schema-fake"
    model_name = "fixture"
    # Explicitly marks this in-memory fixture as safe for the legacy cache identity.
    allow_legacy_cache_identity = True

    def decode(self, request: DecodeRequest):
        return [
            CandidateEvidence(
                "a",
                "今日は晴れです。",
                acoustic=0.9,
                mora=0.9,
                lexical=0.8,
                preservation=0.9,
                cross_model=0.8,
                source=self.name,
            ),
            CandidateEvidence(
                "b",
                "今日は雨です。",
                acoustic=0.1,
                mora=0.1,
                lexical=0.2,
                preservation=0.2,
                cross_model=0.1,
                source=self.name,
            ),
        ]


def validate(instance_path: Path, schema_path: Path) -> None:
    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(instance)


def test_candidate_and_rights_examples_validate() -> None:
    validate(
        ROOT / "examples/candidates.json",
        ROOT / "schemas/candidate-manifest.schema.json",
    )
    validate(
        ROOT / "data/rights_registry.example.json",
        ROOT / "schemas/rights-registry.schema.json",
    )


def test_generated_transcript_validates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "fixture.wav"
        audio.write_bytes(b"fixture")
        result = SemanticASRTranscriber(SchemaFakeAdapter()).transcribe(audio, duration_ms=1_000)
        outputs = write_outputs(result, root / "out")
        validate(
            Path(outputs["json"]),
            ROOT / "schemas/transcript-result.schema.json",
        )
