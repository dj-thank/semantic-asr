from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from semantic_asr.cli import main


def test_demo_and_fuse_cli() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        demo = root / "demo.json"
        assert main(["demo", "--output", str(demo)]) == 0
        payload = json.loads(demo.read_text(encoding="utf-8"))
        assert payload["selected"] == "昨日学校を行きました"

        manifest = root / "candidates.json"
        manifest.write_text(
            json.dumps(
                {
                    "candidates": [
                        {"candidateId": "a", "text": "三人です", "acoustic": 0.51, "mora": 0.49},
                        {"candidateId": "b", "text": "二人です", "acoustic": 0.50, "mora": 0.51},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        fused = root / "fused.json"
        assert main(["fuse", str(manifest), "--output", str(fused)]) == 0
        fused_payload = json.loads(fused.read_text(encoding="utf-8"))
        assert fused_payload["observedTranscript"] in {"三人です", "二人です"}
        assert fused_payload["evidencePlan"]["selected"]


def test_calibration_cli() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "calibration.jsonl"
        rows = [
            {"confidence": confidence, "correct": correct}
            for confidence, correct in zip(
                [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.45, 0.35, 0.2, 0.1],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                strict=True,
            )
        ]
        source.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        target = root / "profile.json"
        assert main(["calibrate", str(source), "--output", str(target)]) == 0
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["sampleCount"] == 10
        assert payload["profile"]["temperature"] > 0
        assert len(payload["profile"]["digest"]) == 64


def test_rights_cli_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        registry = Path(directory) / "rights.json"
        registry.write_text(
            json.dumps(
                {
                    "assets": [
                        {
                            "assetId": "fixture",
                            "sourceName": "Fixture",
                            "sourceUrl": "https://example.invalid",
                            "version": "1",
                            "licenseName": "Fixture",
                            "licenseUrl": "https://example.invalid/license",
                            "train": "allow",
                            "deriveFeatures": "allow",
                            "redistributeRaw": "deny",
                            "exportSpeakerId": "deny",
                            "attribution": "Fixture",
                            "reviewedAt": "2026-08-29",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert main(["rights", str(registry), "fixture", "train"]) == 0
        with pytest.raises(PermissionError):
            main(["rights", str(registry), "fixture", "redistribute_raw"])
