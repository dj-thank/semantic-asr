from __future__ import annotations

import json

import pytest

from semantic_asr.cli_root import main


def test_research_ledger_cli_routes_and_reports_stable_digest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["research-ledger", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schemaVersion"] == "semantic-asr-research-ledger-v1"
    assert len(payload["digest"]) == 64
    assert {row["source_id"] for row in payload["sources"]} >= {
        "qwen3.8-flash-next",
        "qwen3-asr",
        "adaptive-ger",
        "mbr-for-asr",
    }
    assert all(row["status"] != "rejected" for row in payload["sources"])


def test_research_ledger_cli_writes_markdown(tmp_path) -> None:
    output = tmp_path / "research-ledger.md"
    assert main(["research-ledger", "--format", "markdown", "--output", str(output)]) == 0
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Semantic ASR research ledger")
    assert "qwen3.8-flash-next" in text
    assert "Claim boundary" in text
