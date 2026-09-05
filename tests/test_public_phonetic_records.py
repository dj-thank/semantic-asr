"""Audit the complete public error ledger and stored model-decision regressions."""

import importlib.util
import json
from pathlib import Path

from semantic_asr.candidate_pool import lenient_surface_key
from semantic_asr.evaluation import edit_distance

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research" / "phonetic-20260905"


def restore(reference, edits):
    pieces, position = [], 0
    for start, end, replacement in edits:
        assert position <= start <= end <= len(reference)
        pieces.extend((reference[position:start], replacement))
        position = end
    return "".join((*pieces, reference[position:]))


def test_public_error_records_are_complete_and_numerically_reconstructible():
    payload = json.loads((STUDY / "errors.json").read_text(encoding="utf-8"))
    assert len(payload["records"]) == 59
    assert len({(r["wave"], r["id"]) for r in payload["records"]}) == 59
    for row in payload["records"]:
        ref = lenient_surface_key(row["reference"])
        base = restore(row["reference"], row["baseline_edits"])
        selected = restore(row["reference"], row["selected_edits"])
        assert edit_distance(ref, lenient_surface_key(base)) == row["errors"][0]
        assert edit_distance(ref, lenient_surface_key(selected)) == row["errors"][1]
        assert any(row["errors"][:2])


def test_recorded_heldout_decisions_replay_without_acoustic_models():
    spec = importlib.util.spec_from_file_location(
        "replay_probe", ROOT / "scripts" / "replay_phonetic_decisions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.replay(STUDY) == 8
