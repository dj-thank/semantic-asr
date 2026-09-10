import copy

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.frozen_consensus import select_frozen_consensus


def candidates():
    return {
        role: CandidateEvidence(
            candidate_id=role,
            text="元の文" if role == "whisper" else "候補の文。",
            source=role,
            rank=1,
            acoustic=-2.5 if role == "whisper" else None,
            metadata={
                "model": role,
                "modelRevision": str(index) * 40,
                "audioWindowSha256": "a" * 64,
                "startMs": 0,
                "sampleCount": 16000,
                "sampleRate": 16000,
                "language": "ja",
            },
        )
        for index, role in enumerate(("whisper", "qwen", "reazon", "parakeet"), 1)
    }


def test_selects_original_candidate_without_inventing_scores_or_mutating_inputs():
    rows = candidates()
    before = copy.deepcopy(rows)
    selected = select_frozen_consensus(rows)
    assert selected is rows["qwen"]
    assert selected.acoustic is None and selected.cross_model is None
    assert rows == before


def test_disagreement_keeps_whisper_and_its_scores():
    rows = candidates()
    rows["reazon"] = CandidateEvidence.from_dict({**rows["reazon"].as_dict(), "text": "別の文"})
    assert select_frozen_consensus(rows) is rows["whisper"]


def test_punctuation_equivalence_preserves_selected_original_text():
    rows = candidates()
    rows["reazon"] = CandidateEvidence.from_dict({**rows["reazon"].as_dict(), "text": "候補の文"})
    assert select_frozen_consensus(rows).text == "候補の文。"


def test_empty_normalized_agreement_cannot_replace_whisper():
    rows = candidates()
    for role in ("qwen", "reazon", "parakeet"):
        rows[role] = CandidateEvidence.from_dict({**rows[role].as_dict(), "text": "…。"})
    assert select_frozen_consensus(rows) is rows["whisper"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("audioWindowSha256", "b" * 64),
        ("startMs", 1),
        ("sampleCount", 8000),
        ("sampleRate", 8000),
        ("language", "en"),
    ],
)
def test_rejects_cross_audio_window_or_language(field, value):
    rows = candidates()
    rows["reazon"].metadata[field] = value
    with pytest.raises(ValueError):
        select_frozen_consensus(rows)


@pytest.mark.parametrize("field", ["modelRevision", "model", "audioWindowSha256", "sampleCount"])
def test_rejects_missing_provenance(field):
    rows = candidates()
    del rows["qwen"].metadata[field]
    with pytest.raises(ValueError):
        select_frozen_consensus(rows)


def test_rejects_same_model_as_independent_support():
    rows = candidates()
    rows["reazon"].metadata.update(model="qwen", modelRevision="2" * 40)
    with pytest.raises(ValueError):
        select_frozen_consensus(rows)


def test_rejects_extra_reference_channel():
    rows = candidates()
    rows["reference"] = rows["qwen"]
    with pytest.raises(ValueError):
        select_frozen_consensus(rows)


def test_rejects_reused_candidate_id():
    rows = candidates()
    rows["reazon"] = CandidateEvidence.from_dict(
        {**rows["reazon"].as_dict(), "candidate_id": "qwen"}
    )
    with pytest.raises(ValueError):
        select_frozen_consensus(rows)


def test_rejects_renamed_copy_of_same_artifact():
    rows = candidates()
    rows["reazon"].metadata["modelRevision"] = "2" * 40
    with pytest.raises(ValueError):
        select_frozen_consensus(rows)


def test_cli_keeps_full_inputs_and_requires_explicit_export(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from semantic_asr.contracts import sha256_json

    source = tmp_path / "input.json"
    source.write_text(
        json.dumps({r: c.as_dict() for r, c in candidates().items()}), encoding="utf-8"
    )
    out = tmp_path / "result"
    script = Path(__file__).resolve().parents[1] / "scripts/select_frozen_consensus.py"
    command = [sys.executable, str(script), str(source), "--output-dir", str(out)]
    denied = subprocess.run(command, capture_output=True, text=True)
    assert denied.returncode != 0 and not out.exists()
    success = subprocess.run(command + ["--allow-local-research"], capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    receipt = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    digest = receipt.pop("evidenceSha256")
    assert digest == sha256_json(receipt)
    assert receipt["decision"] == "provisional"
    assert receipt["candidates"] == json.loads(source.read_text(encoding="utf-8"))
    assert receipt["selected"]["text"] == "候補の文。"
    assert receipt["selected"]["cross_model"] is None
    duplicate = subprocess.run(command + ["--allow-local-research"], capture_output=True, text=True)
    assert duplicate.returncode != 0
