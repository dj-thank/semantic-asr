"""Stored decision re-evaluation is distinct from model inference or training."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "research" / "phonetic-20260905"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "audit_public", ROOT / "scripts/audit_public_decisions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_file(tmp_path, *, mutate=None):
    items = json.loads((POLICIES / "decision-fixtures.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(items)
    raw = "\n".join(json.dumps(item, ensure_ascii=False) for item in items).encode("utf-8")
    path = tmp_path / "input.jsonl"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_replayed_audit_records_counts_and_does_not_claim_promotion(tmp_path):
    path, digest = input_file(tmp_path)
    report = load_script().audit(path, expected_sha256=digest, policies=POLICIES, iterations=100)
    assert report["decision_count"] == 8
    assert report["new_model_inference"] is False and report["new_weight_training"] is False
    assert report["promotion"] == "not-evaluated" and report["fresh_publication_test"] is False
    assert all(r["role"] == "exposed-regression" for r in report["reports"])
    for r in report["reports"]:
        assert r["samples"] == r["harmed"] + r["improved"] + r["tied"]
        assert r["corpus_comparison"]["aggregation"] == "corpus-error-rate"
        assert r["utterance_mean_comparison"]["aggregation"] == "utterance-mean"


def test_digest_mismatch_precedes_decision_or_evaluation(tmp_path):
    path, _ = input_file(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_script().audit(path, expected_sha256="0" * 64, policies=POLICIES, iterations=100)


def test_duplicate_samples_are_not_independent_evidence(tmp_path):
    path, digest = input_file(tmp_path, mutate=lambda items: items.append(items[0]))
    with pytest.raises(ValueError, match="duplicate"):
        load_script().audit(path, expected_sha256=digest, policies=POLICIES, iterations=100)


def test_changed_expected_output_is_a_replay_failure(tmp_path):
    def mutate(items):
        items[0]["expected_selected_id"] = "not-a-real-candidate"

    path, digest = input_file(tmp_path, mutate=mutate)
    with pytest.raises(ValueError, match="no longer reproduces"):
        load_script().audit(path, expected_sha256=digest, policies=POLICIES, iterations=100)
