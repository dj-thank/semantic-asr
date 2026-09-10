"""Fail-closed receipt accounting and stage integrity; synthetic data only."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cycle():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        yield importlib.import_module("research_cycle")
    finally:
        sys.path.remove(scripts)


def arguments(source, output, *extra):
    return [
        "--candidates",
        str(source),
        "--output-dir",
        str(output),
        "--allow-raw-export",
        "--ranker",
        "pairwise",
        "--epochs",
        "1",
        "--max-trials",
        "1",
        "--audio-seconds",
        "1",
        "--max-audio-seconds",
        "60",
        "--max-wall-seconds",
        "60",
        "--max-storage-bytes",
        "10485760",
        "--bootstrap-iterations",
        "100",
        "--evaluation-role",
        "synthetic",
        *extra,
    ]


@pytest.fixture(scope="module")
def completed_case(cycle, tmp_path_factory):
    from semantic_asr.contracts import CandidateEvidence

    directory = tmp_path_factory.mktemp("receipt-integrity")
    source = directory / "synthetic.jsonl"
    rows = []
    for split in ("train", "calibration", "test"):
        for index in range(4):
            sample = f"{split}-{index}"
            text = f"{split}の合成例{index}です"
            candidates = [
                CandidateEvidence("a", text, acoustic=-0.1, avg_logprob=-0.1, rank=1),
                CandidateEvidence("b", text + "ない", acoustic=-0.8, avg_logprob=-0.8, rank=2),
            ]
            rows.append(
                {
                    "sampleId": sample,
                    "sourceId": sample,
                    "groupId": sample,
                    "split": split,
                    "reference": text,
                    "rightsDecision": "allow",
                    "licenseId": "synthetic-test-only",
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
            )
    cycle.write_rows(source, rows)
    output = directory / "completed"
    assert cycle.main(arguments(source, output)) == 0
    return source, output


def snapshot(directory):
    return {
        str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("spent_seconds", -1),
        ("spent_seconds", True),
        ("spent_seconds", "0"),
        ("spent_seconds", None),
        ("spent_seconds", float("nan")),
        ("spent_seconds", float("inf")),
        ("spent_seconds", float("-inf")),
        ("trials_started", -1),
        ("trials_started", True),
        ("trials_started", 0.5),
        ("trials_started", "1"),
        ("trials_started", None),
        ("trials_started", 2),
        ("trials_started", 0),
        ("schema", "unrecognized-cycle-v2"),
        ("status", "unrecognized"),
    ],
)
def test_invalid_resume_accounting_is_rejected_without_writes(
    cycle, completed_case, tmp_path, field, value
):
    source, completed = completed_case
    output = tmp_path / "resume"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    receipt[field] = value
    # Intentionally encode non-standard NaN/Infinity to test untrusted input.
    (output / "cycle.json").write_text(json.dumps(receipt), encoding="utf-8")
    before = snapshot(output)
    assert cycle.main(arguments(source, output, "--resume")) == 2
    assert snapshot(output) == before


@pytest.mark.parametrize("value", [None, True, -1, "yesterday", float("nan"), float("inf")])
def test_running_receipt_requires_finite_start_time(cycle, completed_case, tmp_path, value):
    source, completed = completed_case
    output = tmp_path / "resume"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    receipt.update(status="running", active_started_at=value)
    (output / "cycle.json").write_text(json.dumps(receipt), encoding="utf-8")
    before = snapshot(output)
    assert cycle.main(arguments(source, output, "--resume")) == 2
    assert snapshot(output) == before


def test_unaccounted_training_attempt_is_rejected(cycle, completed_case, tmp_path):
    source, completed = completed_case
    output = tmp_path / "resume"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    receipt["attempts"].append({"stage": "fit-train", "status": "failed"})
    cycle.write_json(output / "cycle.json", receipt)
    before = snapshot(output)
    assert cycle.main(arguments(source, output, "--resume")) == 2
    assert snapshot(output) == before


def test_running_future_start_is_rejected(cycle, completed_case, tmp_path):
    source, completed = completed_case
    output = tmp_path / "resume"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    receipt.update(status="running", active_started_at=time.time() + 3600)
    cycle.write_json(output / "cycle.json", receipt)
    assert cycle.main(arguments(source, output, "--resume")) == 2


@pytest.mark.parametrize("artifact", ["ranker.json", "test-reference.jsonl"])
def test_last_stage_cannot_mutate_completed_artifact(
    cycle, completed_case, tmp_path, monkeypatch, artifact
):
    source, completed = completed_case
    output = tmp_path / "partial"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    # Resume a valid prefix immediately before report; all earlier stages are real.
    receipt.update(status="partial", completed=list(cycle.STAGES[:-1]))
    receipt["attempts"] = receipt["attempts"][:-1]
    receipt["stages"] = receipt["stages"][:-1]
    for name in cycle.OUTPUTS["report"]:
        receipt["artifacts"].pop(name)
        (output / name).unlink()
    cycle.write_json(output / "cycle.json", receipt)
    expected = receipt["artifacts"][artifact]
    original = cycle.run_stage

    def mutate_after_report(*args, **kwargs):
        original(*args, **kwargs)
        with (output / artifact).open("a", encoding="utf-8") as handle:
            handle.write(" ")

    monkeypatch.setattr(cycle, "run_stage", mutate_after_report)
    assert cycle.main(arguments(source, output, "--resume")) == 1
    after = cycle.read_json(output / "cycle.json")
    assert after["status"] == "failed"
    assert after["completed"] == list(cycle.STAGES[:-1])
    assert after["artifacts"][artifact] == expected
    assert after["attempts"][-1]["status"] == "failed"
    assert "completed artifact changed" in after["reason"]


def report_prefix(cycle, completed_case, tmp_path):
    """An actual completed synthetic run, rewound only in its test copy."""
    source, completed = completed_case
    output = tmp_path / "prefix"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    receipt.update(status="partial", completed=list(cycle.STAGES[:-1]))
    receipt["attempts"] = receipt["attempts"][:-1]
    receipt["stages"] = receipt["stages"][:-1]
    for name in cycle.OUTPUTS["report"]:
        receipt["artifacts"].pop(name)
        (output / name).unlink()
    cycle.write_json(output / "cycle.json", receipt)
    return source, output, receipt


def test_partial_artifact_validation_is_atomic_and_recoverable(
    cycle, completed_case, tmp_path, monkeypatch
):
    source, output, before = report_prefix(cycle, completed_case, tmp_path)
    original = cycle.run_stage

    def incomplete(*args, **kwargs):
        original(*args, **kwargs)
        (output / "report-raw.json").unlink()

    monkeypatch.setattr(cycle, "run_stage", incomplete)
    assert cycle.main(arguments(source, output, "--resume")) == 1
    failed = cycle.read_json(output / "cycle.json")
    assert failed["artifacts"] == before["artifacts"]
    assert failed["completed"] == before["completed"]
    abandoned = (output / "report.json").read_bytes()
    monkeypatch.setattr(cycle, "run_stage", original)
    assert cycle.main(arguments(source, output, "--resume")) == 0
    after = cycle.read_json(output / "cycle.json")
    assert after["trials_started"] == 1
    assert after["attempts"][-2]["status"] == "failed"
    assert after["attempts"][-1]["status"] == "completed"
    assert any(
        p.read_bytes() == abandoned for p in (output / "attempt-artifacts").rglob("report.json")
    )


def test_successful_noop_cannot_reuse_abandoned_stage_outputs(
    cycle, completed_case, tmp_path, monkeypatch
):
    source, output, before = report_prefix(cycle, completed_case, tmp_path)
    abandoned = b'{"stale": true}\n'
    for name in cycle.OUTPUTS["report"]:
        (output / name).write_bytes(abandoned)
    monkeypatch.setattr(cycle, "run_stage", lambda *args, **kwargs: None)
    assert cycle.main(arguments(source, output, "--resume")) == 1
    after = cycle.read_json(output / "cycle.json")
    assert after["completed"] == before["completed"]
    assert after["artifacts"] == before["artifacts"]
    archived = list((output / "attempt-artifacts").rglob("report*.json"))
    assert len(archived) == 2
    assert all(p.read_bytes() == abandoned for p in archived)


@pytest.mark.parametrize("kind", ["external", "internal", "dangling", "hardlink", "fifo"])
def test_unsafe_stage_output_leaves_terminal_failure_without_following_links(
    cycle, completed_case, tmp_path, monkeypatch, kind
):
    import os

    source, output, before = report_prefix(cycle, completed_case, tmp_path)
    target = tmp_path / "must-remain-private"
    target.write_bytes(b"not-an-artifact")
    original = cycle.run_stage

    def unsafe(*args, **kwargs):
        original(*args, **kwargs)
        path = output / "report.json"
        path.unlink()
        try:
            if kind == "hardlink":
                os.link(target, path)
            elif kind == "fifo":
                if not hasattr(os, "mkfifo"):
                    pytest.skip("POSIX FIFO creation is unavailable")
                os.mkfifo(path)
            else:
                destinations = {
                    "external": target,
                    "internal": output / "ranker.json",
                    "dangling": tmp_path / "missing",
                }
                path.symlink_to(destinations[kind])
        except OSError as error:
            pytest.skip(f"filesystem alias creation unavailable: {error}")

    monkeypatch.setattr(cycle, "run_stage", unsafe)
    assert cycle.main(arguments(source, output, "--resume")) == 1
    after = cycle.read_json(output / "cycle.json")
    assert after["status"] == "failed"
    assert after["artifacts"] == before["artifacts"]
    assert after["completed"] == before["completed"]
    assert after["attempts"][-1]["status"] == "failed"
    assert target.read_bytes() == b"not-an-artifact"
    assert "artifact_inventory_error" in after


@pytest.mark.parametrize("name", ["cycle.json", "config.json", "ranker.json", "cycle.json.tmp"])
def test_resume_rejects_file_aliases_before_any_output_write(cycle, completed_case, tmp_path, name):
    source, completed = completed_case
    output = tmp_path / "resume-alias"
    shutil.copytree(completed, output)
    path = output / name
    target = tmp_path / "original"
    target.write_bytes(path.read_bytes() if path.exists() else b"private")
    path.unlink(missing_ok=True)
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    before = snapshot(output)
    target_before = target.read_bytes()
    assert cycle.main(arguments(source, output, "--resume")) == 2
    assert snapshot(output) == before
    assert target.read_bytes() == target_before


@pytest.mark.parametrize(
    "mutation",
    [
        "empty-stages",
        "invalid-stage",
        "negative-time",
        "bool-time",
        "nan-time",
        "duplicate-stage",
        "failed-completed-attempt",
        "nonterminal-running-attempt",
    ],
)
def test_inconsistent_execution_ledger_rejected_without_writes(
    cycle, completed_case, tmp_path, mutation
):
    source, completed = completed_case
    output = tmp_path / "bad-ledger"
    shutil.copytree(completed, output)
    receipt = cycle.read_json(output / "cycle.json")
    if mutation == "empty-stages":
        receipt["stages"] = []
    elif mutation == "invalid-stage":
        receipt["stages"][0] = "not-a-stage"
    elif mutation in {"negative-time", "bool-time", "nan-time"}:
        receipt["stages"][0]["seconds"] = {
            "negative-time": -1,
            "bool-time": True,
            "nan-time": float("nan"),
        }[mutation]
    elif mutation == "duplicate-stage":
        receipt["stages"].insert(0, receipt["stages"][0])
    elif mutation == "failed-completed-attempt":
        receipt["stages"][0]["status"] = "not-completed"
    else:
        receipt["attempts"].insert(0, {"stage": "provision", "status": "running"})
    (output / "cycle.json").write_text(json.dumps(receipt), encoding="utf-8")
    before = snapshot(output)
    assert cycle.main(arguments(source, output, "--resume")) == 2
    assert snapshot(output) == before


@pytest.mark.parametrize(
    "payload",
    [
        '{"x": 1, "x": 2}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{"x": 1e999}',
        '{"x": {"y": 1, "y": 2}}',
    ],
)
def test_ambiguous_json_is_rejected(cycle, tmp_path, payload):
    path = tmp_path / "ambiguous.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        cycle.read_json(path)
    with pytest.raises(ValueError):
        cycle.rows(path)


def test_inventory_error_does_not_mask_original_failure(
    cycle, completed_case, tmp_path, monkeypatch
):
    source, output, before = report_prefix(cycle, completed_case, tmp_path)

    def unsafe(*args, **kwargs):
        (output / "orphan-link").symlink_to(tmp_path / "missing")
        raise RuntimeError("original-worker-failure")

    monkeypatch.setattr(cycle, "run_stage", unsafe)
    assert cycle.main(arguments(source, output, "--resume")) == 1
    after = cycle.read_json(output / "cycle.json")
    assert after["status"] == "failed"
    assert after["completed"] == before["completed"]
    assert "original-worker-failure" in after["reason"]
    assert "artifact_inventory_error" in after


@pytest.mark.parametrize("mutation", ["schema", "empty-cohort", "reordered-cohort"])
def test_freeze_binds_schema_and_exact_inference_cohort(cycle, completed_case, tmp_path, mutation):
    _, completed = completed_case
    output = tmp_path / "bad-freeze"
    shutil.copytree(completed, output)
    frozen = cycle.read_json(output / "freeze.json")
    if mutation == "schema":
        frozen["schema"] = "unknown"
    elif mutation == "empty-cohort":
        frozen["expected_sample_ids"] = []
    else:
        frozen["expected_sample_ids"].reverse()
    cycle.write_json(output / "freeze.json", frozen)
    with pytest.raises(ValueError):
        cycle.verify_freeze(output)


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_id", "missing"),
        ("selected_id", "missing"),
        ("baseline_text", "injected text"),
        ("selected_text", "injected text"),
        ("candidate_texts", ["injected text"]),
        ("requires_additional_evidence", "false"),
    ],
)
def test_evaluation_rejects_decision_not_bound_to_frozen_candidates(
    cycle, completed_case, tmp_path, field, value
):
    _, completed = completed_case
    output = tmp_path / "bad-decision"
    shutil.copytree(completed, output)
    decisions = cycle.rows(output / "decisions.jsonl")
    decisions[0][field] = value
    cycle.write_rows(output / "decisions.jsonl", decisions)
    before = {name: (output / name).read_bytes() for name in cycle.OUTPUTS["evaluate"]}
    with pytest.raises(ValueError):
        cycle.evaluate(output, cycle.read_json(output / "config.json"))
    assert {name: (output / name).read_bytes() for name in before} == before


def test_final_inventory_time_cannot_turn_over_budget_run_into_success(
    cycle, completed_case, tmp_path, monkeypatch
):
    source, output, _ = report_prefix(cycle, completed_case, tmp_path)
    clock = time.monotonic
    offset = [0.0]
    scan_name = "output_files" if hasattr(cycle, "output_files") else "files_under"
    scan = getattr(cycle, scan_name)

    def slow_inventory(directory):
        paths = scan(directory)
        if cycle.read_json(output / "cycle.json")["completed"] == list(cycle.STAGES):
            offset[0] = 61.0
        return paths

    monkeypatch.setattr(cycle.time, "monotonic", lambda: clock() + offset[0])
    monkeypatch.setattr(cycle, scan_name, slow_inventory)
    assert cycle.main(arguments(source, output, "--resume")) == 3
    after = cycle.read_json(output / "cycle.json")
    assert after["status"] == "partial"
    assert after["spent_seconds"] >= 61
    assert after["completed"] == list(cycle.STAGES)


@pytest.mark.parametrize("mutation", ["consistent-selection", "wrong-config"])
def test_evaluation_replays_frozen_policy_before_reference_evaluation(
    cycle, completed_case, tmp_path, mutation
):
    _, completed = completed_case
    output = tmp_path / "policy-replay"
    shutil.copytree(completed, output)
    config = cycle.read_json(output / "config.json")
    if mutation == "wrong-config":
        config["seed"] += 1
    else:
        decisions = cycle.rows(output / "decisions.jsonl")
        candidates = cycle.rows(output / "test-inference.jsonl")[0]["candidates"]
        # A different candidate remains a valid member of the frozen pool. Its
        # text/ID alone are not proof that the frozen policy selected it.
        other = next(c for c in candidates if c["candidate_id"] != decisions[0]["selected_id"])
        decisions[0]["selected_id"] = other["candidate_id"]
        decisions[0]["selected_text"] = other["text"]
        cycle.write_rows(output / "decisions.jsonl", decisions)
    before = {name: (output / name).read_bytes() for name in cycle.OUTPUTS["evaluate"]}
    with pytest.raises(ValueError):
        cycle.evaluate(output, config)
    assert {name: (output / name).read_bytes() for name in before} == before


def test_recovered_active_step_remains_valid_on_subsequent_resume(cycle, completed_case, tmp_path):
    source, output, before = report_prefix(cycle, completed_case, tmp_path)
    before["status"] = "running"
    before["active_started_at"] = time.time()
    before["attempts"].append({"stage": "report", "status": "running"})
    before["stages"].append({"name": "009-report", "status": "running"})
    cycle.write_json(output / "cycle.json", before)
    assert cycle.main(arguments(source, output, "--resume")) == 0
    assert cycle.main(arguments(source, output, "--resume")) == 0
    after = cycle.read_json(output / "cycle.json")
    assert after["attempts"][-2]["status"] == "interrupted"
    assert after["stages"][-2]["status"] == "not-completed"
    assert after["trials_started"] == 1
