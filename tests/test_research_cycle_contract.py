"""Integration contracts for the finite post-candidate cycle, not ASR accuracy."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cycle(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("research_cycle")


def inputs(tmp_path):
    from semantic_asr.contracts import CandidateEvidence

    rows = []
    for split in ("train", "calibration", "test"):
        for index in range(4):
            text = f"{split}の金額は{index + 1}円です"
            candidates = [
                CandidateEvidence("a", text, acoustic=-0.1, avg_logprob=-0.1, rank=1),
                CandidateEvidence("b", text + "ない", acoustic=-0.8, avg_logprob=-0.8, rank=2),
            ]
            rows.append(
                {
                    "sampleId": f"{split}-{index}",
                    "groupId": f"{split}-{index}",
                    "sourceId": f"{split}-{index}",
                    "split": split,
                    "reference": text,
                    "rightsDecision": "allow",
                    "licenseId": "synthetic-test-only",
                    "candidates": [c.as_dict() for c in candidates],
                }
            )
    path = tmp_path / "candidates.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def args(source, output, *extra):
    return [
        "--candidates",
        str(source),
        "--output-dir",
        str(output),
        "--allow-raw-export",
        "--ranker",
        "pairwise",
        "--epochs",
        "3",
        "--max-trials",
        "1",
        "--max-audio-seconds",
        "60",
        "--audio-seconds",
        "1",
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


def test_cycle_executes_once_and_resumes_without_duplicate_stages(cycle, tmp_path):
    source = inputs(tmp_path)
    out = tmp_path / "cycle"
    assert cycle.main(args(source, out)) == 0
    receipt = json.loads((out / "cycle.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    assert receipt["trials_started"] == 1
    assert receipt["completed"] == list(cycle.STAGES)
    before = {p.name: p.read_bytes() for p in out.glob("*.json") if p.name != "cycle.json"}
    assert cycle.main(args(source, out, "--resume")) == 0
    assert before == {p.name: p.read_bytes() for p in out.glob("*.json") if p.name != "cycle.json"}
    report = json.loads((out / "paired-report.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 4
    assert report["strict_cer"]["samples"] == 4
    assert report["promotion"] == "not-evaluated"


@pytest.mark.parametrize("changed", ["input", "ranker", "freeze", "scores", "config"])
def test_resume_rejects_changed_inputs_or_artifacts(cycle, tmp_path, changed):
    source = inputs(tmp_path)
    out = tmp_path / "cycle"
    assert cycle.main(args(source, out)) == 0
    cli = args(source, out, "--resume")
    paths = {
        "input": source,
        "ranker": out / "ranker.json",
        "freeze": out / "freeze.json",
        "scores": out / "calibration-scores.jsonl",
    }
    if changed == "config":
        cli[cli.index("--epochs") + 1] = "4"
    else:
        with paths[changed].open("a", encoding="utf-8") as handle:
            handle.write(" ")
    assert cycle.main(cli) != 0


def test_no_reference_enters_selector(cycle):
    from semantic_asr.contracts import CandidateEvidence

    class Spy:
        name = "spy"

        def score(self, candidates, *, context):
            assert context == ""
            assert all("reference" not in c.metadata for c in candidates)
            return {c.candidate_id: -float(i) for i, c in enumerate(candidates)}

    candidates = (
        CandidateEvidence("a", "えー、えー学校を行く", rank=1),
        CandidateEvidence("b", "学校に行く", rank=2),
    )
    first = cycle.select_candidates(candidates, Spy(), None)
    assert first["baseline_id"] == "a"
    with pytest.raises(TypeError):
        cycle.select_candidates(candidates, Spy(), None, reference="学校に行く")


def test_missing_freeze_prevents_selection(cycle, tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        cycle.verify_freeze(tmp_path)


@pytest.mark.parametrize(
    "flag,value", [("--max-trials", "0"), ("--epochs", "0"), ("--bootstrap-iterations", "99")]
)
def test_invalid_limits_fail_before_training(cycle, tmp_path, flag, value):
    cli = args(inputs(tmp_path), tmp_path / "cycle")
    cli[cli.index(flag) + 1] = value
    with pytest.raises(SystemExit):
        cycle.main(cli)


def test_exposed_data_cannot_claim_unseen_evaluation(cycle, tmp_path):
    cli = args(inputs(tmp_path), tmp_path / "cycle")
    cli[cli.index("--evaluation-role") + 1] = "unseen"
    with pytest.raises(SystemExit):
        cycle.main(cli)


def test_interruption_resumes_after_training_without_repeating_trial(cycle, monkeypatch, tmp_path):
    source = inputs(tmp_path)
    out = tmp_path / "cycle"
    original = cycle.run_stage

    def interrupt(name, *arguments, **kwargs):
        if name.endswith("-calibrate"):
            raise KeyboardInterrupt()
        return original(name, *arguments, **kwargs)

    monkeypatch.setattr(cycle, "run_stage", interrupt)
    assert cycle.main(args(source, out)) == 3
    receipt = cycle.read_json(out / "cycle.json")
    assert receipt["completed"] == ["provision", "fit-train", "score-calibration"]
    assert receipt["trials_started"] == 1
    ranker_before = (out / "ranker.json").read_bytes()
    monkeypatch.setattr(cycle, "run_stage", original)
    assert cycle.main(args(source, out, "--resume")) == 0
    after = cycle.read_json(out / "cycle.json")
    assert after["trials_started"] == 1
    assert after["spent_seconds"] > receipt["spent_seconds"]
    assert (out / "ranker.json").read_bytes() == ranker_before
    assert [a["stage"] for a in after["attempts"]].count("fit-train") == 1


def test_interrupted_optimizer_cannot_reset_trial_budget(cycle, monkeypatch, tmp_path):
    source = inputs(tmp_path)
    out = tmp_path / "cycle"
    original = cycle.run_stage

    def interrupt(name, *arguments, **kwargs):
        if name.endswith("fit-train"):
            raise KeyboardInterrupt()
        return original(name, *arguments, **kwargs)

    monkeypatch.setattr(cycle, "run_stage", interrupt)
    assert cycle.main(args(source, out)) == 3
    monkeypatch.setattr(cycle, "run_stage", original)
    assert cycle.main(args(source, out, "--resume")) == 3
    assert cycle.read_json(out / "cycle.json")["trials_started"] == 1
    assert not (out / "ranker.json").exists()


def test_missing_decision_is_rejected_by_frozen_expected_cohort(cycle, tmp_path):
    out = tmp_path / "cycle"
    assert cycle.main(args(inputs(tmp_path), out)) == 0
    decisions = cycle.rows(out / "decisions.jsonl")
    cycle.write_rows(out / "decisions.jsonl", decisions[:-1])
    with pytest.raises(ValueError, match="cohort"):
        cycle.evaluate(out, cycle.read_json(out / "config.json"))


def test_nested_reference_metadata_rejected(cycle):
    from semantic_asr.contracts import CandidateEvidence

    candidate = CandidateEvidence("a", "仮説", metadata={"evidence": {"reference": "正解"}})
    with pytest.raises(ValueError, match="supervised"):
        cycle.select_candidates((candidate,), None, None)


def test_occupied_writer_lock_prevents_execution(cycle, tmp_path):
    from semantic_asr.experiment_runner import _checkpoint_writer_lock

    source = inputs(tmp_path)
    out = tmp_path / "cycle"
    with _checkpoint_writer_lock(out.with_name(out.name + "-writer")):
        assert cycle.main(args(source, out)) == 2
    assert not out.exists()


def test_success_exit_without_artifacts_is_failure(cycle, monkeypatch, tmp_path):
    monkeypatch.setattr(cycle, "run_stage", lambda *arguments, **kwargs: None)
    out = tmp_path / "cycle"
    assert cycle.main(args(inputs(tmp_path), out)) == 1
    assert cycle.read_json(out / "cycle.json")["completed"] == []


def test_config_mutation_during_stage_fails(cycle, monkeypatch, tmp_path):
    out = tmp_path / "cycle"
    original = cycle.run_stage

    def mutate(*arguments, **kwargs):
        original(*arguments, **kwargs)
        config = cycle.read_json(out / "config.json")
        config["seed"] += 1
        cycle.write_json(out / "config.json", config)

    monkeypatch.setattr(cycle, "run_stage", mutate)
    assert cycle.main(args(inputs(tmp_path), out)) == 1
    assert "configuration changed" in cycle.read_json(out / "cycle.json")["reason"]


def test_all_stages_partial_does_not_become_success_on_resume(cycle, tmp_path):
    source = inputs(tmp_path)
    out = tmp_path / "cycle"
    assert cycle.main(args(source, out)) == 0
    receipt = cycle.read_json(out / "cycle.json")
    receipt.update(status="partial", spent_seconds=61)
    cycle.write_json(out / "cycle.json", receipt)
    assert cycle.main(args(source, out, "--resume")) == 3
    assert cycle.read_json(out / "cycle.json")["status"] == "partial"
