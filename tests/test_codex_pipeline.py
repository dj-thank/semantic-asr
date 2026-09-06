from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "codex_pipeline_under_test", ROOT / "scripts/codex_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path):
    rows = []
    for index, split in enumerate(("train", "calibration", "test")):
        path = tmp_path / f"{index}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setparams((1, 2, 16000, 160, "NONE", "not compressed"))
            handle.writeframes(bytes([index, 0]) * 160)
        rows.append(
            {
                "sampleId": f"sample-{index}",
                "groupId": f"group-{index}",
                "sourceId": f"source-{index}",
                "split": split,
                "audioPath": str(path),
                "reference": f"金額は{index + 1}円です",
                "rightsDecision": "allow",
                "licenseId": "test-fixture-only",
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path, rows


def research_args(path):
    return argparse.Namespace(
        allow_local_research=True,
        manifest=str(path),
        max_records=3,
        max_audio_seconds=1,
        model="fixture-model",
        model_revision="a" * 40,
        model_artifact_sha256=None,
        beam_size=3,
        hypotheses=3,
        bootstrap_iterations=20,
        evaluation_role="regression-exposed",
        device="cpu",
        compute_type="int8",
        ranker="pairwise",
    )


def save_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_plan_is_read_only(driver, monkeypatch, capsys):
    monkeypatch.setattr(driver, "source_identity", lambda: pytest.fail("plan executed git"))
    assert driver.main(["plan"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["trials"] == 1
    assert result["automatic_promotion"] is False


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "inf", "0.5", "true"])
def test_invalid_budgets_rejected(driver, value):
    with pytest.raises((ValueError, argparse.ArgumentTypeError)):
        driver.positive(value)


def test_output_never_overwrites_or_writes_checkout(driver, tmp_path):
    with pytest.raises(FileExistsError):
        driver.output_path(str(tmp_path))
    with pytest.raises(ValueError):
        driver.output_path(str(ROOT / "new-private-run"))
    with pytest.raises(ValueError):
        driver.output_path(str(Path(tmp_path.anchor)))


def test_symlink_containment(driver, tmp_path):
    link = tmp_path / "checkout"
    try:
        link.symlink_to(ROOT, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit creating symlinks")
    with pytest.raises(ValueError):
        driver.output_path(str(link / "new-run"))
    with pytest.raises(ValueError, match="escape"):
        driver.files_under(tmp_path)


def test_authorization_and_rights_are_separate(driver, tmp_path):
    path, rows = manifest(tmp_path)
    args = research_args(path)
    args.allow_local_research = False
    with pytest.raises(driver.Blocked, match="publication"):
        driver.research_input(args)
    args.allow_local_research = True
    rows[1]["rightsDecision"] = "review"
    save_manifest(path, rows)
    with pytest.raises(PermissionError):
        driver.research_input(args)


@pytest.mark.parametrize(
    "change",
    [
        "missing-split",
        "unknown-split",
        "missing-license",
        "duplicate-audio",
        "relative-audio",
        "empty-audio",
        "wrong-format",
    ],
)
def test_bad_manifest_rejected(driver, tmp_path, change):
    path, rows = manifest(tmp_path)
    if change == "missing-split":
        del rows[0]["split"]
    elif change == "unknown-split":
        rows[0]["split"] = "unseen-by-assumption"
    elif change == "missing-license":
        del rows[1]["licenseId"]
    elif change == "duplicate-audio":
        rows[1]["audioPath"] = rows[0]["audioPath"]
    elif change == "relative-audio":
        rows[0]["audioPath"] = "0.wav"
    else:
        with wave.open(rows[0]["audioPath"], "wb") as handle:
            handle.setparams((1, 2, 8000 if change == "wrong-format" else 16000, 0, "NONE", "x"))
            if change == "wrong-format":
                handle.writeframes(b"\0\0" * 20)
    save_manifest(path, rows)
    with pytest.raises((ValueError, PermissionError)):
        driver.research_input(research_args(path))


def test_manifest_identity_and_finite_duration(driver, tmp_path):
    path, _ = manifest(tmp_path)
    args = research_args(path)
    receipt = driver.research_input(args)
    assert receipt["record_count"] == 3
    assert receipt["audio_seconds"] == pytest.approx(0.03)
    assert receipt["speaker_disjointness_verified"] is False
    assert len(set(receipt["audio_sha256"])) == 3
    args.max_audio_seconds = 0.01
    with pytest.raises(driver.BudgetExceeded):
        driver.research_input(args)
    args.max_audio_seconds = 1
    args.max_records = 2
    with pytest.raises(ValueError):
        driver.research_input(args)


@pytest.mark.parametrize("revision", [None, "main", "latest", "a" * 39, "g" * 40])
def test_mutable_model_revision_rejected(driver, tmp_path, revision):
    path, _ = manifest(tmp_path)
    args = research_args(path)
    args.model_revision = revision
    with pytest.raises(ValueError):
        driver.research_input(args)


def test_local_model_artifact_uses_existing_hash_contract(driver, tmp_path):
    from semantic_asr.revisions import sha256_artifact

    path, _ = manifest(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").write_bytes(b"fixture-not-real-weights")
    args = research_args(path)
    args.model = str(model)
    args.model_revision = None
    args.model_artifact_sha256 = sha256_artifact(model)
    assert driver.research_input(args)["model"]["artifact_sha256"] == args.model_artifact_sha256
    (model / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        driver.research_input(args)


def test_stage_failure_and_log_survive(driver, tmp_path):
    receipt = {"stages": []}
    with pytest.raises(subprocess.CalledProcessError):
        driver.run_stage(
            "failure",
            [sys.executable, "-c", "print('fixture'); raise SystemExit(7)"],
            tmp_path,
            receipt,
            time.monotonic() + 10,
            1024 * 1024,
        )
    persisted = json.loads((tmp_path / "receipt.json").read_text())
    assert persisted["stages"][0]["returncode"] == 7
    assert persisted["stages"][0]["status"] == "not-completed"
    assert "fixture" in (tmp_path / "failure.log").read_text()


def test_timeout_leaves_no_running_child(driver, tmp_path):
    receipt = {"stages": []}
    started = time.monotonic()
    with pytest.raises(driver.BudgetExceeded):
        driver.run_stage(
            "timeout",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            receipt,
            started + 0.3,
            1024 * 1024,
        )
    assert time.monotonic() - started < 10
    assert receipt["stages"][0]["status"] == "not-completed"


def test_storage_budget_stops_child(driver, tmp_path):
    receipt = {"stages": []}
    with pytest.raises(driver.BudgetExceeded):
        driver.run_stage(
            "storage",
            [sys.executable, "-c", "print('x'*100000)"],
            tmp_path,
            receipt,
            time.monotonic() + 10,
            4096,
        )
    assert receipt["stages"][0]["status"] == "not-completed"


def test_junit_distinguishes_skips_and_failures(driver, tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text(
        "<testsuites><testsuite><testcase/><testcase><skipped/></testcase>"
        "<testcase><failure/></testcase><testcase><error/></testcase></testsuite></testsuites>"
    )
    assert driver.junit_counts(path) == {"tests": 4, "skipped": 1, "failures": 1, "errors": 1}
    path.write_text("<testsuites/>")
    with pytest.raises(ValueError, match="no executed"):
        driver.junit_counts(path)


def test_blocked_run_is_not_success_and_has_receipt(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(
        driver, "source_identity", lambda: {"dirty": False, "head": "a", "tree": "b"}
    )

    def blocked(*args):
        raise driver.Blocked("test environment unavailable")

    monkeypatch.setattr(driver, "check", blocked)
    out = tmp_path / "blocked"
    assert (
        driver.main(
            [
                "check",
                "--output-dir",
                str(out),
                "--max-wall-seconds",
                "10",
                "--max-storage-bytes",
                "100000",
            ]
        )
        == 2
    )
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "blocked"
    assert receipt["promotion"] == "not-evaluated"
    assert receipt["stages"] == []


def test_source_mutation_fails_even_when_commands_pass(driver, monkeypatch, tmp_path):
    calls = iter(
        [{"dirty": False, "head": "a", "tree": "b"}, {"dirty": True, "head": "a", "tree": "b"}]
    )
    monkeypatch.setattr(driver, "source_identity", lambda: next(calls))
    monkeypatch.setattr(driver, "check", lambda *args: None)
    out = tmp_path / "changed"
    assert (
        driver.main(
            [
                "check",
                "--output-dir",
                str(out),
                "--max-wall-seconds",
                "10",
                "--max-storage-bytes",
                "100000",
            ]
        )
        == 1
    )
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["reason"] == "source changed during execution"


def test_research_calls_shared_driver_with_explicit_authorization(driver, monkeypatch, tmp_path):
    path, rows = manifest(tmp_path)
    args = research_args(path)
    output = tmp_path / "out"
    output.mkdir()
    monkeypatch.setattr(driver, "require_modules", lambda names: None)
    scripts = tmp_path / "bin"
    scripts.mkdir()
    executable = scripts / ("semantic-asr.exe" if os.name == "nt" else "semantic-asr")
    executable.touch()
    monkeypatch.setattr(driver.sysconfig, "get_path", lambda name: str(scripts))
    calls = []

    def stage(name, command):
        calls.append((name, command))
        if name == "post-candidate":
            folder = output / "pipeline"
            folder.mkdir()
            for name in ("report.json", "report-raw.json", "ranker.json", "calibration.json"):
                payload = (
                    {"sample_count": 1, "baseline_cer": 0.2, "cascade_cer": 0.2, "mbr_cer": 0.2}
                    if name.startswith("report")
                    else {"profile": {"fixture": True}}
                )
                (folder / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = {"source": {"head": "a" * 40, "tree": "b" * 40}, "environment": driver.environment()}
    driver.research(args, output, receipt, stage)
    assert [name for name, _ in calls] == ["generate", "post-candidate"]
    assert "--allow-raw-export" in calls[0][1]
    assert "--allow-raw-export" in calls[1][1]
    assert "scripts/run_real_audio_pipeline.py" in calls[1][1]
    assert receipt["new_acoustic_or_lora_weights"] is False
    assert rows  # The test uses generated silent WAV fixtures, never a real ASR model.


def test_truncated_audio_rejected(driver, tmp_path):
    path, rows = manifest(tmp_path)
    audio = Path(rows[0]["audioPath"])
    audio.write_bytes(audio.read_bytes()[:-10])
    with pytest.raises(ValueError, match="truncated"):
        driver.research_input(research_args(path))


def test_actual_post_candidate_driver_trains_and_evaluates_synthetic_rows(
    driver, monkeypatch, tmp_path
):
    """Real CLI optimization/evaluation; synthetic candidates, not real ASR inference."""
    from semantic_asr.contracts import CandidateEvidence

    # Cloud setup activation does not persist into the agent session.
    monkeypatch.setenv("PATH", "")
    rows = []
    for index, split in enumerate(("train", "calibration", "test")):
        for number in range(4):
            text = f"金額は{index * 10 + number + 1}円です"
            candidates = [
                CandidateEvidence("a", text, acoustic=-0.1, avg_logprob=-0.1, rank=1),
                CandidateEvidence("b", text + "ない", acoustic=-0.8, avg_logprob=-0.8, rank=2),
            ]
            rows.append(
                {
                    "sampleId": f"{split}-{number}",
                    "groupId": f"{split}-{number}",
                    "sourceId": f"{split}-{number}",
                    "split": split,
                    "reference": text,
                    "rightsDecision": "allow",
                    "licenseId": "synthetic-test-only",
                    "candidates": [candidate.as_dict() for candidate in candidates],
                }
            )
    source = tmp_path / "synthetic.jsonl"
    save_manifest(source, rows)
    output = tmp_path / "pipeline"
    receipt = {"stages": []}
    driver.run_stage(
        "post-candidate",
        [
            sys.executable,
            "scripts/run_real_audio_pipeline.py",
            "--candidates",
            str(source),
            "--output-dir",
            str(output),
            "--allow-raw-export",
            "--ranker",
            "pairwise",
            "--bootstrap-iterations",
            "20",
        ],
        tmp_path,
        receipt,
        time.monotonic() + 30,
        10 * 1024 * 1024,
    )
    report = json.loads((output / "report.json").read_text())
    assert report["sample_count"] == 4
    assert (output / "ranker.json").stat().st_size > 100
    assert (output / "calibration.json").stat().st_size > 100


def test_model_free_optimization_uses_current_teacher_contract(driver, tmp_path):
    output = tmp_path / "optimization.json"
    driver.run_stage(
        "optimization",
        [sys.executable, "scripts/run_v02_model_free_validation.py", "--output", str(output)],
        tmp_path,
        {"stages": []},
        time.monotonic() + 30,
        10 * 1024 * 1024,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    consensus = result["teacherConsensus"]
    assert consensus["usable_for_distillation"]
    assert set(consensus["active_teachers"]) == {"teacher-8b", "teacher-12b"}
    assert consensus["teacher_entropies"]["teacher-8b"] != 0.18
    assert consensus["teacher_entropies"]["teacher-12b"] != 0.22
    assert result["progressiveReranking"]["early_exit"]
