"""Characterize the fixed Codex verification runner without ASR/model downloads."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "codex_verify.py"
spec = importlib.util.spec_from_file_location("codex_verify", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    (root / "source.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return root


@pytest.mark.parametrize("profile", runner.PROFILES)
def test_plan_only_is_side_effect_free(tmp_path, profile):
    output = tmp_path / "not-created"
    assert runner.main(["--plan", "--profile", profile, "--output-dir", str(output)]) == 0
    assert not output.exists()


def test_unknown_profile_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        runner.commands(tmp_path, tmp_path, "magic")


@pytest.mark.parametrize("where", ["root", "checkout", "child", "existing"])
def test_bad_output_rejected(repo, tmp_path, where):
    output = {"root": Path(repo.anchor), "checkout": repo,
              "child": repo / "new", "existing": tmp_path}[where]
    with pytest.raises(ValueError):
        runner.safe_output(repo, output)


def test_symlink_into_checkout_rejected(repo, tmp_path):
    link = tmp_path / "link"
    try:
        link.symlink_to(repo, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not allow symlink creation")
    with pytest.raises(ValueError):
        runner.safe_output(repo, link / "output")


def test_dirty_source_is_hashed_not_hidden(repo):
    before = runner.source_identity(repo)
    (repo / "source.py").write_text("x = 2\n", encoding="utf-8")
    dirty = runner.source_identity(repo)
    assert before["head"] == dirty["head"]
    assert before["effective_source_sha256"] != dirty["effective_source_sha256"]
    assert dirty["dirty"] is True
    (repo / "new.py").write_text("y = 3\n", encoding="utf-8")
    assert runner.source_identity(repo)["effective_source_sha256"] != dirty["effective_source_sha256"]


def test_failure_stops_later_commands_and_records_evidence(repo, tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    stages = [runner.Stage("fail", (sys.executable, "-c", "raise SystemExit(7)"), repo),
              runner.Stage("never", (sys.executable, "-c", "raise AssertionError"), repo)]
    monkeypatch.setattr(runner, "commands", lambda *args: stages)
    assert runner.verify(repo, output, "installed", 30, 10) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed"
    assert [s["name"] for s in report["stages"]] == ["fail"]
    assert report["stages"][0]["returncode"] == 7
    assert report["source_before"] == report["source_after"]
    assert report["experiment_complete"] is False
    assert report["promotion_approved"] is False
    assert str(repo) not in json.dumps(report)
    with pytest.raises(ValueError):
        runner.verify(repo, output, "installed", 30, 10)


def test_success_does_not_imply_research_completion(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "commands", lambda *args: [
        runner.Stage("ok", (sys.executable, "-c", "print('OK')"), repo)
    ])
    output = tmp_path / "evidence"
    assert runner.verify(repo, output, "installed", 30, 10) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed"
    assert report["new_model_inference"] is False
    assert report["new_acoustic_or_llm_weights"] is False


def test_source_modification_is_failure(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "commands", lambda *args: [
        runner.Stage("modify", (sys.executable, "-c",
            "from pathlib import Path; Path('source.py').write_text('x = 2\\n')"), repo)
    ])
    output = tmp_path / "evidence"
    assert runner.verify(repo, output, "installed", 30, 10) == 1
    assert json.loads((output / "report.json").read_text())["status"] == "source-changed"


@pytest.mark.parametrize("total,stage", [(0, 10), (-1, 10), (10, 0), (10, -1)])
def test_invalid_budgets_do_not_create_output(repo, tmp_path, total, stage):
    output = tmp_path / "evidence"
    with pytest.raises(ValueError):
        runner.verify(repo, output, "installed", total, stage)
    assert not output.exists()


def test_timeout_is_nonzero_and_recorded(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "commands", lambda *args: [
        runner.Stage("timeout", (sys.executable, "-c", "import time; time.sleep(20)"), repo)
    ])
    output = tmp_path / "evidence"
    assert runner.verify(repo, output, "installed", 1, 1) == 124
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "timeout"
    assert report["stages"][0]["returncode"] is None


def test_zero_remaining_budget_never_starts_child(tmp_path):
    marker = tmp_path / "marker"
    stage = runner.Stage("no-time", ("nonexistent-executable",), tmp_path)
    assert runner.run_stage(stage, marker, 0) == ("timeout", None)
    assert not marker.exists()


def test_failure_to_start_is_explicit(tmp_path):
    stage = runner.Stage("missing", (str(tmp_path / "missing-executable"),), tmp_path)
    assert runner.run_stage(stage, tmp_path / "log", 1) == ("failed-to-start", None)


def test_junit_retains_skips_and_expected_failures(tmp_path):
    xml = tmp_path / "pytest.xml"
    xml.write_text(
        '<testsuites><testsuite><testcase/><testcase><failure/></testcase>'
        '<testcase><skipped type="pytest.skip"/></testcase>'
        '<testcase><skipped type="pytest.xfail"/></testcase>'
        '<testcase><error/></testcase></testsuite></testsuites>', encoding="utf-8"
    )
    assert runner.test_counts(xml) == {
        "tests": 5, "failures": 1, "errors": 1, "skipped": 2, "xfail": 1
    }
    assert runner.test_counts(tmp_path / "missing") is None


def test_wheel_must_come_from_this_run(tmp_path):
    with pytest.raises(ValueError):
        runner.wheel_commands(tmp_path)
    folder = tmp_path / "wheelhouse"
    folder.mkdir()
    (folder / "one.whl").touch()
    stages = runner.wheel_commands(tmp_path)
    assert all(stage.cwd == tmp_path for stage in stages)
    assert "--no-index" in stages[1].argv and "--no-deps" in stages[1].argv
    assert "-I" in stages[2].argv
    (folder / "two.whl").touch()
    with pytest.raises(ValueError):
        runner.wheel_commands(tmp_path)


def test_fixed_commands_do_not_download_models_or_run_real_audio(tmp_path):
    for profile in runner.PROFILES:
        stages = runner.commands(tmp_path, tmp_path / "out", profile)
        text = " ".join(" ".join(stage.argv) for stage in stages)
        assert "generate-candidates" not in text
        assert "train_public_weight_pilot" not in text
        assert "--fix" not in text
        assert "--no-isolation" not in text
        assert "--skip-dependency-check" not in text


def test_model_offline_environment_and_wheel_pythonpath_removal(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "should-not-leak"))
    stage = runner.Stage("wheel-env-probe", (sys.executable, "-c", (
        "import os; assert 'PYTHONPATH' not in os.environ; "
        "assert os.environ['HF_HUB_OFFLINE'] == '1'"
    )), tmp_path)
    assert runner.run_stage(stage, tmp_path / "env.log", 10) == ("passed", 0)


def test_no_shell_is_used():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "danger-full-access" not in source
    assert os.path.isfile(SCRIPT)


def test_real_cli_entry_creates_demo_and_smoke(tmp_path, monkeypatch):
    root = SCRIPT.parents[1]
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    stages = runner.commands(root, tmp_path, "installed")
    for name, filename in (("demo", "demo.json"), ("smoke", "research-smoke.json")):
        stage = next(s for s in stages if s.name == name)
        assert "semantic_asr.cli_root" not in stage.argv
        assert runner.run_stage(stage, tmp_path / f"{name}.log", 30) == ("passed", 0)
        assert json.loads((tmp_path / filename).read_text(encoding="utf-8"))


def test_zero_exit_without_required_artifact_is_failure(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "commands", lambda *args: [
        runner.Stage("demo", (sys.executable, "-c", "pass"), repo)
    ])
    output = tmp_path / "evidence"
    assert runner.verify(repo, output, "installed", 30, 10) == 1
    assert json.loads((output / "report.json").read_text())["status"] == "missing-artifact"


def test_invalid_junit_does_not_pass(repo, tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    monkeypatch.setattr(runner, "commands", lambda *args: [
        runner.Stage("tests", (sys.executable, "-c",
            f"from pathlib import Path; Path({str(output / 'pytest.xml')!r}).write_text('<testsuites/>')"),
            repo)
    ])
    assert runner.verify(repo, output, "installed", 30, 10) == 1
    assert json.loads((output / "report.json").read_text())["status"] == "invalid-test-evidence"
