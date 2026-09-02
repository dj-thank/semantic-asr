from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "run_real_audio_pipeline.py"
    spec = importlib.util.spec_from_file_location("test_run_real_audio_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidate_row(
    *,
    reference: str | None = "秘密の参照文",
    decision: str | None = "allow",
    license_id: str | None = "fixture-license",
    nested_license: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sampleId": "sample-1",
        "candidates": [{"candidateId": "candidate-1", "text": "仮説"}],
    }
    if reference is not None:
        row["reference"] = reference
    if decision is not None:
        row["rightsDecision"] = decision
    if license_id is not None:
        if nested_license:
            row["generation"] = {"licenseId": license_id}
        else:
            row["licenseId"] = license_id
    return row


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_cli(monkeypatch, pipeline):
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        commands.append(command)
        if command[1] in {"train-listwise-ranker", "train-ranker"}:
            output = Path(command[command.index("--output") + 1])
            output.write_text('{"profile":{"name":"fixture-ranker"}}', encoding="utf-8")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    return commands


def test_reference_bearing_input_requires_explicit_local_research_authorization(
    monkeypatch, tmp_path: Path
) -> None:
    pipeline = _load_script()
    source = tmp_path / "candidates.jsonl"
    _write_rows(source, [_candidate_row()])
    output = tmp_path / "external-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_audio_pipeline.py",
            "--candidates",
            str(source),
            "--output-dir",
            str(output),
        ],
    )

    with pytest.raises(SystemExit):
        pipeline.main()
    assert not output.exists()


@pytest.mark.parametrize(
    ("decision", "license_id"),
    [
        (None, "fixture-license"),
        ("review", "fixture-license"),
        ("deny", "fixture-license"),
        ("allow", None),
    ],
)
def test_reference_bearing_input_fails_closed_without_allow_license_evidence(
    monkeypatch, tmp_path: Path, decision: str | None, license_id: str | None
) -> None:
    pipeline = _load_script()
    source = tmp_path / "candidates.jsonl"
    _write_rows(source, [_candidate_row(decision=decision, license_id=license_id)])
    output = tmp_path / "external-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_audio_pipeline.py",
            "--candidates",
            str(source),
            "--output-dir",
            str(output),
            "--allow-raw-export",
        ],
    )

    with pytest.raises(PermissionError, match="rights|license"):
        pipeline.main()
    assert not output.exists()


def test_generated_shape_accepts_nested_license_with_explicit_authorization(
    monkeypatch, tmp_path: Path
) -> None:
    pipeline = _load_script()
    source = tmp_path / "candidates.jsonl"
    _write_rows(source, [_candidate_row(nested_license=True)])
    output = tmp_path / "external-output"
    commands = _fake_cli(monkeypatch, pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_audio_pipeline.py",
            "--candidates",
            str(source),
            "--output-dir",
            str(output),
            "--allow-raw-export",
        ],
    )

    assert pipeline.main() == 0
    assert (output / "all-candidates.jsonl").is_file()
    assert len(commands) == 7
    assert all(
        str(output) in argument
        for command in commands
        for argument in command
        if argument.endswith((".json", ".jsonl")) or argument == str(output / "splits")
    )


def test_reference_free_metadata_rows_keep_existing_safe_command_path(
    monkeypatch, tmp_path: Path
) -> None:
    pipeline = _load_script()
    source = tmp_path / "metadata-only.jsonl"
    _write_rows(source, [_candidate_row(reference=None, decision=None, license_id=None)])
    output = tmp_path / "external-output"
    _fake_cli(monkeypatch, pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_audio_pipeline.py",
            "--candidates",
            str(source),
            "--output-dir",
            str(output),
        ],
    )

    assert pipeline.main() == 0
    assert (output / "all-candidates.jsonl").is_file()


def test_pipeline_rejects_checkout_output_even_when_ignored(tmp_path: Path) -> None:
    pipeline = _load_script()
    with pytest.raises(ValueError, match="outside the repository"):
        pipeline.ensure_safe_output_dir(ROOT / "runs" / "pipeline")


def test_pipeline_rejects_filesystem_root_output() -> None:
    pipeline = _load_script()
    with pytest.raises(ValueError, match="filesystem root"):
        pipeline.ensure_safe_output_dir(Path(Path.cwd().anchor))


def test_pipeline_rejects_symlinked_output_into_checkout(tmp_path: Path) -> None:
    pipeline = _load_script()
    link = tmp_path / "checkout-link"
    try:
        link.symlink_to(ROOT, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="outside the repository"):
        pipeline.ensure_safe_output_dir(link / "pipeline")


def test_pipeline_checks_derived_output_paths_for_symlinks(tmp_path: Path) -> None:
    pipeline = _load_script()
    output = tmp_path / "external-output"
    output.mkdir()
    split_link = output / "splits"
    try:
        split_link.symlink_to(ROOT, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="outside the repository"):
        pipeline._pipeline_output_paths(output)


def test_pipeline_checks_each_split_file_for_symlinks(tmp_path: Path) -> None:
    pipeline = _load_script()
    output = tmp_path / "external-output"
    split_dir = output / "splits"
    split_dir.mkdir(parents=True)
    split_link = split_dir / "train.jsonl"
    try:
        split_link.symlink_to(ROOT / "README.md")
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="outside the repository"):
        pipeline._pipeline_output_paths(output)
