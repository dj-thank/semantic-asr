"""Export completion is a verified set of bytes, not the presence of one file."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_runtime_reliability import actual_result

from semantic_asr import outputs


@pytest.mark.parametrize("facade", [False, True])
def test_export_records_all_selected_artifacts_and_evidence(tmp_path, facade):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    paths = result.write(root) if facade else outputs.write_outputs(result.longform, root)
    assert "manifest" in paths
    manifest = outputs.verify_output_bundle(paths["manifest"])
    assert manifest["schema"] == "semantic-asr-output-bundle-v1"
    assert manifest["status"] == "complete"
    assert manifest["evidenceSha256"] == result.evidence_sha256
    assert set(manifest["artifacts"]) == set(paths) - {"manifest"}
    assert str(tmp_path) not in Path(paths["manifest"]).read_text(encoding="utf-8")
    for name, artifact in manifest["artifacts"].items():
        data = Path(paths[name]).read_bytes()
        assert artifact == {
            "filename": Path(paths[name]).name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "sizeBytes": len(data),
        }


def test_interrupted_overwrite_removes_old_completion_marker(tmp_path, monkeypatch):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    paths = result.write(root)
    original = outputs.atomic_write
    calls = []

    def interrupt(path, content, **kwargs):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("injected disk failure")
        return original(path, content, **kwargs)

    monkeypatch.setattr(outputs, "atomic_write", interrupt)
    with pytest.raises(OSError, match="injected disk failure"):
        result.write(root, overwrite=True)
    assert not Path(paths.get("manifest", root / "actual.semantic-asr.bundle.json")).exists()
    assert len(calls) == 2
    monkeypatch.setattr(outputs, "atomic_write", original)
    repaired = result.write(root, overwrite=True)
    assert "manifest" in repaired
    outputs.verify_output_bundle(repaired["manifest"])


@pytest.mark.parametrize("operation", ["change", "delete", "broken-symlink"])
def test_verifier_rejects_changed_or_missing_artifacts(tmp_path, operation):
    result, _ = actual_result(tmp_path)
    paths = result.write(tmp_path / "out")
    assert "manifest" in paths
    path = Path(paths["observed"])
    if operation == "change":
        path.write_text("別の発話", encoding="utf-8")
    else:
        path.unlink()
        if operation == "broken-symlink":
            path.symlink_to(tmp_path / "absent.txt")
    with pytest.raises((ValueError, FileNotFoundError)):
        outputs.verify_output_bundle(paths["manifest"])


def test_known_conflict_preserves_previous_bundle(tmp_path):
    result, _ = actual_result(tmp_path)
    paths = result.write(tmp_path / "out")
    before = {key: Path(path).read_bytes() for key, path in paths.items()}
    with pytest.raises(FileExistsError):
        result.write(tmp_path / "out")
    assert {key: Path(path).read_bytes() for key, path in paths.items()} == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_export_rejects_broken_symlinks_before_any_publication(tmp_path, overwrite):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    root.mkdir()
    link = root / "actual.txt"
    link.symlink_to(tmp_path / "missing.txt")
    with pytest.raises((FileExistsError, ValueError)):
        result.write(root, overwrite=overwrite)
    assert list(root.iterdir()) == [link]
    assert link.is_symlink()


def test_concurrent_bundle_write_and_read_are_rejected(tmp_path, monkeypatch):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    paths = result.write(root)
    original = outputs.atomic_write
    checked = False

    def concurrent(path, content, **kwargs):
        nonlocal checked
        if not checked:
            checked = True
            with pytest.raises(RuntimeError, match="active writer"):
                result.write(root, overwrite=True)
            with pytest.raises(RuntimeError, match="active writer"):
                outputs.verify_output_bundle(paths["manifest"])
        return original(path, content, **kwargs)

    monkeypatch.setattr(outputs, "atomic_write", concurrent)
    result.write(root, overwrite=True)
    assert checked
    outputs.verify_output_bundle(paths["manifest"])


@pytest.mark.parametrize("change", ["schema", "status", "traversal", "size", "duplicate"])
def test_verifier_rejects_malformed_completion_manifest(tmp_path, change):
    result, _ = actual_result(tmp_path)
    paths = result.write(tmp_path / "out")
    assert "manifest" in paths
    path = Path(paths["manifest"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if change in {"schema", "status"}:
        payload[change] = "invalid"
    elif change == "traversal":
        payload["artifacts"]["observed"]["filename"] = "../outside.txt"
    elif change == "size":
        payload["artifacts"]["observed"]["sizeBytes"] = True
    else:
        payload["artifacts"]["observed"] = payload["artifacts"]["json"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        outputs.verify_output_bundle(path)


def test_selected_formats_and_empty_export_keep_their_contract(tmp_path):
    result, _ = actual_result(tmp_path)
    assert result.write(tmp_path / "empty", formats=set()) == {}
    assert not (tmp_path / "empty").exists()
    paths = result.write(tmp_path / "out", formats={"observed"})
    assert set(paths) == {"observed", "manifest"}
    assert set(outputs.verify_output_bundle(paths["manifest"])["artifacts"]) == {"observed"}


def test_reduced_formats_replace_manifest_without_claiming_old_files(tmp_path):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    full = result.write(root)
    reduced = result.write(root, formats={"observed"}, overwrite=True)
    assert Path(full["srt"]).exists()
    assert set(outputs.verify_output_bundle(reduced["manifest"])["artifacts"]) == {"observed"}


def test_expected_evidence_and_legacy_export_are_checked(tmp_path):
    result, _ = actual_result(tmp_path)
    paths = result.write(tmp_path / "out")
    outputs.verify_output_bundle(paths["manifest"], expected_evidence_sha256=result.evidence_sha256)
    with pytest.raises(ValueError, match="expected evidence"):
        outputs.verify_output_bundle(paths["manifest"], expected_evidence_sha256="0" * 64)
    with pytest.raises(ValueError, match="manifest"):
        outputs.verify_output_bundle(paths["json"])


def test_duplicate_json_keys_and_symlinked_manifest_are_rejected(tmp_path):
    result, _ = actual_result(tmp_path)
    paths = result.write(tmp_path / "out")
    manifest = Path(paths["manifest"])
    valid = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        valid.replace('"status": "complete"', '"status": "bad", "status": "complete"')
    )
    with pytest.raises(ValueError, match="duplicate key"):
        outputs.verify_output_bundle(manifest)
    saved = tmp_path / "saved.json"
    saved.write_text(valid, encoding="utf-8")
    manifest.unlink()
    manifest.symlink_to(saved)
    with pytest.raises(ValueError, match="manifest"):
        outputs.verify_output_bundle(manifest)


def test_invalid_utf8_is_rejected_before_any_publication(tmp_path):
    root = tmp_path / "out"
    with pytest.raises(UnicodeEncodeError):
        outputs.publish_output_documents(
            {"good": (root / "good.txt", "valid"), "bad": (root / "bad.txt", "\ud800")},
            manifest_path=root / "result.bundle.json",
            evidence_sha256="a" * 64,
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "filename",
    [
        ".semantic-asr-export.lock",
        ".SEMANTIC-ASR-EXPORT.LOCK",
        ".semantic-asr-export.lock.",
        ".semantic-asr-export.lock ",
    ],
)
def test_manifest_cannot_replace_shared_lock_inode(tmp_path, filename):
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="manifest filename"):
        outputs.publish_output_documents(
            {"observed": (root / "clip.txt", "speech")},
            overwrite=True,
            manifest_path=root / filename,
            evidence_sha256="a" * 64,
        )
    assert not root.exists()


def test_artifacts_cannot_alias_shared_lock_or_each_other(tmp_path):
    root = tmp_path / "out"
    for documents in (
        {"observed": (root / ".SEMANTIC-ASR-EXPORT.LOCK", "speech")},
        {"a": (root / "clip.txt", "first"), "b": (root / "CLIP.TXT", "second")},
    ):
        with pytest.raises(ValueError):
            outputs.publish_output_documents(
                documents,
                overwrite=True,
                manifest_path=root / "clip.bundle.json",
                evidence_sha256="a" * 64,
            )
    assert not root.exists()


def test_another_process_lock_is_released_on_termination(tmp_path):
    result, _ = actual_result(tmp_path)
    root = tmp_path / "out"
    paths = result.write(root)
    code = (
        "import sys; from pathlib import Path; "
        "from semantic_asr.outputs import _bundle_lock; "
        "lock = _bundle_lock(Path(sys.argv[1])); lock.__enter__(); "
        "print('locked', flush=True); sys.stdin.read()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="active writer"):
            result.write(root, overwrite=True)
        with pytest.raises(RuntimeError, match="active writer"):
            outputs.verify_output_bundle(paths["manifest"])
    finally:
        process.terminate()
        process.communicate(timeout=15)
    outputs.verify_output_bundle(paths["manifest"])
    result.write(root, overwrite=True)


def test_manifest_write_failure_is_not_a_complete_bundle(tmp_path, monkeypatch):
    result, _ = actual_result(tmp_path)
    original = outputs.atomic_write

    def fail_manifest(path, content, **kwargs):
        if path.name.endswith(".bundle.json"):
            raise OSError("injected manifest failure")
        return original(path, content, **kwargs)

    monkeypatch.setattr(outputs, "atomic_write", fail_manifest)
    root = tmp_path / "out"
    with pytest.raises(OSError, match="manifest failure"):
        result.write(root)
    assert (root / "actual.observed.txt").exists()
    assert not (root / "actual.semantic-asr.bundle.json").exists()


def test_external_change_during_publication_prevents_completion(tmp_path, monkeypatch):
    result, _ = actual_result(tmp_path)
    original = outputs.atomic_write

    def tamper(path, content, **kwargs):
        published = original(path, content, **kwargs)
        if path.name.endswith(".observed.txt"):
            path.write_text("別の発話", encoding="utf-8")
        return published

    monkeypatch.setattr(outputs, "atomic_write", tamper)
    root = tmp_path / "out"
    with pytest.raises(ValueError, match="does not match manifest"):
        result.write(root)
    assert not (root / "actual.semantic-asr.bundle.json").exists()


def test_overlapping_bundle_names_share_lock_and_require_reverification(tmp_path, monkeypatch):
    root = tmp_path / "out"
    first = {"observed": (root / "clip.observed.txt", "original\n")}
    second = {"normalized": (root / "clip.observed.txt", "replacement\n")}
    first_manifest = root / "clip.semantic-asr.bundle.json"
    second_manifest = root / "clip.observed.semantic-asr.bundle.json"
    original = outputs.atomic_write

    def concurrent(path, content, **kwargs):
        with pytest.raises(RuntimeError, match="active writer"):
            outputs.publish_output_documents(
                second, overwrite=True, manifest_path=second_manifest, evidence_sha256="b" * 64
            )
        return original(path, content, **kwargs)

    monkeypatch.setattr(outputs, "atomic_write", concurrent)
    outputs.publish_output_documents(first, manifest_path=first_manifest, evidence_sha256="a" * 64)
    monkeypatch.setattr(outputs, "atomic_write", original)
    outputs.publish_output_documents(
        second, overwrite=True, manifest_path=second_manifest, evidence_sha256="b" * 64
    )
    outputs.verify_output_bundle(second_manifest)
    with pytest.raises(ValueError, match="does not match manifest"):
        outputs.verify_output_bundle(first_manifest)
