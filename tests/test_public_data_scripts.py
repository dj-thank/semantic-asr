from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

from semantic_asr.contracts import CandidateEvidence

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_data_extra_declares_all_loader_dependencies() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = payload["project"]["optional-dependencies"]["public-data"]
    names = {str(requirement).split(">=", 1)[0].split("<", 1)[0] for requirement in requirements}
    assert {"datasets", "numpy", "scipy", "soundfile"} <= names


def test_prepare_manifest_requires_explicit_raw_export(monkeypatch, tmp_path: Path) -> None:
    prepare = _load_script("prepare_public_manifest")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_public_manifest.py", "reazonspeech-test", "--output-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit):
        prepare.main()


def test_prepare_rights_allow_only_exact_asset_or_explicit_operator_decision() -> None:
    prepare = _load_script("prepare_public_manifest")
    exact_revision = prepare.PUBLIC_DATASET_REVISIONS[prepare.DATASETS["reazonspeech-test"]["path"]]
    assert prepare.resolve_rights_decision("reazonspeech-test", exact_revision) == "allow"
    assert prepare.resolve_rights_decision("reazonspeech-test", "0" * 40) == "review"
    assert prepare.resolve_rights_decision("jsut-basic5000", "1" * 40) == "review"
    assert prepare.resolve_rights_decision("jsut-basic5000", "1" * 40, requested="allow") == "allow"


def test_reference_digest_groups_normalized_duplicates_into_one_split() -> None:
    prepare = _load_script("prepare_public_manifest")
    first = prepare.normalized_reference_digest("東 京です")
    second = prepare.normalized_reference_digest("東京です")
    assert first == second
    assert prepare.assign_split(first, "fixed-seed") == prepare.assign_split(second, "fixed-seed")


def test_prepare_rejects_checkout_output_even_when_ignored() -> None:
    prepare = _load_script("prepare_public_manifest")
    with pytest.raises(ValueError, match="outside the repository"):
        prepare.ensure_safe_output_dir(ROOT / "data" / "reazon")


def test_prepare_registry_review_cannot_be_overridden() -> None:
    prepare = _load_script("prepare_public_manifest")
    exact_revision = prepare.PUBLIC_DATASET_REVISIONS[prepare.DATASETS["reazonspeech-test"]["path"]]
    with pytest.raises(PermissionError, match="review"):
        prepare.validate_rights_for_export(
            "reazonspeech-test",
            exact_revision,
            requested="allow",
            registry_path=ROOT / "data" / "rights_registry.example.json",
        )


def test_prepare_materializes_exact_asset_only_in_external_destination(
    monkeypatch, tmp_path, capsys
) -> None:
    prepare = _load_script("prepare_public_manifest")

    class FakeDataset:
        def cast_column(self, _name, _audio):
            return self

        def shuffle(self, *, seed):
            assert seed == 20260902
            return self

        def select(self, selection):
            assert list(selection) == [0]
            return self

        def __len__(self):
            return 1

        def __iter__(self):
            return iter([{"audio": {"bytes": b"audio"}, "transcription": "参照文"}])

    class FakeAudio:
        def __init__(self, *, decode):
            assert decode is False

    class FakeSoundFile:
        @staticmethod
        def read(_source, *, dtype):
            assert dtype == "float32"
            return prepare.np.zeros(16, dtype=prepare.np.float32), 16000

        @staticmethod
        def write(path, array, rate, *, subtype):
            assert rate == 16000
            assert subtype == "PCM_16"
            Path(path).write_bytes(b"WAV")

    monkeypatch.setattr(prepare, "Audio", FakeAudio)
    monkeypatch.setattr(prepare, "load_dataset", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(prepare, "sf", FakeSoundFile)
    monkeypatch.setattr(prepare, "resample_poly", lambda array, *_args: array)
    output_dir = tmp_path / "public-data"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_public_manifest.py",
            "reazonspeech-test",
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--allow-raw-export",
        ],
    )

    assert prepare.main() == 0
    row = json.loads((output_dir / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["reference"] == "参照文"
    assert row["rightsDecision"] == "allow"
    source_digest = prepare.hashlib.sha256(b"audio").hexdigest()
    reference_digest = prepare.normalized_reference_digest("参照文")
    expected_wav = output_dir / "wav" / f"reazonspeech-test-{source_digest[:16]}.wav"
    assert row["audioPath"] == str(expected_wav.resolve())
    assert row["sourceId"] == f"audio-sha256:{source_digest}"
    assert row["groupId"] == f"audio-sha256:{source_digest}"
    assert row["nearDuplicateId"] == f"reference-sha256:{reference_digest}"
    assert row["split"] == prepare.assign_split(reference_digest, "semantic-asr-public-v1")
    assert expected_wav.read_bytes() == b"WAV"
    summary = json.loads(capsys.readouterr().out)
    assert summary["rawExport"] is True


def test_probe_redacts_references_and_hypotheses_by_default(monkeypatch, tmp_path, capsys) -> None:
    probe = _load_script("probe_second_ear")

    class FakeAdapter:
        def __init__(self, **_kwargs):
            pass

        def decode(self, _request):
            return [CandidateEvidence("candidate", "秘密の仮説", source="fake")]

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sampleId": "sample-1",
                "audioPath": str(tmp_path / "audio.wav"),
                "reference": "秘密の参照文",
                "durationSeconds": 1.25,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "Qwen3ASRAdapter", FakeAdapter)
    monkeypatch.setattr(sys, "argv", ["probe_second_ear.py", str(manifest)])

    assert probe.main() == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["sampleId"] == "sample-1"
    assert rows[-1]["rawFieldsIncluded"] is False
    assert rows[-1]["hypothesisCount"] == 1
    assert "reference" not in rows[-1]
    assert "hypotheses" not in rows[-1]
    rendered = json.dumps(rows, ensure_ascii=False)
    assert "秘密の参照文" not in rendered
    assert "秘密の仮説" not in rendered


def test_probe_local_research_opt_in_includes_sensitive_fields(
    monkeypatch, tmp_path, capsys
) -> None:
    probe = _load_script("probe_second_ear")

    class FakeAdapter:
        def __init__(self, **_kwargs):
            pass

        def decode(self, _request):
            return [CandidateEvidence("candidate", "ローカル仮説", source="fake")]

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sampleId": "sample-1",
                "audioPath": str(tmp_path / "audio.wav"),
                "reference": "ローカル参照",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "Qwen3ASRAdapter", FakeAdapter)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe_second_ear.py", str(manifest), "--local-research-output"],
    )

    assert probe.main() == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["rawFieldsIncluded"] is True
    assert rows[-1]["reference"] == "ローカル参照"
    assert rows[-1]["hypotheses"] == ["ローカル仮説"]


def test_probe_rejects_output_inside_checkout() -> None:
    probe = _load_script("probe_second_ear")
    with pytest.raises(ValueError, match="outside the repository"):
        probe.ensure_safe_output_path(ROOT / "runs" / "probe.jsonl")
