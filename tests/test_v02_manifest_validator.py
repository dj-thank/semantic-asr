from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "v02-experiment-manifest.json"


def _validator_module() -> ModuleType:
    path = ROOT / "scripts" / "validate_v02_manifest.py"
    spec = importlib.util.spec_from_file_location("semantic_asr_validate_v02_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load validate_v02_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _validator_module()
MANIFEST_FROM_PAYLOAD: Callable[[object], Any] = VALIDATOR.manifest_from_payload


def _payload() -> dict[str, object]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _first_record(payload: dict[str, object]) -> dict[str, object]:
    records = payload["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    return records[0]


def test_validator_accepts_canonical_five_role_manifest() -> None:
    manifest = MANIFEST_FROM_PAYLOAD(_payload())
    assert manifest.integrity_report()["counts"] == {
        "train": 1,
        "dev": 1,
        "calibration": 1,
        "test": 1,
        "regression-exposed": 1,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: _first_record(payload).__setitem__("sampleId", 7), "must be a string"),
        (lambda payload: _first_record(payload).pop("pcmSha256"), "missing required"),
        (
            lambda payload: _first_record(payload).__setitem__("sample_id", "alias"),
            "unknown record 0 field",
        ),
        (lambda payload: payload.__setitem__("splitSeed", True), "must be an integer"),
        (lambda payload: payload.__setitem__("unexpected", True), "unknown manifest field"),
    ),
)
def test_validator_rejects_coercion_missing_fields_aliases_and_unknowns(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        MANIFEST_FROM_PAYLOAD(payload)


def test_validator_main_writes_redacted_manifest_without_private_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _payload()
    first = _first_record(payload)
    first["reference"] = "公開禁止の参照文"
    first["speakerId"] = "real-person-name"
    first["metadata"] = {"localPath": "/private/audio.wav"}
    source = tmp_path / "manifest.json"
    target = tmp_path / "public.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert VALIDATOR.main([str(source), "--public-redacted-output", str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    rendered = target.read_text(encoding="utf-8")
    assert report["publicRedactedOutput"] == str(target)
    assert "公開禁止の参照文" not in rendered
    assert "real-person-name" not in rendered
    assert "/private/audio.wav" not in rendered
    redacted = json.loads(rendered)
    assert "sampleId" not in redacted["records"][0]
    assert "sampleIdSha256" in redacted["records"][0]
