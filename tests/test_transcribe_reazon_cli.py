from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    path = scripts_dir / "transcribe_reazon.py"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("transcribe_reazon_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_only_one_second_ear_artifact_argument(tmp_path: Path) -> None:
    module = _load_script()
    args = module.argparse.Namespace(
        second_ear_model_dir=str(tmp_path), second_ear_model_sha256=None
    )
    with pytest.raises(ValueError, match="supplied together"):
        module.validate_second_ear_args(args)


@pytest.mark.parametrize(
    "directory,digest",
    [(None, "a" * 64), ("model", None)],
)
def test_cli_pairing_rule_is_explicit(directory, digest) -> None:
    module = _load_script()
    args = module.argparse.Namespace(
        second_ear_model_dir=directory,
        second_ear_model_sha256=digest,
    )
    with pytest.raises(ValueError, match="supplied together"):
        module.validate_second_ear_args(args)
