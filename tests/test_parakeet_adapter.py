from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.parakeet_adapter import ParakeetJapaneseCtcAdapter


def test_parakeet_adapter_requires_explicit_artifact_and_native_budget(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"model")
    (model_dir / "tokens.txt").write_text("0 <blk>\n", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        ParakeetJapaneseCtcAdapter(model_dir, artifact_sha256="0" * 64)


def test_parakeet_decode_request_contract_is_explicit() -> None:
    request = DecodeRequest("clip.wav", language="ja", beam_size=1, hypotheses=1)
    assert request.beam_size == 1
    assert request.hypotheses == 1
