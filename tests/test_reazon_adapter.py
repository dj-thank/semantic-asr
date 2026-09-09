"""Bounded opt-in backend contracts; mocked native decode does not measure accuracy."""

import sys
import types
import wave

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.reazon_adapter import ReazonSpeechK2Adapter
from semantic_asr.revisions import sha256_artifact


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    folder = tmp_path / "model"
    folder.mkdir()
    for name in (
        "encoder-fixture.int8.onnx",
        "decoder-fixture.int8.onnx",
        "joiner-fixture.int8.onnx",
        "tokens.txt",
    ):
        (folder / name).write_text(name, encoding="utf-8")
    stream = types.SimpleNamespace(result=types.SimpleNamespace(text="えー、えー学校を行く"))
    stream.accept_waveform = lambda rate, data: setattr(stream, "samples", len(data))
    native = types.SimpleNamespace(create_stream=lambda: stream, decode_stream=lambda _: None)
    fake = types.SimpleNamespace(
        OfflineRecognizer=types.SimpleNamespace(from_transducer=lambda **kw: native)
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    monkeypatch.setattr(
        "semantic_asr.reazon_adapter.importlib.metadata.version", lambda _: "fixture"
    )
    instance = ReazonSpeechK2Adapter(folder, artifact_sha256=sha256_artifact(folder))
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(b"\0\0" * 32000)
    return instance, audio, stream, folder


def test_preserves_observed_text_without_faking_confidence(adapter):
    model, path, stream, _ = adapter
    candidates = model.decode(
        DecodeRequest(str(path), beam_size=4, hypotheses=1, start_ms=250, end_ms=750)
    )
    assert candidates[0].text == "えー、えー学校を行く"
    assert candidates[0].acoustic is None and candidates[0].avg_logprob is None
    assert candidates[0].hypothesis_count == 1
    assert stream.samples == 8000
    assert candidates[0].metadata["sampleCount"] == 8000


@pytest.mark.parametrize(
    "options",
    [
        {"language": "en"},
        {"initial_prompt": "正解"},
        {"hotwords": ("正解",)},
        {"return_timestamps": True},
    ],
)
def test_unsupported_conditioning_is_not_silently_ignored(adapter, options):
    model, path, _, _ = adapter
    with pytest.raises(ValueError):
        model.decode(DecodeRequest(str(path), beam_size=4, hypotheses=1, **options))


def test_empty_decode_is_not_invented_speech(adapter):
    model, path, stream, _ = adapter
    stream.result.text = ""
    assert model.decode(DecodeRequest(str(path), beam_size=4, hypotheses=1)) == []


def test_model_hash_mismatch_rejected_before_native_load(adapter):
    _, _, _, folder = adapter
    with pytest.raises(ValueError):
        ReazonSpeechK2Adapter(folder, artifact_sha256="0" * 64)


def test_native_search_budget_is_explicit(adapter):
    model, path, _, _ = adapter
    with pytest.raises(ValueError, match="beam"):
        model.decode(DecodeRequest(str(path), beam_size=1, hypotheses=1))
