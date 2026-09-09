from types import SimpleNamespace

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.native_whisper_adapter import NativeWhisperAdapter


def test_preserves_native_utterance_without_inventing_window_score(monkeypatch):
    import sys

    np = pytest.importorskip("numpy")

    import semantic_asr.native_whisper_adapter as module

    calls = []
    segment = SimpleNamespace(
        text=" あの、VPNではありません。",
        start=0,
        end=0.5,
        avg_logprob=-0.3,
        no_speech_prob=0.01,
        tokens=[1, 2],
    )

    def decode(audio, **kwargs):
        calls.append(kwargs)
        return iter([segment]), None

    monkeypatch.setitem(sys.modules, "faster_whisper.audio", SimpleNamespace(decode_audio=None))
    monkeypatch.setattr(module, "decode_audio_window", lambda *a, **k: np.zeros(16000))
    adapter = object.__new__(NativeWhisperAdapter)
    adapter.model = SimpleNamespace(transcribe=decode)
    adapter.model_name = "fixture"
    adapter.model_revision = "a" * 40
    adapter.model_artifact_sha256 = None
    adapter.runtime_revision = "test"
    adapter.compute_type, adapter.device, adapter.cpu_threads = "int8", "cpu", 2
    result = adapter.decode(DecodeRequest("fixture.wav", hypotheses=1, start_ms=1000))[0]
    assert result.text == "あの、VPNではありません。"
    assert result.acoustic is None and result.beam_confidence is None
    assert result.metadata["nativeOutputPreserved"] is True
    assert result.metadata["utteranceSpans"][0]["startMs"] == 1000
    assert calls[0]["condition_on_previous_text"] is False
    assert result.metadata["nativeSegmentScores"][0]["avgLogprob"] == -0.3


def test_native_profile_has_one_hypothesis_and_no_borrowed_calibration():
    from semantic_asr.api import runtime_profile

    profile = runtime_profile("whisper-native-cpu-v1")
    assert profile.hypotheses == 1 and profile.confidence_calibration is None
    with pytest.raises(ValueError, match="one hypothesis"):
        object.__new__(NativeWhisperAdapter).decode(DecodeRequest("unused.wav"))


def test_native_cli_validates_local_artifact_pair_before_loading():
    from semantic_asr.run_cli import build_parser, build_profile_adapters

    args = build_parser().parse_args(
        [
            "unused.wav",
            "--profile",
            "whisper-native-cpu-v1",
            "--whisper-model-dir",
            "model",
        ]
    )
    with pytest.raises(ValueError, match="both model directory"):
        build_profile_adapters(args)


def test_native_cli_loads_the_requested_backend(monkeypatch):
    import semantic_asr.native_whisper_adapter as module
    from semantic_asr.run_cli import build_parser, build_profile_adapters

    captured = []
    model = object()
    monkeypatch.setattr(module, "NativeWhisperAdapter", lambda **kw: captured.append(kw) or model)
    args = build_parser().parse_args(
        [
            "unused.wav",
            "--profile",
            "whisper-native-cpu-v1",
            "--whisper-model-dir",
            "model",
            "--whisper-artifact-sha256",
            "a" * 64,
        ]
    )
    assert build_profile_adapters(args) == (model, None)
    assert captured[0]["artifact_sha256"] == "a" * 64
