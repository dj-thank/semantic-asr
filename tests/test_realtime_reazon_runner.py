import math
import wave
from pathlib import Path

import pytest

from scripts.realtime_reazon import (
    SAMPLE_RATE,
    VAD_BUFFER_S,
    VAD_MIN_SPEECH_S,
    VAD_NUM_THREADS,
    WINDOW_SIZE,
    _drain_vad_queue,
    build_vad,
    validate_wave,
)


class FakeVadQueue:
    def __init__(self, count: int) -> None:
        self.count = count
        self.pop_calls = 0

    def empty(self) -> bool:
        return self.count == 0

    def pop(self) -> None:
        if not self.count:
            raise AssertionError("pop called on empty VAD queue")
        self.count -= 1
        self.pop_calls += 1


class FakeSherpa:
    class SileroVadModelConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class VadModelConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class VoiceActivityDetector:
        def __init__(self, config, *, buffer_size_in_seconds: float) -> None:
            self.config = config
            self.buffer_size_in_seconds = buffer_size_in_seconds


def write_wave(
    path: Path,
    *,
    channels: int = 1,
    width: int = 2,
    rate: int = SAMPLE_RATE,
    frames: int = 320,
) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        handle.writeframes(b"\0" * frames * channels * width)


def test_drain_vad_queue_consumes_each_completed_segment_once():
    vad = FakeVadQueue(3)
    assert _drain_vad_queue(vad) == 3
    assert vad.count == 0
    assert vad.pop_calls == 3
    assert _drain_vad_queue(vad) == 0
    assert vad.pop_calls == 3


def test_build_vad_matches_pinned_hayamimi_shape(tmp_path: Path):
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"fixture")

    vad = build_vad(FakeSherpa, model)
    outer = vad.config.kwargs
    silero = outer["silero_vad"].kwargs
    assert outer["sample_rate"] == SAMPLE_RATE == 16_000
    assert outer["num_threads"] == VAD_NUM_THREADS == 1
    assert vad.buffer_size_in_seconds == VAD_BUFFER_S == 30.0
    assert silero == {
        "model": str(model),
        "threshold": 0.5,
        "min_silence_duration": 0.35,
        "min_speech_duration": VAD_MIN_SPEECH_S,
        "window_size": WINDOW_SIZE,
        "max_speech_duration": 12.0,
    }


@pytest.mark.parametrize(
    ("keyword", "value", "error"),
    [
        ("threshold", True, TypeError),
        ("threshold", 0.0, ValueError),
        ("threshold", 1.1, ValueError),
        ("threshold", math.nan, ValueError),
        ("min_silence_seconds", 0.0, ValueError),
        ("min_silence_seconds", math.nan, ValueError),
        ("max_speech_seconds", 0.0, ValueError),
        ("max_speech_seconds", 30.1, ValueError),
        ("max_speech_seconds", math.inf, ValueError),
    ],
)
def test_build_vad_rejects_invalid_numeric_controls(tmp_path: Path, keyword, value, error):
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"fixture")
    with pytest.raises(error):
        build_vad(FakeSherpa, model, **{keyword: value})


def test_build_vad_requires_local_nonempty_model(tmp_path: Path):
    missing = tmp_path / "missing.onnx"
    with pytest.raises(ValueError, match="existing non-empty"):
        build_vad(FakeSherpa, missing)

    empty = tmp_path / "empty.onnx"
    empty.touch()
    with pytest.raises(ValueError, match="existing non-empty"):
        build_vad(FakeSherpa, empty)


def test_validate_wave_accepts_only_mono_pcm16_16khz(tmp_path: Path):
    valid = tmp_path / "valid.wav"
    write_wave(valid, frames=321)
    assert validate_wave(valid) == (321, SAMPLE_RATE)

    stereo = tmp_path / "stereo.wav"
    write_wave(stereo, channels=2)
    with pytest.raises(ValueError, match="mono"):
        validate_wave(stereo)

    pcm8 = tmp_path / "pcm8.wav"
    write_wave(pcm8, width=1)
    with pytest.raises(ValueError, match="PCM16"):
        validate_wave(pcm8)

    low_rate = tmp_path / "8k.wav"
    write_wave(low_rate, rate=8_000)
    with pytest.raises(ValueError, match="16 kHz"):
        validate_wave(low_rate)
