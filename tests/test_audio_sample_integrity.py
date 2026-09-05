"""Sample-level regressions: no ASR model or network is needed."""

from __future__ import annotations

import wave
from unittest.mock import Mock

import pytest

from semantic_asr.api import _materialise_audio, _mono_float32
from semantic_asr.audio import decode_audio_window

np = pytest.importorskip("numpy")


def write_pcm(path, samples, *, rate=16_000):
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def test_every_signed_16_bit_sample_round_trips_exactly():
    original = np.arange(-32768, 32768, dtype=np.int16)
    path, temporary = _materialise_audio(original)
    try:
        with wave.open(str(path), "rb") as stream:
            actual = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        np.testing.assert_array_equal(actual, original)
    finally:
        temporary.unlink()


@pytest.mark.parametrize("dtype,scale", [(np.int8, 128), (np.int16, 32768), (np.int32, 2**31)])
def test_integer_pcm_is_scaled_before_downmix(dtype, scale):
    samples = np.array([0, scale // 4, -(scale // 4), scale // 2, -(scale // 2)], dtype=dtype)
    stereo = np.stack([samples, samples], axis=1)
    np.testing.assert_array_equal(
        _mono_float32(stereo, np), np.array([0, 0.25, -0.25, 0.5, -0.5], dtype=np.float32)
    )


def test_unsigned_8_bit_pcm_uses_midpoint():
    np.testing.assert_array_equal(
        _mono_float32(np.array([0, 128, 255], dtype=np.uint8), np),
        np.array([-1, 0, 127 / 128], dtype=np.float32),
    )


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.complex64, object])
def test_ambiguous_pcm_types_fail_before_materialization(dtype):
    with pytest.raises(TypeError, match="audio must be"):
        _materialise_audio(np.array([0, 1], dtype=dtype))


@pytest.mark.parametrize("sample_rate", [0, -1, True, 16000.5, 8000])
def test_arrays_cannot_be_silently_relabelled_with_a_different_rate(sample_rate):
    with pytest.raises((ValueError, TypeError)):
        _materialise_audio(np.zeros(16), sample_rate=sample_rate)


@pytest.mark.parametrize(
    "start,end", [(None, None), (0, 100), (125, 600), (900, None), (1000, 1100)]
)
def test_native_window_matches_full_decode_exactly(tmp_path, start, end):
    samples = (np.arange(16000, dtype=np.int32) * 11 - 30000).astype(np.int16)
    path = tmp_path / "native.wav"
    write_pcm(path, samples)
    fallback = Mock(side_effect=AssertionError("native PCM must not decode the entire file"))
    actual = decode_audio_window(path, start_ms=start, end_ms=end, decoder=fallback)
    expected = (samples.astype(np.float32) / 32768)[
        (start or 0) * 16 : None if end is None else end * 16
    ]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    fallback.assert_not_called()


def test_many_windows_read_only_their_own_samples(tmp_path, monkeypatch):
    path = tmp_path / "long.wav"
    write_pcm(path, np.zeros(16000 * 12, dtype=np.int16))
    requested = []
    original = wave.Wave_read.readframes

    def spy(self, count):
        requested.append(count)
        return original(self, count)

    monkeypatch.setattr(wave.Wave_read, "readframes", spy)
    decoder = Mock(side_effect=AssertionError("unexpected full decode"))
    for index in range(12):
        decode_audio_window(path, start_ms=index * 1000, end_ms=(index + 1) * 1000, decoder=decoder)
    assert requested == [16000] * 12
    assert sum(requested) == 16000 * 12  # not 12 reads of the whole 12-second recording


def test_non_native_audio_retains_upstream_resampling(tmp_path):
    path = tmp_path / "8khz.wav"
    write_pcm(path, np.zeros(8000), rate=8000)
    decoded = np.linspace(-1, 1, 16000, dtype=np.float32)
    fallback = Mock(return_value=decoded)
    np.testing.assert_array_equal(
        decode_audio_window(path, start_ms=250, end_ms=500, decoder=fallback), decoded[4000:8000]
    )
    fallback.assert_called_once_with(str(path), sampling_rate=16000)


def test_truncated_native_payload_is_not_silently_padded(tmp_path):
    path = tmp_path / "truncated.wav"
    write_pcm(path, np.zeros(16000))
    path.write_bytes(path.read_bytes()[:-32])
    with pytest.raises(ValueError, match="truncated"):
        decode_audio_window(path, decoder=Mock())


def test_float_full_scale_is_saturated_without_wrapping():
    path, temporary = _materialise_audio(np.array([-2, -1, -0.5, 0, 0.5, 1, 2], np.float32))
    try:
        with wave.open(str(path), "rb") as stream:
            samples = np.frombuffer(stream.readframes(7), dtype="<i2").tolist()
        assert samples == [-32768, -32768, -16384, 0, 16384, 32767, 32767]
    finally:
        temporary.unlink()
