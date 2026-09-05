"""Bounded, deterministic PCM WAV loading for phonetic-only inference."""

from __future__ import annotations

import hashlib
import math
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from ..deliberation_evidence import _is_sha256
from .contracts import PhoneticRuntimeLimits


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class LoadedWaveform:
    samples: array
    sample_rate: int
    source_audio_sha256: str
    source_path: Path
    start_ms: int
    end_ms: int
    original_channels: int

    def __post_init__(self) -> None:
        if self.samples.typecode != "f":
            raise TypeError("waveform samples must use float32 array storage")
        if self.sample_rate < 1 or self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("loaded waveform timing is invalid")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        if self.original_channels < 1:
            raise ValueError("original_channels must be positive")
        if not self.samples:
            raise ValueError("loaded waveform is empty")
        if any(not math.isfinite(value) for value in self.samples):
            raise ValueError("loaded waveform contains non-finite values")

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def _strict_ms(value: object | None, *, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer number of milliseconds")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def load_pcm16_wav(
    path: str | Path,
    *,
    expected_sample_rate: int,
    limits: PhoneticRuntimeLimits | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    expected_source_audio_sha256: str | None = None,
) -> LoadedWaveform:
    """Load one bounded mono crop while hashing the complete source file.

    Only uncompressed 16-bit little-endian PCM is accepted in v1. Mono is preserved; stereo is
    downmixed by the exact arithmetic mean. Resampling is deliberately not implicit because it
    would add an unrecorded signal-processing implementation to the model identity.
    """

    limits = limits or PhoneticRuntimeLimits()
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("audio path must identify a regular file")
    if source.stat().st_size > limits.maximum_file_bytes:
        raise ValueError("audio file exceeds the configured byte limit")
    source_sha256 = sha256_file(source)
    if expected_source_audio_sha256 is not None:
        if not _is_sha256(expected_source_audio_sha256):
            raise ValueError("expected source-audio digest must be SHA-256")
        if source_sha256 != expected_source_audio_sha256:
            raise ValueError("audio file digest does not match expected source audio")

    with wave.open(str(source), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("compressed WAV input is not supported")
        channels = wav.getnchannels()
        if channels not in limits.allowed_channels:
            raise ValueError("WAV channel count is not allowed by the runtime limits")
        sample_width = wav.getsampwidth()
        if limits.require_pcm16 and sample_width != 2:
            raise ValueError("phonetic runtime requires 16-bit PCM WAV input")
        sample_rate = wav.getframerate()
        if sample_rate != expected_sample_rate:
            raise ValueError(
                f"WAV sample rate {sample_rate} does not match required {expected_sample_rate}"
            )
        frame_count = wav.getnframes()
        if frame_count < 1:
            raise ValueError("WAV contains no audio frames")
        total_ms = math.ceil(frame_count * 1000 / sample_rate)
        begin_ms = _strict_ms(start_ms, name="start_ms", default=0)
        finish_ms = _strict_ms(end_ms, name="end_ms", default=total_ms)
        if begin_ms >= finish_ms or finish_ms > total_ms:
            raise ValueError("requested audio crop is outside the WAV duration")
        if (finish_ms - begin_ms) / 1000.0 > limits.maximum_audio_seconds:
            raise ValueError("requested audio crop exceeds maximum_audio_seconds")
        begin_frame = begin_ms * sample_rate // 1000
        finish_frame = min(frame_count, math.ceil(finish_ms * sample_rate / 1000))
        requested_frames = finish_frame - begin_frame
        if requested_frames < 1:
            raise ValueError("requested audio crop contains no complete frames")
        maximum_frames = math.ceil(limits.maximum_audio_seconds * sample_rate)
        if requested_frames > maximum_frames:
            raise ValueError("requested audio crop exceeds the computed frame limit")
        wav.setpos(begin_frame)
        raw = wav.readframes(requested_frames)
        expected_bytes = requested_frames * channels * sample_width
        if len(raw) != expected_bytes:
            raise ValueError("WAV ended before the requested bounded crop was read")

    integers = array("h")
    integers.frombytes(raw)
    if sys.byteorder != "little":
        integers.byteswap()
    if len(integers) != requested_frames * channels:
        raise ValueError("decoded PCM sample count does not match WAV metadata")
    scale = 1.0 / 32768.0
    samples = array("f")
    if channels == 1:
        samples.extend(value * scale for value in integers)
    else:
        for index in range(0, len(integers), channels):
            samples.append(sum(integers[index : index + channels]) * scale / channels)
    absolute_start_ms = round(begin_frame * 1000 / sample_rate)
    absolute_end_ms = round(finish_frame * 1000 / sample_rate)
    return LoadedWaveform(
        samples=samples,
        sample_rate=sample_rate,
        source_audio_sha256=source_sha256,
        source_path=source,
        start_ms=absolute_start_ms,
        end_ms=absolute_end_ms,
        original_channels=channels,
    )
