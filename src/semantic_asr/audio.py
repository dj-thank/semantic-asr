"""Sample-preserving audio input and bounded reads for native Whisper WAV files.

Only mono, 16 kHz, signed 16-bit PCM takes the seek fast path. All other encodings
use the caller's decoder, preserving its resampling/downmixing behavior.
"""

from __future__ import annotations

import os
import wave
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any


def require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def validate_audio_span(start_ms: int | None, end_ms: int | None) -> None:
    start = 0 if start_ms is None else require_integer(start_ms, name="start_ms")
    if end_ms is not None:
        require_integer(end_ms, name="end_ms", minimum=1)
        if end_ms <= start:
            raise ValueError("end_ms must be greater than start_ms")


def pcm_to_float32(audio: Any, np: Any) -> Any:
    """Normalize integer PCM before conversion or downmixing; never infer a gain.

    Signed integers use their full negative range; uint8 uses the WAV midpoint.
    Other unsigned, boolean, complex and object arrays have no supported PCM
    interpretation and fail rather than silently corrupting the recording.
    """
    array = np.asarray(audio)
    if array.dtype.kind == "i":
        scale = float(2 ** (array.dtype.itemsize * 8 - 1))
        return array.astype(np.float32) / scale
    if array.dtype.kind == "u" and array.dtype.itemsize == 1:
        return (array.astype(np.float32) - 128.0) / 128.0
    if array.dtype.kind == "f":
        return array.astype(np.float32)
    raise TypeError("audio must be floating point, signed integer PCM, or unsigned 8-bit PCM")


def decode_audio_window(
    path: str | os.PathLike[str],
    *,
    start_ms: int | None = None,
    end_ms: int | None = None,
    decoder: Callable[..., Any],
) -> Any:
    """Read exactly the requested samples without re-decoding an entire PCM file.

    Returns the same float32 scaling as faster-whisper's signed-16-bit decoder.
    No process-global cache, mutable model state, or unbounded waveform retention
    is introduced. A malformed/truncated native payload is an error, not padding.
    """
    import numpy as np

    validate_audio_span(start_ms, end_ms)
    start = (start_ms or 0) * 16
    stop = None if end_ms is None else end_ms * 16
    with ExitStack() as stack:
        try:
            stream = stack.enter_context(wave.open(os.fspath(path), "rb"))
        except (OSError, EOFError, wave.Error):
            stream = None
        if stream is not None:
            native = (
                stream.getframerate() == 16_000
                and stream.getnchannels() == 1
                and stream.getsampwidth() == 2
                and stream.getcomptype() == "NONE"
            )
            if native:
                total = stream.getnframes()
                first = min(start, total)
                last = total if stop is None else min(stop, total)
                count = max(0, last - first)
                stream.setpos(first)
                payload = stream.readframes(count)
                if len(payload) != count * 2:
                    raise ValueError("truncated native PCM audio window")
                return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    waveform = decoder(os.fspath(path), sampling_rate=16_000)
    return waveform[start:stop]
