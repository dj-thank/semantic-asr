from __future__ import annotations

import hashlib
import math
import wave
from pathlib import Path

from semantic_asr.phonetic_runtime.contracts import (
    DualCTCModelConfig,
    LogMelFrontendConfig,
    PhoneticInventory,
)


def phone_inventory() -> PhoneticInventory:
    return PhoneticInventory(
        kind="phone",
        symbols=("<blk>", "a", "i", "k", "m"),
        blank_symbol="<blk>",
        language="ja",
        revision="fixture-phone-r1",
    )


def mora_inventory() -> PhoneticInventory:
    return PhoneticInventory(
        kind="mora",
        symbols=("<blk>", "ア", "イ", "カ", "マ"),
        blank_symbol="<blk>",
        language="ja",
        revision="fixture-mora-r1",
    )


def model_config() -> DualCTCModelConfig:
    return DualCTCModelConfig(
        frontend=LogMelFrontendConfig(
            sample_rate=16_000,
            n_fft=64,
            window_length=64,
            hop_length=16,
            n_mels=16,
            frequency_min=20.0,
            frequency_max=7_000.0,
        ),
        hidden_dimension=32,
        encoder_layers=1,
        attention_heads=4,
        feedforward_dimension=64,
        convolution_kernel=3,
        subsampling_layers=1,
        dropout=0.0,
        maximum_frames=1_000,
        architecture_revision="fixture-dual-ctc-r1",
    )


def write_wav(
    path: Path,
    *,
    frequency: float = 440.0,
    seconds: float = 0.25,
    channels: int = 1,
    sample_rate: int = 16_000,
) -> str:
    frame_count = round(seconds * sample_rate)
    samples = []
    for index in range(frame_count):
        value = int(12_000 * math.sin(2.0 * math.pi * frequency * index / sample_rate))
        samples.extend([value] * channels)
    raw = bytearray()
    for value in samples:
        raw.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(raw))
    return hashlib.sha256(path.read_bytes()).hexdigest()
