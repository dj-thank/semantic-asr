from __future__ import annotations

import wave

import pytest

from semantic_asr.phonetic_runtime.audio import load_pcm16_wav
from semantic_asr.phonetic_runtime.contracts import (
    DualCTCModelConfig,
    PhoneticInventory,
    PhoneticRuntimeLimits,
)

from _phonetic_runtime_fixture import model_config, phone_inventory, write_wav


def test_inventory_round_trip_and_blank_contract() -> None:
    inventory = phone_inventory()
    encoded = inventory.encode(("k", "a", "i"))

    assert inventory.decode(encoded) == ("k", "a", "i")
    assert inventory.blank_id == 0
    assert inventory.digest

    with pytest.raises(ValueError, match="blank"):
        inventory.encode(("<blk>",))


def test_inventory_rejects_duplicate_or_misplaced_blank() -> None:
    with pytest.raises(ValueError, match="index zero"):
        PhoneticInventory(
            kind="phone",
            symbols=("a", "<blk>"),
            revision="bad",
        )

    with pytest.raises(ValueError, match="unique"):
        PhoneticInventory(
            kind="phone",
            symbols=("<blk>", "a", "a"),
            revision="bad",
        )


def test_bounded_wav_reader_hashes_full_source_and_crops_absolute_time(tmp_path) -> None:
    path = tmp_path / "audio.wav"
    digest = write_wav(path, seconds=0.5)

    waveform = load_pcm16_wav(
        path,
        expected_sample_rate=16_000,
        start_ms=100,
        end_ms=300,
        expected_source_audio_sha256=digest,
    )

    assert waveform.source_audio_sha256 == digest
    assert waveform.start_ms == 100
    assert waveform.end_ms == 300
    assert len(waveform.samples) == 3_200
    assert waveform.original_channels == 1


def test_stereo_is_deterministically_downmixed(tmp_path) -> None:
    path = tmp_path / "stereo.wav"
    write_wav(path, channels=2)

    waveform = load_pcm16_wav(path, expected_sample_rate=16_000)

    assert waveform.original_channels == 2
    assert len(waveform.samples) == 4_000
    assert any(abs(value) > 0.01 for value in waveform.samples)


def test_reader_rejects_wrong_rate_duration_and_sample_width(tmp_path) -> None:
    wrong_rate = tmp_path / "wrong-rate.wav"
    write_wav(wrong_rate, sample_rate=8_000)
    with pytest.raises(ValueError, match="sample rate"):
        load_pcm16_wav(wrong_rate, expected_sample_rate=16_000)

    long_audio = tmp_path / "long.wav"
    write_wav(long_audio, seconds=0.4)
    with pytest.raises(ValueError, match="maximum_audio_seconds"):
        load_pcm16_wav(
            long_audio,
            expected_sample_rate=16_000,
            limits=PhoneticRuntimeLimits(maximum_audio_seconds=0.1),
        )

    eight_bit = tmp_path / "8bit.wav"
    with wave.open(str(eight_bit), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(16_000)
        wav.writeframes(b"\x80" * 100)
    with pytest.raises(ValueError, match="16-bit PCM"):
        load_pcm16_wav(eight_bit, expected_sample_rate=16_000)


def test_model_config_rejects_even_convolution_kernel() -> None:
    valid = model_config()

    with pytest.raises(ValueError, match="must be odd"):
        DualCTCModelConfig(
            frontend=valid.frontend,
            hidden_dimension=valid.hidden_dimension,
            encoder_layers=valid.encoder_layers,
            attention_heads=valid.attention_heads,
            feedforward_dimension=valid.feedforward_dimension,
            convolution_kernel=4,
            subsampling_layers=valid.subsampling_layers,
            dropout=valid.dropout,
            maximum_frames=valid.maximum_frames,
            architecture_revision=valid.architecture_revision,
        )
