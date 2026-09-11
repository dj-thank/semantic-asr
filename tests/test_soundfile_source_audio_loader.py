from __future__ import annotations

from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")
soundfile = pytest.importorskip("soundfile")

from semantic_asr.phonetic_dataset import file_sha256  # noqa: E402
from semantic_asr.phonetic_feature_export import SoundFileSourceAudioLoader  # noqa: E402


def test_loader_preserves_mono_sample_rate_and_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "mono.wav"
    samples = numpy.linspace(-0.5, 0.5, 160, dtype=numpy.float32)
    soundfile.write(path, samples, 16_000, subtype="FLOAT")

    loaded = SoundFileSourceAudioLoader().load(path)

    assert loaded.sample_rate == 16_000
    assert loaded.file_sha256 == file_sha256(path)
    assert loaded.source_name == "mono.wav"
    assert len(loaded.samples) == 160
    assert loaded.samples[0] == pytest.approx(float(samples[0]))


def test_loader_rejects_implicit_stereo_downmix(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    samples = numpy.zeros((160, 2), dtype=numpy.float32)
    soundfile.write(path, samples, 16_000, subtype="FLOAT")

    with pytest.raises(ValueError, match="explicitly mono"):
        SoundFileSourceAudioLoader().load(path)
