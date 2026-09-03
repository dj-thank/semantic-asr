from __future__ import annotations

import pytest

from semantic_asr.adapters import pad_features_to_window, window_frames

np = pytest.importorskip("numpy")


def test_short_features_are_zero_padded_to_the_window() -> None:
    features = np.ones((128, 80), dtype=np.float32)
    padded = pad_features_to_window(features, 3000)
    assert padded.shape == (1, 128, 3000)
    assert padded[0, :, :80].sum() == 128 * 80
    assert padded[0, :, 80:].sum() == 0


def test_long_features_are_trimmed_and_exact_features_untouched() -> None:
    features = np.ones((1, 128, 3500), dtype=np.float32)
    assert pad_features_to_window(features, 3000).shape == (1, 128, 3000)
    exact = np.ones((1, 128, 3000), dtype=np.float32)
    assert pad_features_to_window(exact, 3000) is exact


def test_window_frames_falls_back_to_whisper_default() -> None:
    class _Extractor:
        nb_max_frames = 1500

    class _Model:
        feature_extractor = _Extractor()

    assert window_frames(_Model()) == 1500
    assert window_frames(object()) == 3000
    with pytest.raises(ValueError):
        pad_features_to_window(np.ones((128, 10)), 0)
