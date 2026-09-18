"""Synthetic signal contract tests, not recognition accuracy evidence."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def noise_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('jev_noise', ROOT / 'scripts' / 'collect_jev_noise_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('snr', [10.0, 20.0])
def test_noise_has_requested_snr_and_is_reproducible(noise_module, snr):
    np = pytest.importorskip('numpy')
    audio = (0.05 * np.sin(np.arange(16000) * 0.02)).astype('float32')
    before = audio.copy()
    first, report = noise_module.corrupt(audio, snr, 20260919)
    second, again = noise_module.corrupt(audio, snr, 20260919)
    assert np.array_equal(audio, before)
    assert np.array_equal(first, second)
    assert report == again
    assert abs(report['achieved_snr_db'] - snr) < 0.05
    assert report['clipped_samples'] == 0
    assert np.isfinite(first).all()
    assert np.array_equal(first * 32768, np.rint(first * 32768))
    assert first.dtype == np.float32


def test_noise_changes_with_seed(noise_module):
    np = pytest.importorskip('numpy')
    signal = np.full(16000, .05, dtype='float32')
    a, _ = noise_module.corrupt(signal, 20, 1)
    b, _ = noise_module.corrupt(signal, 20, 2)
    assert not np.array_equal(a, b)


def test_clipping_is_reported_not_hidden(noise_module):
    np = pytest.importorskip('numpy')
    signal = np.full(16000, .999, dtype='float32')
    mixed, report = noise_module.corrupt(signal, 10, 17)
    assert report['clipped_samples'] > 0
    assert mixed.max() <= 32767 / 32768
    assert mixed.min() >= -1
    assert report['achieved_snr_db'] != 10


@pytest.mark.parametrize('kind', ['empty', 'nan', 'silent', 'stereo'])
def test_invalid_audio_is_rejected(noise_module, kind):
    np = pytest.importorskip('numpy')
    signal = {'empty': np.zeros(0), 'nan': np.array([float('nan')]),
              'silent': np.zeros(10), 'stereo': np.zeros((10,2))}[kind]
    with pytest.raises(ValueError):
        noise_module.corrupt(signal, 20, 17)


def test_unregistered_snr_is_rejected(noise_module):
    np = pytest.importorskip('numpy')
    with pytest.raises(ValueError):
        noise_module.corrupt(np.ones(10), 15, 17)
