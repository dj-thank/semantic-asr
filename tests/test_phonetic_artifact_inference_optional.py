from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from semantic_asr.phonetic_runtime.artifact import (  # noqa: E402
    load_dual_ctc_artifact,
    read_dual_ctc_metadata,
    save_dual_ctc_artifact,
)
from semantic_asr.phonetic_runtime.inference import DualCTCPosteriorRuntime  # noqa: E402
from semantic_asr.phonetic_runtime.torch_model import DualPhoneMoraCTC  # noqa: E402

from _phonetic_runtime_fixture import (  # noqa: E402
    model_config,
    mora_inventory,
    phone_inventory,
    write_wav,
)


def save_fixture(tmp_path: Path):
    config = model_config()
    phone = phone_inventory()
    mora = mora_inventory()
    torch.manual_seed(7)
    model = DualPhoneMoraCTC(config, phone, mora)
    directory = tmp_path / "artifact"
    metadata = save_dual_ctc_artifact(
        directory,
        model,
        name="fixture-dual-ctc",
        revision="weights-r1",
        model_config=config,
        phone_inventory=phone,
        mora_inventory=mora,
        training_manifest_sha256="a" * 64,
        runtime_revision="runtime-r1",
    )
    return directory, metadata, model


def test_artifact_round_trip_preserves_every_tensor(tmp_path: Path) -> None:
    directory, metadata, original = save_fixture(tmp_path)

    loaded = load_dual_ctc_artifact(directory)

    assert loaded.metadata == metadata
    for name, value in original.state_dict().items():
        assert torch.equal(value.cpu(), loaded.model.state_dict()[name].cpu())


def test_artifact_detects_weight_and_metadata_tampering(tmp_path: Path) -> None:
    directory, _metadata, _model = save_fixture(tmp_path)
    weights = directory / "weights.npz"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="weights file digest mismatch"):
        read_dual_ctc_metadata(directory)

    clean_directory, _metadata, _model = save_fixture(tmp_path / "second")
    metadata_path = clean_directory / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["revision"] = "tampered"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata digest mismatch"):
        read_dual_ctc_metadata(clean_directory)


def test_runtime_emits_full_inventory_posteriors_bound_to_source_audio(tmp_path: Path) -> None:
    directory, _metadata, _model = save_fixture(tmp_path)
    audio = tmp_path / "audio.wav"
    source_digest = write_wav(audio, seconds=0.3)
    runtime = DualCTCPosteriorRuntime.from_artifact(directory)

    phone, mora = runtime.infer(
        audio,
        start_ms=40,
        end_ms=240,
        expected_source_audio_sha256=source_digest,
    )

    assert phone.source_audio_sha256 == source_digest
    assert mora.source_audio_sha256 == source_digest
    assert phone.vocabulary == phone_inventory().symbols
    assert mora.vocabulary == mora_inventory().symbols
    assert phone.frames[0].start_ms >= 40
    assert phone.frames[-1].end_ms <= 240
    assert all(
        abs(sum(value for _, value in frame.probabilities) - 1.0) < 1e-6
        for frame in (*phone.frames, *mora.frames)
    )
