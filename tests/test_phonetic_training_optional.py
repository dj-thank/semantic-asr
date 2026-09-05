from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from semantic_asr.phonetic_runtime.artifact import load_dual_ctc_artifact  # noqa: E402
from semantic_asr.phonetic_runtime.manifest import (  # noqa: E402
    PhoneticManifestRow,
    PhoneticSplitManifest,
)
from semantic_asr.phonetic_runtime.training import (  # noqa: E402
    DualCTCTrainingConfig,
    train_dual_ctc_model,
)

from _phonetic_runtime_fixture import (  # noqa: E402
    model_config,
    mora_inventory,
    phone_inventory,
    write_wav,
)


def manifest_row(tmp_path: Path, split: str, index: int) -> PhoneticManifestRow:
    path = tmp_path / f"{split}-{index}.wav"
    digest = write_wav(path, frequency=250.0 + index * 100.0, seconds=0.2)
    phone = ("k", "a") if index % 2 else ("m", "a")
    mora = ("カ",) if index % 2 else ("マ",)
    return PhoneticManifestRow(
        utterance_id=f"{split}-{index}",
        audio_path=path.resolve(),
        source_audio_sha256=digest,
        sample_rate=16_000,
        phone_symbols=phone,
        mora_symbols=mora,
        speaker_id=f"{split}-speaker-{index}",
        session_id=f"{split}-session-{index}",
        source_id=f"{split}-source-{index}",
        license_id="fixture-license",
        rights_decision="allow",
        split=split,  # type: ignore[arg-type]
    )


def test_one_epoch_training_saves_best_calibration_checkpoint(tmp_path: Path) -> None:
    manifest = PhoneticSplitManifest(
        name="fixture",
        revision="r1",
        rows=(
            manifest_row(tmp_path, "train", 1),
            manifest_row(tmp_path, "train", 2),
            manifest_row(tmp_path, "calibration", 3),
            manifest_row(tmp_path, "calibration", 4),
        ),
        source_manifest_sha256="a" * 64,
    )
    artifact_directory = tmp_path / "dual-ctc-artifact"

    result = train_dual_ctc_model(
        manifest,
        phone_inventory=phone_inventory(),
        mora_inventory=mora_inventory(),
        model_config=model_config(),
        training_config=DualCTCTrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            seed=9,
            device="cpu",
            maximum_audio_seconds=1.0,
        ),
        artifact_directory=artifact_directory,
        artifact_name="fixture",
        artifact_revision="weights-r1",
        runtime_revision="runtime-r1",
    )

    assert result.best_epoch == 1
    assert result.best_calibration_loss >= 0.0
    assert result.epoch_metrics[0].update_count == 2
    loaded = load_dual_ctc_artifact(artifact_directory)
    assert loaded.metadata.digest == result.artifact.digest
