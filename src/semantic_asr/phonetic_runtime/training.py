"""Optional deterministic PyTorch trainer for source-audio-only phone/mora CTC heads."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import sha256_json
from .artifact import save_dual_ctc_artifact
from .audio import load_pcm16_wav
from .contracts import (
    DualCTCArtifactMetadata,
    DualCTCModelConfig,
    PhoneticInventory,
    PhoneticRuntimeLimits,
)
from .manifest import PhoneticManifestRow, PhoneticSplitManifest


@dataclass(frozen=True, slots=True)
class DualCTCTrainingConfig:
    epochs: int = 8
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    phone_loss_weight: float = 1.0
    mora_loss_weight: float = 1.0
    seed: int = 20260905
    device: str = "cpu"
    maximum_audio_seconds: float = 35.0
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        for name in (
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "phone_loss_weight",
            "mora_loss_weight",
            "maximum_audio_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, numeric)
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if self.maximum_audio_seconds <= 0.0:
            raise ValueError("maximum_audio_seconds must be positive")
        if self.phone_loss_weight + self.mora_loss_weight <= 0.0:
            raise ValueError("at least one CTC loss weight must be positive")
        if not self.device:
            raise ValueError("training device is required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class TrainingEpochMetrics:
    epoch: int
    training_total_loss: float
    training_phone_loss: float
    training_mora_loss: float
    validation_total_loss: float
    validation_phone_loss: float
    validation_mora_loss: float
    update_count: int

    def __post_init__(self) -> None:
        if isinstance(self.epoch, bool) or self.epoch < 1:
            raise ValueError("epoch must be positive")
        if isinstance(self.update_count, bool) or self.update_count < 1:
            raise ValueError("update_count must be positive")
        for name in (
            "training_total_loss",
            "training_phone_loss",
            "training_mora_loss",
            "validation_total_loss",
            "validation_phone_loss",
            "validation_mora_loss",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class DualCTCTrainingResult:
    artifact: DualCTCArtifactMetadata
    training_config: DualCTCTrainingConfig
    manifest_digest: str
    epoch_metrics: tuple[TrainingEpochMetrics, ...]
    best_epoch: int
    best_validation_loss: float

    def __post_init__(self) -> None:
        if not self.epoch_metrics:
            raise ValueError("training result requires epoch metrics")
        if self.best_epoch not in {row.epoch for row in self.epoch_metrics}:
            raise ValueError("best_epoch is absent from epoch metrics")
        if not math.isfinite(self.best_validation_loss):
            raise ValueError("best_validation_loss must be finite")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "artifactDigest": self.artifact.digest,
                "trainingConfigDigest": self.training_config.digest,
                "manifestDigest": self.manifest_digest,
                "epochMetrics": [asdict(row) for row in self.epoch_metrics],
                "bestEpoch": self.best_epoch,
                "bestCalibrationLoss": self.best_validation_loss,
            }
        )

    def write_report(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifactDigest": self.artifact.digest,
            "trainingConfig": asdict(self.training_config),
            "trainingConfigDigest": self.training_config.digest,
            "manifestDigest": self.manifest_digest,
            "epochMetrics": [asdict(row) for row in self.epoch_metrics],
            "bestEpoch": self.best_epoch,
            "bestCalibrationLoss": self.best_validation_loss,
            "trainingResultDigest": self.digest,
        }
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return destination


def _batch_order(size: int, *, epoch: int, seed: int, shuffle: bool) -> tuple[int, ...]:
    indexes = list(range(size))
    if shuffle:
        random.Random(f"{seed}:{epoch}").shuffle(indexes)
    return tuple(indexes)


def _batch_rows(
    rows: tuple[PhoneticManifestRow, ...],
    *,
    batch_size: int,
    epoch: int,
    seed: int,
    shuffle: bool,
) -> tuple[tuple[PhoneticManifestRow, ...], ...]:
    order = _batch_order(len(rows), epoch=epoch, seed=seed, shuffle=shuffle)
    return tuple(
        tuple(rows[index] for index in order[start : start + batch_size])
        for start in range(0, len(order), batch_size)
    )


def _load_batch(
    rows: tuple[PhoneticManifestRow, ...],
    *,
    model_config: DualCTCModelConfig,
    phone_inventory: PhoneticInventory,
    mora_inventory: PhoneticInventory,
    maximum_audio_seconds: float,
    device: str,
):
    import torch
    from torch.nn import functional as F

    limits = PhoneticRuntimeLimits(maximum_audio_seconds=maximum_audio_seconds)
    waveforms = []
    lengths = []
    phone_targets: list[int] = []
    phone_lengths = []
    mora_targets: list[int] = []
    mora_lengths = []
    for row in rows:
        waveform = load_pcm16_wav(
            row.audio_path,
            expected_sample_rate=model_config.frontend.sample_rate,
            limits=limits,
            expected_source_audio_sha256=row.source_audio_sha256,
        )
        tensor = torch.frombuffer(waveform.samples, dtype=torch.float32).clone()
        waveforms.append(tensor)
        lengths.append(tensor.numel())
        phone = phone_inventory.encode(row.phone_symbols)
        mora = mora_inventory.encode(row.mora_symbols)
        phone_targets.extend(phone)
        phone_lengths.append(len(phone))
        mora_targets.extend(mora)
        mora_lengths.append(len(mora))
    maximum = max(lengths)
    padded = torch.stack([F.pad(row, (0, maximum - row.numel())) for row in waveforms])
    return (
        padded.to(device),
        torch.tensor(lengths, dtype=torch.long, device=device),
        torch.tensor(phone_targets, dtype=torch.long, device=device),
        torch.tensor(phone_lengths, dtype=torch.long, device=device),
        torch.tensor(mora_targets, dtype=torch.long, device=device),
        torch.tensor(mora_lengths, dtype=torch.long, device=device),
    )


def _run_split(
    model,
    rows: tuple[PhoneticManifestRow, ...],
    *,
    training: bool,
    epoch: int,
    optimizer,
    training_config: DualCTCTrainingConfig,
    model_config: DualCTCModelConfig,
    phone_inventory: PhoneticInventory,
    mora_inventory: PhoneticInventory,
) -> tuple[float, float, float, int]:
    import torch

    from .torch_model import multitask_ctc_loss

    if not rows:
        raise ValueError("training/validation split must not be empty")
    batches = _batch_rows(
        rows,
        batch_size=training_config.batch_size,
        epoch=epoch,
        seed=training_config.seed,
        shuffle=training,
    )
    totals = [0.0, 0.0, 0.0]
    updates = 0
    model.train(training)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in batches:
            (
                waveforms,
                input_lengths,
                phone_targets,
                phone_target_lengths,
                mora_targets,
                mora_target_lengths,
            ) = _load_batch(
                batch,
                model_config=model_config,
                phone_inventory=phone_inventory,
                mora_inventory=mora_inventory,
                maximum_audio_seconds=training_config.maximum_audio_seconds,
                device=training_config.device,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(waveforms, input_lengths)
            if torch.any(phone_target_lengths > output.output_lengths):
                raise ValueError("phone target is longer than the available CTC frame grid")
            if torch.any(mora_target_lengths > output.output_lengths):
                raise ValueError("mora target is longer than the available CTC frame grid")
            total, phone_loss, mora_loss = multitask_ctc_loss(
                output,
                phone_targets=phone_targets,
                phone_target_lengths=phone_target_lengths,
                mora_targets=mora_targets,
                mora_target_lengths=mora_target_lengths,
                phone_blank_id=phone_inventory.blank_id,
                mora_blank_id=mora_inventory.blank_id,
                phone_weight=training_config.phone_loss_weight,
                mora_weight=training_config.mora_loss_weight,
            )
            if not torch.isfinite(total):
                raise ValueError("dual CTC training produced a non-finite loss")
            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                updates += 1
            totals[0] += float(total.detach().cpu())
            totals[1] += float(phone_loss.detach().cpu())
            totals[2] += float(mora_loss.detach().cpu())
    divisor = len(batches)
    return totals[0] / divisor, totals[1] / divisor, totals[2] / divisor, updates


def train_dual_ctc_model(
    manifest: PhoneticSplitManifest,
    *,
    phone_inventory: PhoneticInventory,
    mora_inventory: PhoneticInventory,
    model_config: DualCTCModelConfig,
    training_config: DualCTCTrainingConfig,
    artifact_directory: str | Path,
    artifact_name: str,
    artifact_revision: str,
    runtime_revision: str,
) -> DualCTCTrainingResult:
    import torch

    from .artifact import load_dual_ctc_artifact
    from .torch_model import DualPhoneMoraCTC

    manifest.validate_inventories(phone_inventory, mora_inventory)
    if {row.sample_rate for row in manifest.rows} != {model_config.frontend.sample_rate}:
        raise ValueError("manifest sample rate does not match the model frontend")
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        if training_config.device != "cpu":
            raise
    model = DualPhoneMoraCTC(model_config, phone_inventory, mora_inventory).to(
        training_config.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_rows = manifest.rows_for("train")
    validation_rows = manifest.rows_for("validation")
    history: list[TrainingEpochMetrics] = []
    best_state: dict[str, object] | None = None
    best_epoch = 0
    best_loss = math.inf
    for epoch in range(1, training_config.epochs + 1):
        train_total, train_phone, train_mora, updates = _run_split(
            model,
            train_rows,
            training=True,
            epoch=epoch,
            optimizer=optimizer,
            training_config=training_config,
            model_config=model_config,
            phone_inventory=phone_inventory,
            mora_inventory=mora_inventory,
        )
        validation_total, validation_phone, validation_mora, _ = _run_split(
            model,
            validation_rows,
            training=False,
            epoch=epoch,
            optimizer=optimizer,
            training_config=training_config,
            model_config=model_config,
            phone_inventory=phone_inventory,
            mora_inventory=mora_inventory,
        )
        history.append(
            TrainingEpochMetrics(
                epoch=epoch,
                training_total_loss=train_total,
                training_phone_loss=train_phone,
                training_mora_loss=train_mora,
                validation_total_loss=validation_total,
                validation_phone_loss=validation_phone,
                validation_mora_loss=validation_mora,
                update_count=updates,
            )
        )
        if validation_total < best_loss:
            best_loss = validation_total
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("dual CTC trainer did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    metadata = save_dual_ctc_artifact(
        artifact_directory,
        model,
        name=artifact_name,
        revision=artifact_revision,
        model_config=model_config,
        phone_inventory=phone_inventory,
        mora_inventory=mora_inventory,
        training_manifest_sha256=manifest.digest,
        runtime_revision=runtime_revision,
    )
    loaded = load_dual_ctc_artifact(artifact_directory, device=training_config.device)
    if loaded.metadata.digest != metadata.digest:
        raise ValueError("saved dual CTC artifact did not round-trip")
    return DualCTCTrainingResult(
        artifact=metadata,
        training_config=training_config,
        manifest_digest=manifest.digest,
        epoch_metrics=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
    )
