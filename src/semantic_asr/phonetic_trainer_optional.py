"""Optional PyTorch trainer for the shared phone/mora CTC head."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import sha256_json
from .phonetic_dataset import (
    PhoneticFeatureManifest,
    load_feature_array,
)
from .phonetic_heads_optional import JointPhoneMoraCTCHead
from .phonetic_training import (
    JointPhoneticArtifact,
    JointPhoneticHeadConfig,
    PhoneticTrainingManifest,
    PhoneticValidationMetrics,
)

try:  # pragma: no cover - optional dependency
    import torch
    from safetensors.torch import save_file as save_safetensors
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any
    save_safetensors = None


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PhoneticOptimizationConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    random_seed: int = 0
    device: str = "cpu"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer")
        for name in ("learning_rate", "weight_decay", "gradient_clip_norm"):
            value = _strict_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if not self.device:
            raise ValueError("device is required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticSequenceCalibration:
    phone_threshold: float
    mora_threshold: float
    target_true_accept_rate: float
    calibration_manifest_sha256: str
    revision: str
    phone_false_accept_rate: float
    mora_false_accept_rate: float
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "phone_threshold",
            "mora_threshold",
            "target_true_accept_rate",
            "phone_false_accept_rate",
            "mora_false_accept_rate",
        ):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if not 0.0 < self.target_true_accept_rate <= 1.0:
            raise ValueError("target_true_accept_rate must be in (0, 1]")
        for name in ("phone_false_accept_rate", "mora_false_accept_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if len(self.calibration_manifest_sha256) != 64:
            raise ValueError("calibration_manifest_sha256 must be a SHA-256 value")
        if not self.revision:
            raise ValueError("calibration revision is required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticTrainingHistory:
    epoch_total_loss: tuple[float, ...]
    epoch_phone_ctc: tuple[float, ...]
    epoch_mora_ctc: tuple[float, ...]
    optimization_config_digest: str

    def __post_init__(self) -> None:
        if not self.epoch_total_loss:
            raise ValueError("training history must contain epochs")
        if not (
            len(self.epoch_total_loss)
            == len(self.epoch_phone_ctc)
            == len(self.epoch_mora_ctc)
        ):
            raise ValueError("training history epoch lengths differ")
        for value in (
            *self.epoch_total_loss,
            *self.epoch_phone_ctc,
            *self.epoch_mora_ctc,
        ):
            _strict_float(value, name="training history value")
        if len(self.optimization_config_digest) != 64:
            raise ValueError("optimization_config_digest must be a SHA-256 value")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticEvaluationResult:
    metrics: PhoneticValidationMetrics
    calibration: PhoneticSequenceCalibration | None
    phone_positive_scores: tuple[float, ...]
    phone_negative_scores: tuple[float, ...]
    mora_positive_scores: tuple[float, ...]
    mora_negative_scores: tuple[float, ...]
    manifest_sha256: str

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "metrics": asdict(self.metrics),
                "calibration": None if self.calibration is None else asdict(self.calibration),
                "phonePositiveScores": self.phone_positive_scores,
                "phoneNegativeScores": self.phone_negative_scores,
                "moraPositiveScores": self.mora_positive_scores,
                "moraNegativeScores": self.mora_negative_scores,
                "manifestSha256": self.manifest_sha256,
            }
        )


def _require_torch() -> None:
    if torch is None or save_safetensors is None:
        raise RuntimeError("joint phonetic training requires PyTorch and safetensors")


def _edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_value in enumerate(left, 1):
        current = [row_index]
        for column_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _collapse_greedy(values: Tensor, *, blank_index: int) -> tuple[int, ...]:
    output = []
    previous = None
    for raw in values.tolist():
        value = int(raw)
        if value != blank_index and value != previous:
            output.append(value)
        previous = value
    return tuple(output)


def _hard_negative(
    target: tuple[int, ...],
    *,
    vocabulary_size: int,
    blank_index: int,
) -> tuple[int, ...] | None:
    choices = tuple(index for index in range(vocabulary_size) if index != blank_index)
    if len(choices) < 2:
        return None
    output = list(target)
    position = len(output) // 2
    current = output[position]
    replacement_index = (choices.index(current) + 1) % len(choices)
    output[position] = choices[replacement_index]
    return tuple(output)


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        return 0.5
    wins = 0.0
    comparisons = 0
    for left in positive:
        for right in negative:
            wins += float(left > right) + 0.5 * float(left == right)
            comparisons += 1
    return wins / comparisons


def _threshold(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    true_accept_rate: float,
) -> tuple[float, float]:
    if not positive:
        raise ValueError("threshold fitting requires positive scores")
    ordered = sorted(positive)
    index = max(0, min(len(ordered) - 1, math.floor((1.0 - true_accept_rate) * len(ordered))))
    threshold = ordered[index]
    false_accept = (
        sum(value >= threshold for value in negative) / len(negative) if negative else 0.0
    )
    return threshold, false_accept


def _batches(length: int, batch_size: int, *, seed: int, shuffle: bool) -> list[list[int]]:
    indexes = list(range(length))
    if shuffle:
        random.Random(seed).shuffle(indexes)
    return [indexes[start : start + batch_size] for start in range(0, length, batch_size)]


def _collate(manifest: PhoneticFeatureManifest, indexes: Sequence[int], *, device: str):
    _require_torch()
    arrays = [load_feature_array(manifest, manifest.items[index]) for index in indexes]
    dimensions = {array.shape[1] for array in arrays}
    if len(dimensions) != 1:
        raise ValueError("batch mixes feature dimensions")
    maximum_frames = max(array.shape[0] for array in arrays)
    features = torch.zeros(
        (len(arrays), maximum_frames, next(iter(dimensions))),
        dtype=torch.float32,
        device=device,
    )
    lengths = []
    phone_targets = []
    phone_lengths = []
    mora_targets = []
    mora_lengths = []
    for index, (array, item) in enumerate(
        zip(arrays, (manifest.items[value] for value in indexes), strict=True)
    ):
        tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
        features[index, : tensor.shape[0]] = tensor
        lengths.append(tensor.shape[0])
        phone_targets.extend(item.phone_targets)
        phone_lengths.append(len(item.phone_targets))
        mora_targets.extend(item.mora_targets)
        mora_lengths.append(len(item.mora_targets))
    return {
        "features": features,
        "input_lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "phone_targets": torch.tensor(phone_targets, dtype=torch.long, device=device),
        "phone_target_lengths": torch.tensor(phone_lengths, dtype=torch.long, device=device),
        "mora_targets": torch.tensor(mora_targets, dtype=torch.long, device=device),
        "mora_target_lengths": torch.tensor(mora_lengths, dtype=torch.long, device=device),
    }


def train_joint_phonetic_head(
    model: JointPhoneMoraCTCHead,
    manifest: PhoneticFeatureManifest,
    *,
    optimization: PhoneticOptimizationConfig | None = None,
) -> PhoneticTrainingHistory:
    _require_torch()
    optimization = optimization or PhoneticOptimizationConfig()
    if manifest.items[0].feature_dimension != model.config.input_dimension:
        raise ValueError("manifest feature dimension differs from the head configuration")
    torch.manual_seed(optimization.random_seed)
    random.seed(optimization.random_seed)
    if optimization.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(optimization.device)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    total_history = []
    phone_history = []
    mora_history = []
    for epoch in range(optimization.epochs):
        total_sum = 0.0
        phone_sum = 0.0
        mora_sum = 0.0
        batches = _batches(
            len(manifest.items),
            optimization.batch_size,
            seed=optimization.random_seed + epoch,
            shuffle=True,
        )
        for indexes in batches:
            batch = _collate(manifest, indexes, device=str(device))
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["features"])
            loss = model.loss(
                output,
                input_lengths=batch["input_lengths"],
                phone_targets=batch["phone_targets"],
                phone_target_lengths=batch["phone_target_lengths"],
                mora_targets=batch["mora_targets"],
                mora_target_lengths=batch["mora_target_lengths"],
            )
            if not torch.isfinite(loss.total):
                raise ValueError("joint phonetic training produced a non-finite loss")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimization.gradient_clip_norm)
            optimizer.step()
            total_sum += float(loss.total.detach().cpu())
            phone_sum += float(loss.phone_ctc.detach().cpu())
            mora_sum += float(loss.mora_ctc.detach().cpu())
        denominator = max(1, len(batches))
        total_history.append(total_sum / denominator)
        phone_history.append(phone_sum / denominator)
        mora_history.append(mora_sum / denominator)
    return PhoneticTrainingHistory(
        epoch_total_loss=tuple(total_history),
        epoch_phone_ctc=tuple(phone_history),
        epoch_mora_ctc=tuple(mora_history),
        optimization_config_digest=optimization.digest,
    )


def _sequence_score(logits: Tensor, target: tuple[int, ...], *, blank_index: int) -> float:
    _require_torch()
    if not target:
        raise ValueError("CTC sequence score requires a non-empty target")
    log_probs = torch.log_softmax(logits, dim=-1).unsqueeze(1)
    target_tensor = torch.tensor(target, dtype=torch.long, device=logits.device)
    loss = torch.nn.functional.ctc_loss(
        log_probs,
        target_tensor,
        torch.tensor([logits.shape[0]], dtype=torch.long, device=logits.device),
        torch.tensor([len(target)], dtype=torch.long, device=logits.device),
        blank=blank_index,
        reduction="sum",
        zero_infinity=True,
    )
    return -float(loss.detach().cpu()) / len(target)


def evaluate_joint_phonetic_head(
    model: JointPhoneMoraCTCHead,
    manifest: PhoneticFeatureManifest,
    *,
    device: str = "cpu",
    fit_calibration: bool = False,
    target_true_accept_rate: float = 0.95,
    calibration_revision: str = "phonetic-sequence-calibration-v1",
) -> PhoneticEvaluationResult:
    _require_torch()
    if not 0.0 < target_true_accept_rate <= 1.0:
        raise ValueError("target_true_accept_rate must be in (0, 1]")
    runtime = torch.device(device)
    model.to(runtime)
    model.eval()
    phone_errors = 0
    phone_total = 0
    mora_errors = 0
    mora_total = 0
    phone_positive = []
    phone_negative = []
    mora_positive = []
    mora_negative = []
    with torch.inference_mode():
        for item_index, item in enumerate(manifest.items):
            batch = _collate(manifest, (item_index,), device=str(runtime))
            output = model(batch["features"])
            phone_logits = output.phone_logits[0, : item.frame_count]
            mora_logits = output.mora_logits[0, : item.frame_count]
            phone_prediction = _collapse_greedy(
                phone_logits.argmax(dim=-1),
                blank_index=model.config.phone_inventory.blank_index,
            )
            mora_prediction = _collapse_greedy(
                mora_logits.argmax(dim=-1),
                blank_index=model.config.mora_inventory.blank_index,
            )
            phone_errors += _edit_distance(phone_prediction, item.phone_targets)
            phone_total += len(item.phone_targets)
            mora_errors += _edit_distance(mora_prediction, item.mora_targets)
            mora_total += len(item.mora_targets)
            phone_positive.append(
                _sequence_score(
                    phone_logits,
                    item.phone_targets,
                    blank_index=model.config.phone_inventory.blank_index,
                )
            )
            phone_hard = _hard_negative(
                item.phone_targets,
                vocabulary_size=len(model.config.phone_inventory.labels),
                blank_index=model.config.phone_inventory.blank_index,
            )
            if phone_hard is not None:
                phone_negative.append(
                    _sequence_score(
                        phone_logits,
                        phone_hard,
                        blank_index=model.config.phone_inventory.blank_index,
                    )
                )
            mora_positive.append(
                _sequence_score(
                    mora_logits,
                    item.mora_targets,
                    blank_index=model.config.mora_inventory.blank_index,
                )
            )
            mora_hard = _hard_negative(
                item.mora_targets,
                vocabulary_size=len(model.config.mora_inventory.labels),
                blank_index=model.config.mora_inventory.blank_index,
            )
            if mora_hard is not None:
                mora_negative.append(
                    _sequence_score(
                        mora_logits,
                        mora_hard,
                        blank_index=model.config.mora_inventory.blank_index,
                    )
                )
    phone_threshold, phone_false_accept = _threshold(
        phone_positive,
        phone_negative,
        true_accept_rate=target_true_accept_rate,
    )
    mora_threshold, mora_false_accept = _threshold(
        mora_positive,
        mora_negative,
        true_accept_rate=target_true_accept_rate,
    )
    calibration = (
        PhoneticSequenceCalibration(
            phone_threshold=phone_threshold,
            mora_threshold=mora_threshold,
            target_true_accept_rate=target_true_accept_rate,
            calibration_manifest_sha256=manifest.manifest_sha256,
            revision=calibration_revision,
            phone_false_accept_rate=phone_false_accept,
            mora_false_accept_rate=mora_false_accept,
        )
        if fit_calibration
        else None
    )
    metrics = PhoneticValidationMetrics(
        phone_error_rate=phone_errors / max(1, phone_total),
        mora_error_rate=mora_errors / max(1, mora_total),
        phone_candidate_auc=_auc(phone_positive, phone_negative),
        mora_candidate_auc=_auc(mora_positive, mora_negative),
        critical_false_accept_rate=max(phone_false_accept, mora_false_accept),
        validation_sample_count=len(manifest.items),
    )
    return PhoneticEvaluationResult(
        metrics=metrics,
        calibration=calibration,
        phone_positive_scores=tuple(phone_positive),
        phone_negative_scores=tuple(phone_negative),
        mora_positive_scores=tuple(mora_positive),
        mora_negative_scores=tuple(mora_negative),
        manifest_sha256=manifest.manifest_sha256,
    )


def save_joint_phonetic_weights(
    model: JointPhoneMoraCTCHead,
    path: str | Path,
    *,
    config_digest: str,
) -> str:
    _require_torch()
    if len(config_digest) != 64:
        raise ValueError("config_digest must be a SHA-256 value")
    output = Path(path)
    if output.suffix != ".safetensors":
        raise ValueError("joint phonetic weights must use the .safetensors suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_safetensors(
        state,
        str(output),
        metadata={
            "configDigest": config_digest,
            "architecture": "shared-layernorm-gelu-bottleneck-dual-ctc-v1",
        },
    )
    return _file_sha256(output)


def build_joint_phonetic_artifact(
    *,
    head_config: JointPhoneticHeadConfig,
    training_manifest: PhoneticTrainingManifest,
    weights_sha256: str,
    test_evaluation: PhoneticEvaluationResult,
    framework_version: str,
    revision: str,
) -> JointPhoneticArtifact:
    return JointPhoneticArtifact(
        config_digest=head_config.digest,
        training_manifest_digest=training_manifest.digest,
        weights_sha256=weights_sha256,
        serialization="safetensors",
        metrics=test_evaluation.metrics,
        framework="torch",
        framework_version=framework_version,
        revision=revision,
    )
