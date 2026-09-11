"""Reproducible training contracts for joint Japanese phone and mora CTC heads.

The optional neural implementation consumes frozen or explicitly trainable acoustic frame features
and predicts phone and mora posteriors from one shared bottleneck. This module contains no model
weights and makes no quality claim; it defines immutable label, split, artifact, and promotion
contracts used by the optional PyTorch implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .audio_posterior_adapters import FrozenPosteriorModelConfig
from .contracts import sha256_json


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


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


@dataclass(frozen=True, slots=True)
class PhoneticLabelInventory:
    kind: str
    labels: tuple[str, ...]
    blank_symbol: str
    revision: str
    source_manifest_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.kind not in {"phone", "mora"}:
            raise ValueError("phonetic label inventory kind must be phone or mora")
        labels = tuple(str(value) for value in self.labels)
        if len(labels) < 2 or len(labels) != len(set(labels)) or any(not value for value in labels):
            raise ValueError("phonetic labels must be unique, non-empty, and include two symbols")
        if self.blank_symbol not in labels:
            raise ValueError("blank_symbol must be present in labels")
        if not self.revision:
            raise ValueError("label inventory revision is required")
        if not _is_sha256(self.source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be a SHA-256 value")
        object.__setattr__(self, "labels", labels)

    @property
    def blank_index(self) -> int:
        return self.labels.index(self.blank_symbol)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class JointPhoneticHeadConfig:
    input_dimension: int
    hidden_dimension: int
    phone_inventory: PhoneticLabelInventory
    mora_inventory: PhoneticLabelInventory
    encoder_id: str
    encoder_revision: str
    encoder_artifact_sha256: str | None
    dropout: float = 0.10
    phone_loss_weight: float = 1.0
    mora_loss_weight: float = 1.0
    blank_regularization_weight: float = 0.0
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("input_dimension", "hidden_dimension"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.encoder_id or not self.encoder_revision:
            raise ValueError("encoder identity and revision are required")
        if self.encoder_artifact_sha256 is not None and not _is_sha256(
            self.encoder_artifact_sha256
        ):
            raise ValueError("encoder_artifact_sha256 must be a SHA-256 value")
        for name in (
            "dropout",
            "phone_loss_weight",
            "mora_loss_weight",
            "blank_regularization_weight",
        ):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.phone_loss_weight <= 0.0 or self.mora_loss_weight <= 0.0:
            raise ValueError("phone and mora loss weights must be positive")
        if self.blank_regularization_weight < 0.0:
            raise ValueError("blank_regularization_weight must be non-negative")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "phoneInventoryDigest": self.phone_inventory.digest,
                "moraInventoryDigest": self.mora_inventory.digest,
                "architecture": "shared-layernorm-gelu-bottleneck-dual-ctc-v1",
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticTrainingManifest:
    training_manifest_sha256: str
    calibration_manifest_sha256: str
    test_manifest_sha256: str
    speaker_disjoint: bool
    source_disjoint: bool
    rights_registry_sha256: str
    feature_revision: str
    random_seed: int
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for digest in (
            self.training_manifest_sha256,
            self.calibration_manifest_sha256,
            self.test_manifest_sha256,
            self.rights_registry_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("training manifest digests must be SHA-256 values")
        if len(
            {
                self.training_manifest_sha256,
                self.calibration_manifest_sha256,
                self.test_manifest_sha256,
            }
        ) != 3:
            raise ValueError("training, calibration, and test manifests must differ")
        if not self.speaker_disjoint or not self.source_disjoint:
            raise ValueError("phonetic model splits must be speaker and source disjoint")
        if not self.feature_revision:
            raise ValueError("feature_revision is required")
        if isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticValidationMetrics:
    phone_error_rate: float
    mora_error_rate: float
    phone_candidate_auc: float
    mora_candidate_auc: float
    critical_false_accept_rate: float
    validation_sample_count: int

    def __post_init__(self) -> None:
        for name in (
            "phone_error_rate",
            "mora_error_rate",
            "phone_candidate_auc",
            "mora_candidate_auc",
            "critical_false_accept_rate",
        ):
            value = _strict_float(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        if isinstance(self.validation_sample_count, bool) or self.validation_sample_count < 1:
            raise ValueError("validation_sample_count must be positive")


@dataclass(frozen=True, slots=True)
class JointPhoneticArtifact:
    config_digest: str
    training_manifest_digest: str
    weights_sha256: str
    serialization: str
    metrics: PhoneticValidationMetrics
    framework: str
    framework_version: str
    revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for digest in (
            self.config_digest,
            self.training_manifest_digest,
            self.weights_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("phonetic artifact digests must be SHA-256 values")
        if self.serialization not in {"safetensors"}:
            raise ValueError("phonetic weights must use an explicitly checked safe serialization")
        if not self.framework or not self.framework_version or not self.revision:
            raise ValueError("phonetic artifact framework and revision are required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def posterior_configs_from_artifact(
    config: JointPhoneticHeadConfig,
    artifact: JointPhoneticArtifact,
    *,
    sample_rate: int,
    frame_stride_ms: float,
) -> tuple[FrozenPosteriorModelConfig, FrozenPosteriorModelConfig]:
    if artifact.config_digest != config.digest:
        raise ValueError("phonetic artifact is bound to a different head configuration")
    model_id = f"joint-phonetic-head:{artifact.revision}"
    phone = FrozenPosteriorModelConfig(
        kind="phone",
        model_id=model_id,
        model_revision=artifact.revision,
        vocabulary=config.phone_inventory.labels,
        blank_symbol=config.phone_inventory.blank_symbol,
        sample_rate=sample_rate,
        frame_stride_ms=frame_stride_ms,
        artifact_sha256=artifact.weights_sha256,
        revision_policy="artifact-digest",
    )
    mora = FrozenPosteriorModelConfig(
        kind="mora",
        model_id=model_id,
        model_revision=artifact.revision,
        vocabulary=config.mora_inventory.labels,
        blank_symbol=config.mora_inventory.blank_symbol,
        sample_rate=sample_rate,
        frame_stride_ms=frame_stride_ms,
        artifact_sha256=artifact.weights_sha256,
        revision_policy="artifact-digest",
    )
    return phone, mora
