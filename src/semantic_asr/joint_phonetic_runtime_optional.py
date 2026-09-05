"""Optional raw-audio runtime for a frozen encoder plus the trained joint phone/mora head."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .audio_posterior_adapters import (
    PosteriorBundle,
    PosteriorLogits,
    PosteriorResourcePolicy,
    posterior_sequence_from_logits,
)
from .contracts import sha256_json
from .phonetic_heads_optional import JointPhoneMoraCTCHead
from .phonetic_training import (
    JointPhoneticArtifact,
    JointPhoneticHeadConfig,
    posterior_configs_from_artifact,
)

try:  # pragma: no cover - optional runtime dependency
    import torch
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover
    torch = None
    load_safetensors = None


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenAudioFeatureConfig:
    model_id: str
    model_revision: str
    layer_index: int
    sample_rate: int
    feature_dimension: int
    frame_stride_ms: float
    model_artifact_sha256: str | None = None
    revision_policy: str = "exact-commit"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_revision:
            raise ValueError("feature encoder model identity and revision are required")
        if self.revision_policy not in {"exact-commit", "artifact-digest"}:
            raise ValueError("unknown feature encoder revision policy")
        if self.revision_policy == "exact-commit":
            if len(self.model_revision) != 40:
                raise ValueError("feature encoder revision must be an exact 40-character commit")
            try:
                int(self.model_revision, 16)
            except ValueError as exc:
                raise ValueError("feature encoder revision must be hexadecimal") from exc
        if self.revision_policy == "artifact-digest" and not _is_sha256(
            self.model_artifact_sha256 or ""
        ):
            raise ValueError("artifact-digest feature encoders require model_artifact_sha256")
        if self.model_artifact_sha256 is not None and not _is_sha256(
            self.model_artifact_sha256
        ):
            raise ValueError("model_artifact_sha256 must be a SHA-256 value")
        if isinstance(self.layer_index, bool) or self.layer_index < 0:
            raise ValueError("layer_index must be a non-negative integer")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if isinstance(self.feature_dimension, bool) or self.feature_dimension < 1:
            raise ValueError("feature_dimension must be positive")
        stride = float(self.frame_stride_ms)
        if not math.isfinite(stride) or stride <= 0.0:
            raise ValueError("frame_stride_ms must be finite and positive")
        object.__setattr__(self, "frame_stride_ms", stride)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class FrozenFeatureMatrix:
    values: tuple[tuple[float, ...], ...]
    source_audio_sha256: str
    feature_config_digest: str

    def __post_init__(self) -> None:
        if not self.values or not self.values[0]:
            raise ValueError("feature matrix must be non-empty")
        width = len(self.values[0])
        for row in self.values:
            if len(row) != width:
                raise ValueError("feature matrix rows have inconsistent widths")
            if any(not math.isfinite(float(value)) for value in row):
                raise ValueError("feature matrix values must be finite")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        if not _is_sha256(self.feature_config_digest):
            raise ValueError("feature_config_digest must be a SHA-256 value")

    @property
    def frame_count(self) -> int:
        return len(self.values)

    @property
    def feature_dimension(self) -> int:
        return len(self.values[0])

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "sourceAudioSha256": self.source_audio_sha256,
                "featureConfigDigest": self.feature_config_digest,
                "frameCount": self.frame_count,
                "featureDimension": self.feature_dimension,
                "values": self.values,
            }
        )


class FrozenAudioFeatureBackend(Protocol):
    config: FrozenAudioFeatureConfig

    def extract_features(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str,
    ) -> FrozenFeatureMatrix: ...


class TransformersAudioFeatureBackend:
    """Pinned Hugging Face encoder hidden-state adapter with no remote code execution."""

    def __init__(
        self,
        config: FrozenAudioFeatureConfig,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("TransformersAudioFeatureBackend requires PyTorch")
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("TransformersAudioFeatureBackend requires transformers") from exc
        self.config = config
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        declared = getattr(self.feature_extractor, "sampling_rate", None)
        if declared is not None and int(declared) != config.sample_rate:
            raise ValueError("feature extractor sample rate differs from frozen config")

    def extract_features(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str,
    ) -> FrozenFeatureMatrix:
        if sample_rate != self.config.sample_rate:
            raise ValueError("audio sample rate differs from frozen feature config")
        if not _is_sha256(source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        values = tuple(float(value) for value in samples)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("audio samples must be finite and non-empty")
        inputs = self.feature_extractor(
            values,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None or self.config.layer_index >= len(hidden_states):
            raise ValueError("frozen feature layer is absent from encoder output")
        matrix = hidden_states[self.config.layer_index][0].detach().cpu().float()
        if matrix.ndim != 2 or matrix.shape[1] != self.config.feature_dimension:
            raise ValueError("encoder feature dimension differs from frozen config")
        return FrozenFeatureMatrix(
            values=tuple(tuple(float(value) for value in row) for row in matrix.tolist()),
            source_audio_sha256=source_audio_sha256,
            feature_config_digest=self.config.digest,
        )


class JointPhoneticPosteriorExtractor:
    """Run one frozen encoder pass and one shared head pass for phone and mora posteriors."""

    def __init__(
        self,
        *,
        feature_backend: FrozenAudioFeatureBackend,
        head_config: JointPhoneticHeadConfig,
        artifact: JointPhoneticArtifact,
        weights_path: str | Path,
        device: str = "cpu",
        resources: PosteriorResourcePolicy | None = None,
    ) -> None:
        if torch is None or load_safetensors is None:
            raise RuntimeError("joint phonetic runtime requires PyTorch and safetensors")
        if artifact.config_digest != head_config.digest:
            raise ValueError("joint phonetic artifact is bound to a different head config")
        if feature_backend.config.feature_dimension != head_config.input_dimension:
            raise ValueError("feature backend dimension differs from joint head config")
        if feature_backend.config.model_id != head_config.encoder_id:
            raise ValueError("feature backend model differs from joint head encoder")
        if feature_backend.config.model_revision != head_config.encoder_revision:
            raise ValueError("feature backend revision differs from joint head encoder")
        if head_config.encoder_artifact_sha256 is not None and (
            feature_backend.config.model_artifact_sha256
            != head_config.encoder_artifact_sha256
        ):
            raise ValueError("feature backend artifact differs from joint head encoder")
        self.feature_backend = feature_backend
        self.head_config = head_config
        self.artifact = artifact
        self.weights_path = Path(weights_path)
        if _file_sha256(self.weights_path) != artifact.weights_sha256:
            raise ValueError("joint phonetic weights SHA-256 mismatch")
        self.device = torch.device(device)
        self.resources = resources or PosteriorResourcePolicy()
        self.model = JointPhoneMoraCTCHead(head_config).to(self.device)
        state = load_safetensors(str(self.weights_path), device=str(self.device))
        incompatible = self.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("joint phonetic weights do not match the head architecture")
        self.model.eval()
        self.phone_config, self.mora_config = posterior_configs_from_artifact(
            head_config,
            artifact,
            sample_rate=feature_backend.config.sample_rate,
            frame_stride_ms=feature_backend.config.frame_stride_ms,
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "featureConfigDigest": self.feature_backend.config.digest,
                "headConfigDigest": self.head_config.digest,
                "artifactDigest": self.artifact.digest,
                "weightsSha256": self.artifact.weights_sha256,
                "phoneConfigDigest": self.phone_config.digest,
                "moraConfigDigest": self.mora_config.digest,
                "device": str(self.device),
                "resourcePolicyDigest": self.resources.digest,
            }
        )

    def extract(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str,
    ) -> PosteriorBundle:
        feature = self.feature_backend.extract_features(
            samples,
            sample_rate=sample_rate,
            source_audio_sha256=source_audio_sha256,
        )
        if feature.feature_config_digest != self.feature_backend.config.digest:
            raise ValueError("feature matrix is bound to a different backend config")
        if feature.source_audio_sha256 != source_audio_sha256:
            raise ValueError("feature matrix is bound to different source audio")
        if feature.feature_dimension != self.head_config.input_dimension:
            raise ValueError("feature matrix dimension differs from joint head config")
        if feature.frame_count > self.resources.maximum_frames:
            raise ValueError("joint phonetic feature output exceeds maximum_frames")
        tensor = torch.tensor(
            feature.values,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(tensor)
        phone_values = tuple(
            tuple(float(value) for value in row)
            for row in output.phone_logits[0].detach().cpu().tolist()
        )
        mora_values = tuple(
            tuple(float(value) for value in row)
            for row in output.mora_logits[0].detach().cpu().tolist()
        )
        phone_logits = PosteriorLogits(
            values=phone_values,
            source_audio_sha256=source_audio_sha256,
            model_config_digest=self.phone_config.digest,
        )
        mora_logits = PosteriorLogits(
            values=mora_values,
            source_audio_sha256=source_audio_sha256,
            model_config_digest=self.mora_config.digest,
        )
        return PosteriorBundle(
            source_audio_sha256=source_audio_sha256,
            phone=posterior_sequence_from_logits(
                phone_logits,
                self.phone_config,
                resources=self.resources,
            ),
            mora=posterior_sequence_from_logits(
                mora_logits,
                self.mora_config,
                resources=self.resources,
            ),
            extractor_digests=(self.digest,),
        )
