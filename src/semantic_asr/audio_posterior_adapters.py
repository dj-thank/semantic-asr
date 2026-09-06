"""Frozen model adapters for candidate-independent phone and mora posteriorgrams.

The dependency-free core consumes ``PosteriorSequence`` objects. This module defines a strict
backend contract and a generic logit-to-posterior implementation, plus an optional Transformers CTC
adapter. No public model is selected as a default: model identity, revision, label vocabulary,
frame stride, sample rate, and source-audio digest must be supplied or derived exactly.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from .contracts import sha256_json
from .phonetic_evidence import PosteriorFrame, PosteriorKind, PosteriorSequence

RevisionPolicy = Literal["exact-commit", "artifact-digest"]


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


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_commit(value: str) -> bool:
    if len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def canonical_audio_sha256(samples: Sequence[float], sample_rate: int) -> str:
    """Hash canonical little-endian float32 mono samples and their declared sample rate."""

    if isinstance(sample_rate, bool) or sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    digest = hashlib.sha256()
    digest.update(b"semantic-asr-canonical-mono-f32-v1\0")
    digest.update(struct.pack("<I", sample_rate))
    for sample in samples:
        value = _strict_float(sample, name="audio sample")
        digest.update(struct.pack("<f", value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PosteriorResourcePolicy:
    maximum_duration_seconds: float = 30.0
    maximum_frames: int = 6_000
    maximum_vocabulary: int = 2_048
    minimum_frame_duration_ms: float = 1.0
    maximum_frame_duration_ms: float = 100.0

    def __post_init__(self) -> None:
        duration = _strict_float(
            self.maximum_duration_seconds,
            name="maximum_duration_seconds",
        )
        minimum = _strict_float(
            self.minimum_frame_duration_ms,
            name="minimum_frame_duration_ms",
        )
        maximum = _strict_float(
            self.maximum_frame_duration_ms,
            name="maximum_frame_duration_ms",
        )
        if duration <= 0.0 or minimum <= 0.0 or maximum < minimum:
            raise ValueError("posterior resource time limits are invalid")
        if isinstance(self.maximum_frames, bool) or self.maximum_frames < 1:
            raise ValueError("maximum_frames must be positive")
        if isinstance(self.maximum_vocabulary, bool) or self.maximum_vocabulary < 2:
            raise ValueError("maximum_vocabulary must be at least two")
        object.__setattr__(self, "maximum_duration_seconds", duration)
        object.__setattr__(self, "minimum_frame_duration_ms", minimum)
        object.__setattr__(self, "maximum_frame_duration_ms", maximum)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class FrozenPosteriorModelConfig:
    kind: PosteriorKind
    model_id: str
    model_revision: str
    vocabulary: tuple[str, ...]
    blank_symbol: str
    sample_rate: int
    frame_stride_ms: float
    artifact_sha256: str | None = None
    revision_policy: RevisionPolicy = "exact-commit"
    logits_temperature: float = 1.0
    probability_floor: float = 1e-12
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.kind not in {"phone", "mora"}:
            raise ValueError("posterior model kind must be phone or mora")
        if not self.model_id or not self.model_revision:
            raise ValueError("posterior model ID and revision are required")
        vocabulary = tuple(str(value) for value in self.vocabulary)
        if len(vocabulary) != len(set(vocabulary)) or any(not value for value in vocabulary):
            raise ValueError("posterior model vocabulary must be unique and non-empty")
        if self.blank_symbol not in vocabulary:
            raise ValueError("blank_symbol must be present in the vocabulary")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        stride = _strict_float(self.frame_stride_ms, name="frame_stride_ms")
        temperature = _strict_float(
            self.logits_temperature,
            name="logits_temperature",
        )
        floor = _strict_float(self.probability_floor, name="probability_floor")
        if stride <= 0.0 or temperature <= 0.0:
            raise ValueError("frame stride and logits temperature must be positive")
        if not 0.0 < floor < 1.0:
            raise ValueError("probability_floor must be in (0, 1)")
        if self.revision_policy == "exact-commit":
            if not _is_commit(self.model_revision):
                raise ValueError("exact-commit policy requires a 40-character hexadecimal revision")
        elif self.revision_policy == "artifact-digest":
            if not _is_sha256(self.artifact_sha256 or ""):
                raise ValueError("artifact-digest policy requires artifact_sha256")
        else:
            raise ValueError("unknown revision policy")
        if self.artifact_sha256 is not None and not _is_sha256(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a SHA-256 value")
        object.__setattr__(self, "vocabulary", vocabulary)
        object.__setattr__(self, "frame_stride_ms", stride)
        object.__setattr__(self, "logits_temperature", temperature)
        object.__setattr__(self, "probability_floor", floor)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PosteriorLogits:
    """One complete model output bound to canonical input audio and model configuration."""

    values: tuple[tuple[float, ...], ...]
    source_audio_sha256: str
    model_config_digest: str

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("posterior logits require at least one frame")
        width = len(self.values[0])
        if width < 2 or any(len(row) != width for row in self.values):
            raise ValueError("posterior logits must be a rectangular frame-by-label matrix")
        normalized = tuple(
            tuple(_strict_float(value, name="posterior logit") for value in row)
            for row in self.values
        )
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        if not _is_sha256(self.model_config_digest):
            raise ValueError("model_config_digest must be a SHA-256 value")
        object.__setattr__(self, "values", normalized)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "values": self.values,
                "sourceAudioSha256": self.source_audio_sha256,
                "modelConfigDigest": self.model_config_digest,
            }
        )


class AudioPosteriorBackend(Protocol):
    config: FrozenPosteriorModelConfig

    def infer_logits(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str,
    ) -> PosteriorLogits: ...


def _softmax(row: Sequence[float], *, temperature: float, floor: float) -> tuple[float, ...]:
    scaled = [value / temperature for value in row]
    maximum = max(scaled)
    exponentials = [math.exp(max(-80.0, min(80.0, value - maximum))) for value in scaled]
    total = sum(exponentials)
    probabilities = [max(floor, value / total) for value in exponentials]
    renormalizer = sum(probabilities)
    return tuple(value / renormalizer for value in probabilities)


def posterior_sequence_from_logits(
    logits: PosteriorLogits,
    config: FrozenPosteriorModelConfig,
    *,
    resources: PosteriorResourcePolicy | None = None,
) -> PosteriorSequence:
    resources = resources or PosteriorResourcePolicy()
    if logits.model_config_digest != config.digest:
        raise ValueError("posterior logits are bound to a different model configuration")
    if len(config.vocabulary) > resources.maximum_vocabulary:
        raise ValueError("posterior vocabulary exceeds the configured resource limit")
    if len(logits.values) > resources.maximum_frames:
        raise ValueError("posterior frame count exceeds the configured resource limit")
    if any(len(row) != len(config.vocabulary) for row in logits.values):
        raise ValueError("posterior logit width does not match the frozen vocabulary")
    stride = config.frame_stride_ms
    if not resources.minimum_frame_duration_ms <= stride <= resources.maximum_frame_duration_ms:
        raise ValueError("posterior frame stride violates the resource policy")
    frames = []
    for index, row in enumerate(logits.values):
        probabilities = _softmax(
            row,
            temperature=config.logits_temperature,
            floor=config.probability_floor,
        )
        start_ms = round(index * stride)
        end_ms = max(start_ms + 1, round((index + 1) * stride))
        frames.append(
            PosteriorFrame(
                start_ms=start_ms,
                end_ms=end_ms,
                probabilities=tuple(zip(config.vocabulary, probabilities, strict=True)),
            )
        )
    return PosteriorSequence(
        kind=config.kind,
        blank_symbol=config.blank_symbol,
        vocabulary=config.vocabulary,
        frames=tuple(frames),
        encoder=config.model_id,
        encoder_revision=config.model_revision,
        label_set_revision=config.digest,
        source_audio_sha256=logits.source_audio_sha256,
    )


class FrozenAudioPosteriorExtractor:
    """Validate canonical audio, execute one frozen backend, and emit a typed posteriorgram."""

    def __init__(
        self,
        backend: AudioPosteriorBackend,
        *,
        resources: PosteriorResourcePolicy | None = None,
    ) -> None:
        self.backend = backend
        self.config = backend.config
        self.resources = resources or PosteriorResourcePolicy()

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "modelConfigDigest": self.config.digest,
                "resourcePolicyDigest": self.resources.digest,
                "implementation": "frozen-audio-posterior-extractor-v1",
            }
        )

    def extract(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str | None = None,
    ) -> PosteriorSequence:
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                "audio sample rate does not match the frozen model; resampling must be explicit"
            )
        duration = len(samples) / sample_rate
        if duration <= 0.0:
            raise ValueError("audio samples must not be empty")
        if duration > self.resources.maximum_duration_seconds:
            raise ValueError("audio duration exceeds the configured posterior resource limit")
        canonical_digest = canonical_audio_sha256(samples, sample_rate)
        if source_audio_sha256 is None:
            source_audio_sha256 = canonical_digest
        elif not _is_sha256(source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        logits = self.backend.infer_logits(
            samples,
            sample_rate=sample_rate,
            source_audio_sha256=source_audio_sha256,
        )
        if logits.source_audio_sha256 != source_audio_sha256:
            raise ValueError("posterior backend returned logits for different source audio")
        if logits.model_config_digest != self.config.digest:
            raise ValueError("posterior backend returned logits for different model configuration")
        return posterior_sequence_from_logits(logits, self.config, resources=self.resources)


@dataclass(frozen=True, slots=True)
class PosteriorBundle:
    source_audio_sha256: str
    phone: PosteriorSequence | None = None
    mora: PosteriorSequence | None = None

    def __post_init__(self) -> None:
        if self.phone is None and self.mora is None:
            raise ValueError("posterior bundle requires phone or mora evidence")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("posterior bundle source digest must be a SHA-256 value")
        if self.phone is not None:
            if self.phone.kind != "phone" or self.phone.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("phone posterior is bound to different evidence")
        if self.mora is not None:
            if self.mora.kind != "mora" or self.mora.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("mora posterior is bound to different evidence")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "sourceAudioSha256": self.source_audio_sha256,
                "phoneDigest": None if self.phone is None else self.phone.digest,
                "moraDigest": None if self.mora is None else self.mora.digest,
            }
        )


class DualPosteriorExtractor:
    def __init__(
        self,
        *,
        phone: FrozenAudioPosteriorExtractor | None = None,
        mora: FrozenAudioPosteriorExtractor | None = None,
    ) -> None:
        if phone is None and mora is None:
            raise ValueError("dual posterior extractor requires phone or mora extractor")
        if phone is not None and phone.config.kind != "phone":
            raise ValueError("phone extractor has the wrong posterior kind")
        if mora is not None and mora.config.kind != "mora":
            raise ValueError("mora extractor has the wrong posterior kind")
        if phone is not None and mora is not None and phone.config.sample_rate != mora.config.sample_rate:
            raise ValueError("phone and mora extractors must use the same sample rate")
        self.phone = phone
        self.mora = mora

    def extract(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str | None = None,
    ) -> PosteriorBundle:
        source = source_audio_sha256 or canonical_audio_sha256(samples, sample_rate)
        phone = (
            None
            if self.phone is None
            else self.phone.extract(
                samples,
                sample_rate=sample_rate,
                source_audio_sha256=source,
            )
        )
        mora = (
            None
            if self.mora is None
            else self.mora.extract(
                samples,
                sample_rate=sample_rate,
                source_audio_sha256=source,
            )
        )
        return PosteriorBundle(source_audio_sha256=source, phone=phone, mora=mora)


class TransformersCTCBackend:
    """Optional Hugging Face CTC backend with immutable revision and no remote code execution."""

    def __init__(
        self,
        config: FrozenPosteriorModelConfig,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        try:
            import torch
            from transformers import AutoModelForCTC, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TransformersCTCBackend requires torch and transformers"
            ) from exc
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCTC.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        id2label = getattr(self.model.config, "id2label", None)
        if not isinstance(id2label, dict):
            raise ValueError("CTC model config must expose id2label")
        labels = tuple(str(id2label.get(index, id2label.get(str(index), ""))) for index in range(len(id2label)))
        if labels != config.vocabulary:
            raise ValueError("CTC model label order does not match the frozen vocabulary")

    def infer_logits(
        self,
        samples: Sequence[float],
        *,
        sample_rate: int,
        source_audio_sha256: str,
    ) -> PosteriorLogits:
        if sample_rate != self.config.sample_rate:
            raise ValueError("Transformers CTC input sample rate mismatch")
        inputs = self.processor(
            list(samples),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        tensors = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        with self._torch.inference_mode():
            logits = self.model(**tensors).logits[0].detach().to("cpu").float().tolist()
        return PosteriorLogits(
            values=tuple(tuple(float(value) for value in row) for row in logits),
            source_audio_sha256=source_audio_sha256,
            model_config_digest=self.config.digest,
        )


def read_mono_wav(path: str | Path) -> tuple[tuple[float, ...], int, str]:
    """Read a WAV explicitly; no implicit resampling, clipping, or channel guessing."""

    try:
        import soundfile
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("read_mono_wav requires soundfile") from exc
    audio_path = Path(path)
    data, sample_rate = soundfile.read(audio_path, dtype="float32", always_2d=True)
    if data.shape[1] != 1:
        raise ValueError("phone/mora posterior extraction requires explicit mono audio")
    samples = tuple(float(value) for value in data[:, 0])
    source_audio_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    return samples, int(sample_rate), source_audio_sha256
