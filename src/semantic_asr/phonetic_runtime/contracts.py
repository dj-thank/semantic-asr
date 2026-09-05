"""Frozen contracts for a source-audio-only dual phone/mora CTC runtime."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256

PhoneticKind = Literal["phone", "mora"]


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _strict_float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return numeric


@dataclass(frozen=True, slots=True)
class PhoneticInventory:
    """Immutable CTC label space. Index zero is always the blank label."""

    kind: PhoneticKind
    symbols: tuple[str, ...]
    blank_symbol: str = "<blk>"
    unknown_symbol: str | None = None
    language: str = "ja"
    revision: str = ""
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.kind not in {"phone", "mora"}:
            raise ValueError("inventory kind must be phone or mora")
        if not self.language or not self.revision:
            raise ValueError("inventory language and revision are required")
        if not self.blank_symbol:
            raise ValueError("blank_symbol is required")
        if not self.symbols:
            raise ValueError("inventory requires symbols")
        if any(not isinstance(symbol, str) or not symbol for symbol in self.symbols):
            raise TypeError("inventory symbols must be non-empty strings")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError("inventory symbols must be unique")
        if self.symbols[0] != self.blank_symbol:
            raise ValueError("CTC blank_symbol must occupy index zero")
        if self.unknown_symbol is not None and self.unknown_symbol not in self.symbols:
            raise ValueError("unknown_symbol must be present in symbols")

    @property
    def size(self) -> int:
        return len(self.symbols)

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def symbol_to_id(self) -> dict[str, int]:
        return {symbol: index for index, symbol in enumerate(self.symbols)}

    def encode(self, values: tuple[str, ...]) -> tuple[int, ...]:
        mapping = self.symbol_to_id
        output: list[int] = []
        for value in values:
            if value == self.blank_symbol:
                raise ValueError("target labels must not contain the CTC blank")
            try:
                output.append(mapping[value])
            except KeyError:
                if self.unknown_symbol is None:
                    raise ValueError(f"unknown {self.kind} symbol: {value!r}") from None
                output.append(mapping[self.unknown_symbol])
        if not output:
            raise ValueError("target label sequence must not be empty")
        return tuple(output)

    def decode(self, values: tuple[int, ...]) -> tuple[str, ...]:
        output: list[str] = []
        for value in values:
            _strict_int(value, name="label ID", minimum=0)
            if value >= len(self.symbols):
                raise ValueError("label ID exceeds the frozen inventory")
            output.append(self.symbols[value])
        return tuple(output)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class LogMelFrontendConfig:
    sample_rate: int = 16_000
    n_fft: int = 400
    window_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    frequency_min: float = 20.0
    frequency_max: float = 7_600.0
    log_floor: float = 1e-10
    normalize_per_utterance: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("sample_rate", "n_fft", "window_length", "hop_length", "n_mels"):
            _strict_int(getattr(self, name), name=name, minimum=1)
        if self.window_length > self.n_fft:
            raise ValueError("window_length cannot exceed n_fft")
        if self.frequency_max > self.sample_rate / 2:
            raise ValueError("frequency_max cannot exceed the Nyquist frequency")
        minimum = _strict_float(self.frequency_min, name="frequency_min", minimum=0.0)
        maximum = _strict_float(self.frequency_max, name="frequency_max", minimum=0.0)
        floor = _strict_float(self.log_floor, name="log_floor", minimum=0.0)
        if minimum >= maximum:
            raise ValueError("frequency_min must be smaller than frequency_max")
        if floor <= 0.0:
            raise ValueError("log_floor must be positive")
        if not isinstance(self.normalize_per_utterance, bool):
            raise TypeError("normalize_per_utterance must be a boolean")
        object.__setattr__(self, "frequency_min", minimum)
        object.__setattr__(self, "frequency_max", maximum)
        object.__setattr__(self, "log_floor", floor)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DualCTCModelConfig:
    frontend: LogMelFrontendConfig = LogMelFrontendConfig()
    hidden_dimension: int = 256
    encoder_layers: int = 6
    attention_heads: int = 4
    feedforward_dimension: int = 1_024
    convolution_kernel: int = 5
    subsampling_layers: int = 2
    dropout: float = 0.1
    maximum_frames: int = 12_000
    architecture_revision: str = "dual-transformer-ctc-v1"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "hidden_dimension",
            "encoder_layers",
            "attention_heads",
            "feedforward_dimension",
            "convolution_kernel",
            "subsampling_layers",
            "maximum_frames",
        ):
            _strict_int(getattr(self, name), name=name, minimum=1)
        if self.hidden_dimension % self.attention_heads:
            raise ValueError("hidden_dimension must be divisible by attention_heads")
        if self.convolution_kernel % 2 == 0:
            raise ValueError("convolution_kernel must be odd")
        dropout = _strict_float(self.dropout, name="dropout", minimum=0.0, maximum=1.0)
        if dropout >= 1.0:
            raise ValueError("dropout must be smaller than one")
        if not self.architecture_revision:
            raise ValueError("architecture_revision is required")
        object.__setattr__(self, "dropout", dropout)

    @property
    def subsampling_factor(self) -> int:
        return 2**self.subsampling_layers

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "frontendDigest": self.frontend.digest,
                "subsamplingFactor": self.subsampling_factor,
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticRuntimeLimits:
    maximum_audio_seconds: float = 35.0
    maximum_file_bytes: int = 256 * 1024 * 1024
    allowed_channels: tuple[int, ...] = (1, 2)
    require_pcm16: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        seconds = _strict_float(
            self.maximum_audio_seconds,
            name="maximum_audio_seconds",
            minimum=0.001,
        )
        _strict_int(self.maximum_file_bytes, name="maximum_file_bytes", minimum=1)
        if not self.allowed_channels or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.allowed_channels
        ):
            raise ValueError("allowed_channels must contain positive integers")
        if len(self.allowed_channels) != len(set(self.allowed_channels)):
            raise ValueError("allowed_channels must be unique")
        if not isinstance(self.require_pcm16, bool):
            raise TypeError("require_pcm16 must be a boolean")
        object.__setattr__(self, "maximum_audio_seconds", seconds)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class TensorSpecification:
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.dtype:
            raise ValueError("tensor specification requires name and dtype")
        if not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.shape
        ):
            raise ValueError("tensor shape must contain non-negative integers")
        if not _is_sha256(self.sha256):
            raise ValueError("tensor sha256 must be a SHA-256 value")


@dataclass(frozen=True, slots=True)
class DualCTCArtifactMetadata:
    name: str
    revision: str
    model_config: DualCTCModelConfig
    phone_inventory: PhoneticInventory
    mora_inventory: PhoneticInventory
    training_manifest_sha256: str
    runtime_revision: str
    weights_filename: str
    weights_sha256: str
    tensors: tuple[TensorSpecification, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.runtime_revision:
            raise ValueError("artifact name, revision, and runtime_revision are required")
        if self.phone_inventory.kind != "phone" or self.mora_inventory.kind != "mora":
            raise ValueError("artifact inventories are assigned to the wrong heads")
        if not _is_sha256(self.training_manifest_sha256):
            raise ValueError("training_manifest_sha256 must be a SHA-256 value")
        if not self.weights_filename or PathLikeSeparator.present(self.weights_filename):
            raise ValueError("weights_filename must be a safe basename")
        if not _is_sha256(self.weights_sha256):
            raise ValueError("weights_sha256 must be a SHA-256 value")
        if not self.tensors or len({row.name for row in self.tensors}) != len(self.tensors):
            raise ValueError("tensor specifications must be non-empty and unique")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "modelConfig": asdict(self.model_config),
                "modelConfigDigest": self.model_config.digest,
                "phoneInventory": asdict(self.phone_inventory),
                "phoneInventoryDigest": self.phone_inventory.digest,
                "moraInventory": asdict(self.mora_inventory),
                "moraInventoryDigest": self.mora_inventory.digest,
                "trainingManifestSha256": self.training_manifest_sha256,
                "runtimeRevision": self.runtime_revision,
                "weightsFilename": self.weights_filename,
                "weightsSha256": self.weights_sha256,
                "tensors": [asdict(row) for row in self.tensors],
            }
        )


class PathLikeSeparator:
    @staticmethod
    def present(value: str) -> bool:
        return "/" in value or "\\" in value or value in {".", ".."}
