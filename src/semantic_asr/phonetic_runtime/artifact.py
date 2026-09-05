"""Tamper-checked, pickle-free artifact persistence for the dual CTC runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .contracts import (
    DualCTCArtifactMetadata,
    DualCTCModelConfig,
    LogMelFrontendConfig,
    PhoneticInventory,
    TensorSpecification,
)

if TYPE_CHECKING:
    from torch import nn

_METADATA_FILENAME = "metadata.json"
_WEIGHTS_FILENAME = "weights.npz"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: dict[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _strict_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _frontend_from_dict(value: dict[str, Any]) -> LogMelFrontendConfig:
    _exact_keys(
        value,
        {
            "sample_rate",
            "n_fft",
            "window_length",
            "hop_length",
            "n_mels",
            "frequency_min",
            "frequency_max",
            "log_floor",
            "normalize_per_utterance",
            "schema_version",
        },
        name="log-Mel frontend config",
    )
    return LogMelFrontendConfig(**value)


def _model_config_from_dict(value: dict[str, Any]) -> DualCTCModelConfig:
    _exact_keys(
        value,
        {
            "frontend",
            "hidden_dimension",
            "encoder_layers",
            "attention_heads",
            "feedforward_dimension",
            "convolution_kernel",
            "subsampling_layers",
            "dropout",
            "maximum_frames",
            "architecture_revision",
            "schema_version",
        },
        name="dual CTC model config",
    )
    payload = dict(value)
    frontend = payload.pop("frontend")
    if not isinstance(frontend, dict):
        raise TypeError("model frontend config must be an object")
    return DualCTCModelConfig(frontend=_frontend_from_dict(frontend), **payload)


def _inventory_from_dict(value: dict[str, Any]) -> PhoneticInventory:
    _exact_keys(
        value,
        {
            "kind",
            "symbols",
            "blank_symbol",
            "unknown_symbol",
            "language",
            "revision",
            "schema_version",
        },
        name="phonetic inventory",
    )
    payload = dict(value)
    symbols = payload.pop("symbols")
    if not isinstance(symbols, list):
        raise TypeError("inventory symbols must be a JSON array")
    return PhoneticInventory(symbols=tuple(symbols), **payload)


def _tensor_spec_from_dict(value: dict[str, Any]) -> TensorSpecification:
    _exact_keys(value, {"name", "shape", "dtype", "sha256"}, name="tensor specification")
    shape = value["shape"]
    if not isinstance(shape, list):
        raise TypeError("tensor shape must be a JSON array")
    return TensorSpecification(
        name=_strict_string(value["name"], name="tensor name"),
        shape=tuple(_strict_int(item, name="tensor dimension") for item in shape),
        dtype=_strict_string(value["dtype"], name="tensor dtype"),
        sha256=_strict_string(value["sha256"], name="tensor SHA-256"),
    )


def _metadata_payload(metadata: DualCTCArtifactMetadata) -> dict[str, object]:
    return {
        "schemaVersion": metadata.schema_version,
        "name": metadata.name,
        "revision": metadata.revision,
        "modelConfig": asdict(metadata.model_config),
        "modelConfigDigest": metadata.model_config.digest,
        "phoneInventory": asdict(metadata.phone_inventory),
        "phoneInventoryDigest": metadata.phone_inventory.digest,
        "moraInventory": asdict(metadata.mora_inventory),
        "moraInventoryDigest": metadata.mora_inventory.digest,
        "trainingManifestSha256": metadata.training_manifest_sha256,
        "runtimeRevision": metadata.runtime_revision,
        "weightsFilename": metadata.weights_filename,
        "weightsSha256": metadata.weights_sha256,
        "tensors": [asdict(row) for row in metadata.tensors],
    }


def _metadata_from_dict(value: dict[str, Any]) -> DualCTCArtifactMetadata:
    _exact_keys(
        value,
        {
            "schemaVersion",
            "name",
            "revision",
            "modelConfig",
            "modelConfigDigest",
            "phoneInventory",
            "phoneInventoryDigest",
            "moraInventory",
            "moraInventoryDigest",
            "trainingManifestSha256",
            "runtimeRevision",
            "weightsFilename",
            "weightsSha256",
            "tensors",
            "artifactDigest",
        },
        name="dual CTC artifact metadata",
    )
    for name in ("modelConfig", "phoneInventory", "moraInventory"):
        if not isinstance(value[name], dict):
            raise TypeError(f"{name} must be an object")
    if not isinstance(value["tensors"], list):
        raise TypeError("tensors must be an array")
    model_config = _model_config_from_dict(dict(value["modelConfig"]))
    phone_inventory = _inventory_from_dict(dict(value["phoneInventory"]))
    mora_inventory = _inventory_from_dict(dict(value["moraInventory"]))
    if model_config.digest != value["modelConfigDigest"]:
        raise ValueError("model config digest mismatch")
    if phone_inventory.digest != value["phoneInventoryDigest"]:
        raise ValueError("phone inventory digest mismatch")
    if mora_inventory.digest != value["moraInventoryDigest"]:
        raise ValueError("mora inventory digest mismatch")
    metadata = DualCTCArtifactMetadata(
        name=_strict_string(value["name"], name="artifact name"),
        revision=_strict_string(value["revision"], name="artifact revision"),
        model_config=model_config,
        phone_inventory=phone_inventory,
        mora_inventory=mora_inventory,
        training_manifest_sha256=_strict_string(
            value["trainingManifestSha256"],
            name="training manifest SHA-256",
        ),
        runtime_revision=_strict_string(value["runtimeRevision"], name="runtime revision"),
        weights_filename=_strict_string(value["weightsFilename"], name="weights filename"),
        weights_sha256=_strict_string(value["weightsSha256"], name="weights SHA-256"),
        tensors=tuple(_tensor_spec_from_dict(dict(row)) for row in value["tensors"]),
        schema_version=_strict_string(value["schemaVersion"], name="schema version"),
    )
    artifact_digest = _strict_string(value["artifactDigest"], name="artifact digest")
    if not _is_sha256(artifact_digest) or metadata.digest != artifact_digest:
        raise ValueError("dual CTC artifact metadata digest mismatch")
    return metadata


@dataclass(slots=True)
class LoadedDualCTCArtifact:
    model: nn.Module
    metadata: DualCTCArtifactMetadata
    directory: Path

    @property
    def digest(self) -> str:
        return metadata_runtime_digest(self.metadata)


def metadata_runtime_digest(metadata: DualCTCArtifactMetadata) -> str:
    return sha256_json(
        {
            "artifactDigest": metadata.digest,
            "modelConfigDigest": metadata.model_config.digest,
            "phoneInventoryDigest": metadata.phone_inventory.digest,
            "moraInventoryDigest": metadata.mora_inventory.digest,
            "weightsSha256": metadata.weights_sha256,
            "runtimeRevision": metadata.runtime_revision,
        }
    )


def save_dual_ctc_artifact(
    destination: str | Path,
    model: nn.Module,
    *,
    name: str,
    revision: str,
    model_config: DualCTCModelConfig,
    phone_inventory: PhoneticInventory,
    mora_inventory: PhoneticInventory,
    training_manifest_sha256: str,
    runtime_revision: str,
) -> DualCTCArtifactMetadata:
    """Atomically create a new artifact directory. Existing destinations are never replaced."""

    import numpy as np
    import torch

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"artifact destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))
    try:
        arrays: dict[str, np.ndarray] = {}
        specifications: list[TensorSpecification] = []
        for tensor_name, tensor in sorted(model.state_dict().items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("model state_dict contains a non-tensor value")
            array_value = tensor.detach().cpu().contiguous().numpy()
            if array_value.dtype.hasobject:
                raise TypeError("object tensors are not allowed in model artifacts")
            arrays[tensor_name] = array_value
            specifications.append(
                TensorSpecification(
                    name=tensor_name,
                    shape=tuple(int(value) for value in array_value.shape),
                    dtype=str(array_value.dtype),
                    sha256=_sha256_bytes(array_value.tobytes(order="C")),
                )
            )
        weights_path = temporary / _WEIGHTS_FILENAME
        np.savez_compressed(weights_path, **arrays)
        weights_sha256 = _sha256_file(weights_path)
        metadata = DualCTCArtifactMetadata(
            name=name,
            revision=revision,
            model_config=model_config,
            phone_inventory=phone_inventory,
            mora_inventory=mora_inventory,
            training_manifest_sha256=training_manifest_sha256,
            runtime_revision=runtime_revision,
            weights_filename=_WEIGHTS_FILENAME,
            weights_sha256=weights_sha256,
            tensors=tuple(specifications),
        )
        payload = {**_metadata_payload(metadata), "artifactDigest": metadata.digest}
        metadata_path = temporary / _METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with metadata_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return metadata
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_dual_ctc_metadata(directory: str | Path) -> DualCTCArtifactMetadata:
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dual CTC artifact path must be a directory")
    metadata_path = root / _METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("dual CTC metadata must be a JSON object")
    metadata = _metadata_from_dict(payload)
    weights_path = root / metadata.weights_filename
    if not weights_path.is_file():
        raise ValueError("dual CTC weights file is missing")
    if _sha256_file(weights_path) != metadata.weights_sha256:
        raise ValueError("dual CTC weights file digest mismatch")
    return metadata


def load_dual_ctc_artifact(
    directory: str | Path,
    *,
    device: str = "cpu",
) -> LoadedDualCTCArtifact:
    import numpy as np
    import torch

    from .torch_model import DualPhoneMoraCTC

    root = Path(directory).resolve(strict=True)
    metadata = read_dual_ctc_metadata(root)
    weights_path = root / metadata.weights_filename
    expected = {row.name: row for row in metadata.tensors}
    arrays: dict[str, np.ndarray] = {}
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("dual CTC tensor names do not match metadata")
        for name in sorted(archive.files):
            array_value = archive[name]
            specification = expected[name]
            if array_value.dtype.hasobject:
                raise TypeError("object tensor is not allowed")
            if tuple(array_value.shape) != specification.shape:
                raise ValueError(f"tensor shape mismatch: {name}")
            if str(array_value.dtype) != specification.dtype:
                raise ValueError(f"tensor dtype mismatch: {name}")
            if _sha256_bytes(array_value.tobytes(order="C")) != specification.sha256:
                raise ValueError(f"tensor digest mismatch: {name}")
            arrays[name] = array_value.copy()
    model = DualPhoneMoraCTC(
        metadata.model_config,
        metadata.phone_inventory,
        metadata.mora_inventory,
    )
    state = {name: torch.from_numpy(value) for name, value in arrays.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedDualCTCArtifact(model=model, metadata=metadata, directory=root)
