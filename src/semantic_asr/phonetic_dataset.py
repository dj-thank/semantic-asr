"""Rights-aware, digest-bound feature manifests for joint phone/mora CTC training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .contracts import sha256_json
from .phonetic_training import PhoneticLabelInventory

SplitName = Literal["train", "calibration", "test"]


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PhoneticDatasetResourcePolicy:
    maximum_items: int = 2_000_000
    maximum_frames_per_item: int = 60_000
    maximum_feature_dimension: int = 16_384
    maximum_total_feature_cells: int = 10_000_000_000
    maximum_target_labels: int = 20_000

    def __post_init__(self) -> None:
        for name in (
            "maximum_items",
            "maximum_frames_per_item",
            "maximum_feature_dimension",
            "maximum_total_feature_cells",
            "maximum_target_labels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticFeatureItem:
    utterance_id: str
    split: SplitName
    feature_path: str
    feature_sha256: str
    frame_count: int
    feature_dimension: int
    feature_dtype: str
    phone_targets: tuple[int, ...]
    mora_targets: tuple[int, ...]
    phone_inventory_digest: str
    mora_inventory_digest: str
    speaker_id: str
    source_id: str
    source_audio_sha256: str
    feature_revision: str
    rights_decision: str
    license_id: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.feature_path:
            raise ValueError("phonetic feature item requires utterance_id and feature_path")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("phonetic feature item has an invalid split")
        if Path(self.feature_path).is_absolute():
            raise ValueError("feature_path must be relative to the manifest directory")
        if ".." in Path(self.feature_path).parts:
            raise ValueError("feature_path must not traverse outside the manifest directory")
        for digest in (
            self.feature_sha256,
            self.phone_inventory_digest,
            self.mora_inventory_digest,
            self.source_audio_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("phonetic feature item contains an invalid SHA-256")
        for name in ("frame_count", "feature_dimension"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.feature_dtype not in {"float16", "float32", "float64"}:
            raise ValueError("feature_dtype must be float16, float32, or float64")
        if not self.phone_targets or not self.mora_targets:
            raise ValueError("phone and mora targets must both be non-empty")
        if any(isinstance(value, bool) or value < 0 for value in self.phone_targets):
            raise ValueError("phone targets must be non-negative integer IDs")
        if any(isinstance(value, bool) or value < 0 for value in self.mora_targets):
            raise ValueError("mora targets must be non-negative integer IDs")
        if not self.speaker_id or not self.source_id or not self.feature_revision:
            raise ValueError("speaker, source, and feature revision are required")
        if self.rights_decision != "allow":
            raise ValueError("phonetic feature training requires rights_decision='allow'")
        if not self.license_id:
            raise ValueError("license_id is required")
        object.__setattr__(self, "phone_targets", tuple(int(value) for value in self.phone_targets))
        object.__setattr__(self, "mora_targets", tuple(int(value) for value in self.mora_targets))

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def validate_inventories(
        self,
        phone: PhoneticLabelInventory,
        mora: PhoneticLabelInventory,
    ) -> None:
        if self.phone_inventory_digest != phone.digest:
            raise ValueError("feature item is bound to a different phone inventory")
        if self.mora_inventory_digest != mora.digest:
            raise ValueError("feature item is bound to a different mora inventory")
        if any(value >= len(phone.labels) for value in self.phone_targets):
            raise ValueError("phone target ID is outside the frozen inventory")
        if any(value >= len(mora.labels) for value in self.mora_targets):
            raise ValueError("mora target ID is outside the frozen inventory")
        if phone.blank_index in self.phone_targets:
            raise ValueError("phone targets must not contain the CTC blank label")
        if mora.blank_index in self.mora_targets:
            raise ValueError("mora targets must not contain the CTC blank label")


@dataclass(frozen=True, slots=True)
class PhoneticFeatureManifest:
    path: Path
    split: SplitName
    items: tuple[PhoneticFeatureItem, ...]
    manifest_sha256: str
    feature_revision: str
    phone_inventory_digest: str
    mora_inventory_digest: str
    resource_policy_digest: str

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("phonetic feature manifest must not be empty")
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA-256 value")
        if not _is_sha256(self.resource_policy_digest):
            raise ValueError("resource_policy_digest must be a SHA-256 value")
        if any(item.split != self.split for item in self.items):
            raise ValueError("manifest contains an item from a different split")
        if len({item.utterance_id for item in self.items}) != len(self.items):
            raise ValueError("utterance IDs must be unique within a manifest")
        if len({item.feature_path for item in self.items}) != len(self.items):
            raise ValueError("feature paths must be unique within a manifest")
        if any(item.feature_revision != self.feature_revision for item in self.items):
            raise ValueError("manifest mixes feature revisions")
        if any(
            item.phone_inventory_digest != self.phone_inventory_digest for item in self.items
        ):
            raise ValueError("manifest mixes phone inventories")
        if any(
            item.mora_inventory_digest != self.mora_inventory_digest for item in self.items
        ):
            raise ValueError("manifest mixes mora inventories")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "path": self.path.name,
                "split": self.split,
                "itemDigests": [item.digest for item in self.items],
                "manifestSha256": self.manifest_sha256,
                "featureRevision": self.feature_revision,
                "phoneInventoryDigest": self.phone_inventory_digest,
                "moraInventoryDigest": self.mora_inventory_digest,
                "resourcePolicyDigest": self.resource_policy_digest,
            }
        )

    @property
    def root(self) -> Path:
        return self.path.resolve().parent

    def resolve_feature(self, item: PhoneticFeatureItem) -> Path:
        resolved = (self.root / item.feature_path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("resolved feature path escapes the manifest directory") from exc
        return resolved


def _exact_row(row: dict[str, object], line_number: int) -> PhoneticFeatureItem:
    expected = {
        "schemaVersion",
        "utteranceId",
        "split",
        "featurePath",
        "featureSha256",
        "frameCount",
        "featureDimension",
        "featureDtype",
        "phoneTargets",
        "moraTargets",
        "phoneInventoryDigest",
        "moraInventoryDigest",
        "speakerId",
        "sourceId",
        "sourceAudioSha256",
        "featureRevision",
        "rightsDecision",
        "licenseId",
    }
    if set(row) != expected:
        missing = sorted(expected - set(row))
        unknown = sorted(set(row) - expected)
        raise ValueError(
            f"phonetic feature row {line_number} has non-exact schema; "
            f"missing={missing}, unknown={unknown}"
        )
    return PhoneticFeatureItem(
        schema_version=str(row["schemaVersion"]),
        utterance_id=str(row["utteranceId"]),
        split=str(row["split"]),  # type: ignore[arg-type]
        feature_path=str(row["featurePath"]),
        feature_sha256=str(row["featureSha256"]),
        frame_count=int(row["frameCount"]),
        feature_dimension=int(row["featureDimension"]),
        feature_dtype=str(row["featureDtype"]),
        phone_targets=tuple(int(value) for value in row["phoneTargets"]),  # type: ignore[union-attr]
        mora_targets=tuple(int(value) for value in row["moraTargets"]),  # type: ignore[union-attr]
        phone_inventory_digest=str(row["phoneInventoryDigest"]),
        mora_inventory_digest=str(row["moraInventoryDigest"]),
        speaker_id=str(row["speakerId"]),
        source_id=str(row["sourceId"]),
        source_audio_sha256=str(row["sourceAudioSha256"]),
        feature_revision=str(row["featureRevision"]),
        rights_decision=str(row["rightsDecision"]),
        license_id=str(row["licenseId"]),
    )


def load_phonetic_feature_manifest(
    path: str | Path,
    *,
    split: SplitName,
    phone_inventory: PhoneticLabelInventory,
    mora_inventory: PhoneticLabelInventory,
    resources: PhoneticDatasetResourcePolicy | None = None,
) -> PhoneticFeatureManifest:
    resources = resources or PhoneticDatasetResourcePolicy()
    manifest_path = Path(path)
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"phonetic feature row {line_number} must be an object")
            item = _exact_row(value, line_number)
            if item.split != split:
                raise ValueError(
                    f"phonetic feature row {line_number} declares split {item.split!r}, "
                    f"expected {split!r}"
                )
            item.validate_inventories(phone_inventory, mora_inventory)
            rows.append(item)
            if len(rows) > resources.maximum_items:
                raise ValueError("phonetic manifest exceeds maximum_items")
    if not rows:
        raise ValueError("phonetic feature manifest contains no items")
    total_cells = 0
    for item in rows:
        if item.frame_count > resources.maximum_frames_per_item:
            raise ValueError("phonetic feature item exceeds maximum_frames_per_item")
        if item.feature_dimension > resources.maximum_feature_dimension:
            raise ValueError("phonetic feature item exceeds maximum_feature_dimension")
        if len(item.phone_targets) > resources.maximum_target_labels:
            raise ValueError("phone target sequence exceeds maximum_target_labels")
        if len(item.mora_targets) > resources.maximum_target_labels:
            raise ValueError("mora target sequence exceeds maximum_target_labels")
        total_cells += item.frame_count * item.feature_dimension
        if total_cells > resources.maximum_total_feature_cells:
            raise ValueError("phonetic manifest exceeds maximum_total_feature_cells")
    return PhoneticFeatureManifest(
        path=manifest_path,
        split=split,
        items=tuple(rows),
        manifest_sha256=file_sha256(manifest_path),
        feature_revision=rows[0].feature_revision,
        phone_inventory_digest=phone_inventory.digest,
        mora_inventory_digest=mora_inventory.digest,
        resource_policy_digest=resources.digest,
    )


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return duplicate


def validate_phonetic_split_disjointness(
    train: PhoneticFeatureManifest,
    calibration: PhoneticFeatureManifest,
    test: PhoneticFeatureManifest,
) -> None:
    manifests = (train, calibration, test)
    if tuple(manifest.split for manifest in manifests) != ("train", "calibration", "test"):
        raise ValueError("phonetic manifests must be supplied as train, calibration, and test")
    if len({manifest.manifest_sha256 for manifest in manifests}) != 3:
        raise ValueError("phonetic train, calibration, and test files must differ")
    if len({manifest.feature_revision for manifest in manifests}) != 1:
        raise ValueError("phonetic splits use different feature revisions")
    if len({manifest.phone_inventory_digest for manifest in manifests}) != 1:
        raise ValueError("phonetic splits use different phone inventories")
    if len({manifest.mora_inventory_digest for manifest in manifests}) != 1:
        raise ValueError("phonetic splits use different mora inventories")

    dimensions = {item.feature_dimension for manifest in manifests for item in manifest.items}
    if len(dimensions) != 1:
        raise ValueError("phonetic splits use inconsistent feature dimensions")

    fields = {
        "utterance": lambda item: item.utterance_id,
        "source-audio": lambda item: item.source_audio_sha256,
        "speaker": lambda item: item.speaker_id,
        "source": lambda item: item.source_id,
        "feature-digest": lambda item: item.feature_sha256,
    }
    for name, getter in fields.items():
        sets = [set(getter(item) for item in manifest.items) for manifest in manifests]
        contamination = sets[0].intersection(sets[1]) | sets[0].intersection(sets[2]) | sets[1].intersection(sets[2])
        if contamination:
            raise ValueError(f"phonetic split leakage in {name}: {sorted(contamination)[:8]}")


def load_feature_array(
    manifest: PhoneticFeatureManifest,
    item: PhoneticFeatureItem,
):
    """Load one digest-verified numeric `.npy` feature array with pickle disabled."""

    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("loading phonetic features requires NumPy") from exc
    path = manifest.resolve_feature(item)
    if path.suffix.lower() != ".npy":
        raise ValueError("phonetic features must use the .npy format")
    if file_sha256(path) != item.feature_sha256:
        raise ValueError("phonetic feature SHA-256 mismatch")
    values = numpy.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape != (item.frame_count, item.feature_dimension):
        raise ValueError("phonetic feature shape mismatch")
    if str(values.dtype) != item.feature_dtype:
        raise ValueError("phonetic feature dtype mismatch")
    if values.dtype.kind != "f" or not numpy.isfinite(values).all():
        raise ValueError("phonetic features must be finite floating-point values")
    return values
