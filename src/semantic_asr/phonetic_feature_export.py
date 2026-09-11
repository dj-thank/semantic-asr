"""Rights-aware export of frozen audio features and Japanese phone/mora targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from .audio_posterior_adapters import canonical_audio_sha256
from .contracts import sha256_json
from .japanese_phonetic_targets import (
    JapanesePronunciationPolicy,
    japanese_pronunciation_target,
)
from .joint_phonetic_runtime_optional import (
    FrozenAudioFeatureBackend,
    FrozenFeatureMatrix,
)
from .phonetic_dataset import PhoneticFeatureItem, file_sha256

SplitName = Literal["train", "calibration", "test"]


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _exact_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class PhoneticSourceResourcePolicy:
    maximum_items: int = 2_000_000
    maximum_reading_characters: int = 20_000
    maximum_segment_duration_ms: int = 120_000
    maximum_total_audio_samples: int = 100_000_000_000
    maximum_recording_samples: int = 2_000_000_000

    def __post_init__(self) -> None:
        for name in (
            "maximum_items",
            "maximum_reading_characters",
            "maximum_segment_duration_ms",
            "maximum_total_audio_samples",
            "maximum_recording_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticSourceItem:
    utterance_id: str
    split: SplitName
    audio_path: str
    audio_sha256: str
    sample_rate: int
    segment_start_ms: int
    segment_end_ms: int
    reading: str
    speaker_id: str
    source_id: str
    rights_decision: str
    license_id: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("phonetic source schema_version must be '1'")
        if not self.utterance_id or not self.audio_path:
            raise ValueError("phonetic source item requires utterance_id and audio_path")
        path = Path(self.audio_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("audio_path must be a non-traversing relative path")
        if not _is_sha256(self.audio_sha256):
            raise ValueError("audio_sha256 must be a SHA-256 value")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("phonetic source split is invalid")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if (
            isinstance(self.segment_start_ms, bool)
            or isinstance(self.segment_end_ms, bool)
            or self.segment_start_ms < 0
            or self.segment_end_ms <= self.segment_start_ms
        ):
            raise ValueError("source segment requires 0 <= start_ms < end_ms")
        if not self.reading:
            raise ValueError("an explicit kana reading is required")
        if not self.speaker_id or not self.source_id:
            raise ValueError("speaker_id and source_id are required")
        if self.rights_decision != "allow":
            raise ValueError("feature export requires rights_decision='allow'")
        if not self.license_id:
            raise ValueError("license_id is required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticSourceManifest:
    path: Path
    split: SplitName
    items: tuple[PhoneticSourceItem, ...]
    manifest_sha256: str
    resource_policy_digest: str

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("phonetic source manifest must not be empty")
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA-256 value")
        if not _is_sha256(self.resource_policy_digest):
            raise ValueError("resource_policy_digest must be a SHA-256 value")
        if any(item.split != self.split for item in self.items):
            raise ValueError("source manifest mixes splits")
        if len({item.utterance_id for item in self.items}) != len(self.items):
            raise ValueError("utterance IDs must be unique within a source manifest")

    @property
    def root(self) -> Path:
        return self.path.resolve().parent

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "path": self.path.name,
                "split": self.split,
                "itemDigests": [item.digest for item in self.items],
                "manifestSha256": self.manifest_sha256,
                "resourcePolicyDigest": self.resource_policy_digest,
            }
        )

    def resolve_audio(self, item: PhoneticSourceItem) -> Path:
        resolved = (self.root / item.audio_path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("resolved audio path escapes the source manifest directory") from exc
        return resolved


@dataclass(frozen=True, slots=True)
class LoadedSourceRecording:
    samples: tuple[float, ...]
    sample_rate: int
    file_sha256: str
    source_name: str

    def __post_init__(self) -> None:
        if not self.samples or not self.source_name:
            raise ValueError("loaded source recording must be non-empty")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("loaded source sample_rate must be positive")
        if not _is_sha256(self.file_sha256):
            raise ValueError("loaded source file_sha256 must be a SHA-256 value")
        values = tuple(float(value) for value in self.samples)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("loaded source samples must be finite")
        object.__setattr__(self, "samples", values)


class SourceAudioLoader(Protocol):
    def load(self, path: str | Path) -> LoadedSourceRecording: ...


class SoundFileSourceAudioLoader:
    """Decode one explicitly mono file without downmixing or resampling."""

    def load(self, path: str | Path) -> LoadedSourceRecording:
        try:
            import soundfile
        except ImportError as exc:  # pragma: no cover - optional export dependency
            raise RuntimeError("SoundFileSourceAudioLoader requires soundfile") from exc
        source = Path(path)
        before = file_sha256(source)
        values, sample_rate = soundfile.read(source, dtype="float32", always_2d=True)
        after = file_sha256(source)
        if before != after:
            raise ValueError("source audio changed while being decoded")
        if values.shape[1] != 1:
            raise ValueError("phonetic feature export requires explicitly mono audio")
        return LoadedSourceRecording(
            samples=tuple(float(value) for value in values[:, 0]),
            sample_rate=int(sample_rate),
            file_sha256=before,
            source_name=source.name,
        )


@dataclass(frozen=True, slots=True)
class PhoneticFeatureExportConfig:
    feature_dtype: str = "float32"
    feature_subdirectory: str = "features"
    maximum_cached_recordings: int = 2
    fsync_each_row: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.feature_dtype not in {"float16", "float32", "float64"}:
            raise ValueError("feature_dtype must be float16, float32, or float64")
        path = Path(self.feature_subdirectory)
        if path.is_absolute() or ".." in path.parts or not self.feature_subdirectory:
            raise ValueError("feature_subdirectory must be a safe relative path")
        if isinstance(self.maximum_cached_recordings, bool) or self.maximum_cached_recordings < 1:
            raise ValueError("maximum_cached_recordings must be positive")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticFeatureReceipt:
    utterance_id: str
    source_item_digest: str
    source_manifest_digest: str
    source_recording_file_sha256: str
    source_clip_sha256: str
    sample_start: int
    sample_end: int
    sample_rate: int
    pronunciation_target_digest: str
    phone_inventory_digest: str
    mora_inventory_digest: str
    feature_backend_config_digest: str
    feature_matrix_digest: str
    feature_path: str
    feature_sha256: str
    feature_revision: str
    export_config_digest: str

    def __post_init__(self) -> None:
        if not self.utterance_id or not self.feature_path or not self.feature_revision:
            raise ValueError("feature receipt requires utterance and feature identity")
        if not 0 <= self.sample_start < self.sample_end:
            raise ValueError("feature receipt sample range is invalid")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("feature receipt sample_rate must be positive")
        for digest in (
            self.source_item_digest,
            self.source_manifest_digest,
            self.source_recording_file_sha256,
            self.source_clip_sha256,
            self.pronunciation_target_digest,
            self.phone_inventory_digest,
            self.mora_inventory_digest,
            self.feature_backend_config_digest,
            self.feature_matrix_digest,
            self.feature_sha256,
            self.export_config_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("feature receipt contains an invalid digest")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticFeatureExportResult:
    output_manifest: Path
    output_manifest_sha256: str
    item_count: int
    source_manifest_digest: str
    feature_backend_config_digest: str
    pronunciation_policy_digest: str
    phone_inventory_digest: str
    mora_inventory_digest: str
    feature_revision: str
    export_config_digest: str
    run_digest: str
    receipt_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.item_count < 1:
            raise ValueError("feature export result item_count must be positive")
        for digest in (
            self.output_manifest_sha256,
            self.source_manifest_digest,
            self.feature_backend_config_digest,
            self.pronunciation_policy_digest,
            self.phone_inventory_digest,
            self.mora_inventory_digest,
            self.export_config_digest,
            self.run_digest,
            *self.receipt_digests,
        ):
            if not _is_sha256(digest):
                raise ValueError("feature export result contains an invalid digest")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "output_manifest": self.output_manifest.name,
            }
        )


def _source_row(row: dict[str, object], line_number: int) -> PhoneticSourceItem:
    expected = {
        "schemaVersion",
        "utteranceId",
        "split",
        "audioPath",
        "audioSha256",
        "sampleRate",
        "segmentStartMs",
        "segmentEndMs",
        "reading",
        "speakerId",
        "sourceId",
        "rightsDecision",
        "licenseId",
    }
    if set(row) != expected:
        raise ValueError(
            f"source row {line_number} has non-exact schema; "
            f"missing={sorted(expected - set(row))}, unknown={sorted(set(row) - expected)}"
        )
    return PhoneticSourceItem(
        schema_version=str(row["schemaVersion"]),
        utterance_id=str(row["utteranceId"]),
        split=str(row["split"]),  # type: ignore[arg-type]
        audio_path=str(row["audioPath"]),
        audio_sha256=str(row["audioSha256"]),
        sample_rate=_exact_int(row["sampleRate"], name="sampleRate"),
        segment_start_ms=_exact_int(row["segmentStartMs"], name="segmentStartMs"),
        segment_end_ms=_exact_int(row["segmentEndMs"], name="segmentEndMs"),
        reading=str(row["reading"]),
        speaker_id=str(row["speakerId"]),
        source_id=str(row["sourceId"]),
        rights_decision=str(row["rightsDecision"]),
        license_id=str(row["licenseId"]),
    )


def load_phonetic_source_manifest(
    path: str | Path,
    *,
    split: SplitName,
    resources: PhoneticSourceResourcePolicy | None = None,
) -> PhoneticSourceManifest:
    resources = resources or PhoneticSourceResourcePolicy()
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"source row {line_number} must be an object")
            item = _source_row(value, line_number)
            if item.split != split:
                raise ValueError(
                    f"source row {line_number} declares split {item.split!r}, expected {split!r}"
                )
            if len(item.reading) > resources.maximum_reading_characters:
                raise ValueError("source reading exceeds maximum_reading_characters")
            if item.segment_end_ms - item.segment_start_ms > resources.maximum_segment_duration_ms:
                raise ValueError("source segment exceeds maximum_segment_duration_ms")
            rows.append(item)
            if len(rows) > resources.maximum_items:
                raise ValueError("source manifest exceeds maximum_items")
    return PhoneticSourceManifest(
        path=source,
        split=split,
        items=tuple(rows),
        manifest_sha256=file_sha256(source),
        resource_policy_digest=resources.digest,
    )


def _feature_row(item: PhoneticFeatureItem) -> dict[str, object]:
    return {
        "schemaVersion": item.schema_version,
        "utteranceId": item.utterance_id,
        "split": item.split,
        "featurePath": item.feature_path,
        "featureSha256": item.feature_sha256,
        "frameCount": item.frame_count,
        "featureDimension": item.feature_dimension,
        "featureDtype": item.feature_dtype,
        "phoneTargets": item.phone_targets,
        "moraTargets": item.mora_targets,
        "phoneInventoryDigest": item.phone_inventory_digest,
        "moraInventoryDigest": item.mora_inventory_digest,
        "speakerId": item.speaker_id,
        "sourceId": item.source_id,
        "sourceAudioSha256": item.source_audio_sha256,
        "featureRevision": item.feature_revision,
        "rightsDecision": item.rights_decision,
        "licenseId": item.license_id,
    }


def _feature_item(row: dict[str, object]) -> PhoneticFeatureItem:
    expected = set(_feature_row.__annotations__) if False else {
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
        raise ValueError("checkpoint feature row schema is not exact")
    return PhoneticFeatureItem(
        schema_version=str(row["schemaVersion"]),
        utterance_id=str(row["utteranceId"]),
        split=str(row["split"]),  # type: ignore[arg-type]
        feature_path=str(row["featurePath"]),
        feature_sha256=str(row["featureSha256"]),
        frame_count=_exact_int(row["frameCount"], name="frameCount"),
        feature_dimension=_exact_int(row["featureDimension"], name="featureDimension"),
        feature_dtype=str(row["featureDtype"]),
        phone_targets=tuple(row["phoneTargets"]),  # type: ignore[arg-type]
        mora_targets=tuple(row["moraTargets"]),  # type: ignore[arg-type]
        phone_inventory_digest=str(row["phoneInventoryDigest"]),
        mora_inventory_digest=str(row["moraInventoryDigest"]),
        speaker_id=str(row["speakerId"]),
        source_id=str(row["sourceId"]),
        source_audio_sha256=str(row["sourceAudioSha256"]),
        feature_revision=str(row["featureRevision"]),
        rights_decision=str(row["rightsDecision"]),
        license_id=str(row["licenseId"]),
    )


class PhoneticFeatureExporter:
    def __init__(
        self,
        *,
        feature_backend: FrozenAudioFeatureBackend,
        pronunciation_policy: JapanesePronunciationPolicy | None = None,
        audio_loader: SourceAudioLoader | None = None,
        config: PhoneticFeatureExportConfig | None = None,
        source_resources: PhoneticSourceResourcePolicy | None = None,
    ) -> None:
        self.feature_backend = feature_backend
        self.pronunciation_policy = pronunciation_policy or JapanesePronunciationPolicy()
        self.audio_loader = audio_loader or SoundFileSourceAudioLoader()
        self.config = config or PhoneticFeatureExportConfig()
        self.source_resources = source_resources or PhoneticSourceResourcePolicy()
        self.phone_inventory, self.mora_inventory = self.pronunciation_policy.inventories()
        self._cache: OrderedDict[str, LoadedSourceRecording] = OrderedDict()

    @property
    def feature_revision(self) -> str:
        digest = sha256_json(
            {
                "featureBackendConfigDigest": self.feature_backend.config.digest,
                "pronunciationPolicyDigest": self.pronunciation_policy.digest,
                "exportConfigDigest": self.config.digest,
            }
        )
        return f"phonetic-feature-export-v1:{digest}"

    def _load(self, path: Path) -> LoadedSourceRecording:
        key = str(path)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = self.audio_loader.load(path)
        self._cache[key] = value
        while len(self._cache) > self.config.maximum_cached_recordings:
            self._cache.popitem(last=False)
        return value

    def _run_digest(self, source: PhoneticSourceManifest, output: Path) -> str:
        return sha256_json(
            {
                "sourceManifestDigest": source.digest,
                "outputManifestName": output.name,
                "featureBackendConfigDigest": self.feature_backend.config.digest,
                "pronunciationPolicyDigest": self.pronunciation_policy.digest,
                "phoneInventoryDigest": self.phone_inventory.digest,
                "moraInventoryDigest": self.mora_inventory.digest,
                "featureRevision": self.feature_revision,
                "exportConfigDigest": self.config.digest,
                "sourceResourcePolicyDigest": self.source_resources.digest,
            }
        )

    def _recording(self, source: PhoneticSourceManifest, item: PhoneticSourceItem):
        path = source.resolve_audio(item)
        value = self._load(path)
        if value.file_sha256 != item.audio_sha256:
            raise ValueError("source recording SHA-256 mismatch")
        if value.sample_rate != item.sample_rate:
            raise ValueError("source recording sample rate mismatch")
        if value.sample_rate != self.feature_backend.config.sample_rate:
            raise ValueError("source sample rate differs from the frozen feature backend")
        if len(value.samples) > self.source_resources.maximum_recording_samples:
            raise ValueError("source recording exceeds maximum_recording_samples")
        return value

    def _export_item(
        self,
        source: PhoneticSourceManifest,
        item: PhoneticSourceItem,
        *,
        output_root: Path,
    ) -> tuple[PhoneticFeatureItem, PhoneticFeatureReceipt]:
        recording = self._recording(source, item)
        start = math.floor(item.segment_start_ms * recording.sample_rate / 1000)
        end = math.ceil(item.segment_end_ms * recording.sample_rate / 1000)
        if not 0 <= start < end <= len(recording.samples):
            raise ValueError("source segment lies outside the decoded recording")
        clip = recording.samples[start:end]
        clip_sha256 = canonical_audio_sha256(clip, recording.sample_rate)
        target = japanese_pronunciation_target(
            item.reading,
            policy=self.pronunciation_policy,
        )
        phone_targets, mora_targets = target.target_ids(
            self.phone_inventory,
            self.mora_inventory,
        )
        matrix = self.feature_backend.extract_features(
            clip,
            sample_rate=recording.sample_rate,
            source_audio_sha256=clip_sha256,
        )
        if matrix.source_audio_sha256 != clip_sha256:
            raise ValueError("feature matrix is bound to a different audio clip")
        if matrix.feature_config_digest != self.feature_backend.config.digest:
            raise ValueError("feature matrix is bound to a different backend config")
        if matrix.feature_dimension != self.feature_backend.config.feature_dimension:
            raise ValueError("feature matrix dimension differs from frozen backend config")
        if matrix.frame_count < len(phone_targets) + sum(
            left == right for left, right in zip(phone_targets, phone_targets[1:], strict=False)
        ):
            raise ValueError("phone targets cannot align to the emitted feature frames")
        if matrix.frame_count < len(mora_targets) + sum(
            left == right for left, right in zip(mora_targets, mora_targets[1:], strict=False)
        ):
            raise ValueError("mora targets cannot align to the emitted feature frames")
        try:
            import numpy
        except ImportError as exc:  # pragma: no cover - optional export dependency
            raise RuntimeError("phonetic feature export requires NumPy") from exc
        array = numpy.asarray(matrix.values, dtype=self.config.feature_dtype)
        if array.ndim != 2 or not numpy.isfinite(array).all():
            raise ValueError("feature backend emitted an invalid numeric matrix")
        feature_id = sha256_json(
            {
                "sourceItemDigest": item.digest,
                "clipSha256": clip_sha256,
                "featureBackendConfigDigest": self.feature_backend.config.digest,
                "pronunciationTargetDigest": target.digest,
                "featureRevision": self.feature_revision,
            }
        )
        relative = Path(self.config.feature_subdirectory) / f"{feature_id[:32]}.npy"
        feature_path = (output_root / relative).resolve()
        try:
            feature_path.relative_to(output_root.resolve())
        except ValueError as exc:
            raise ValueError("feature output path escapes the output root") from exc
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = feature_path.with_name(feature_path.name + ".tmp")
        with temporary.open("wb") as handle:
            numpy.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, feature_path)
        feature_sha = file_sha256(feature_path)
        output_item = PhoneticFeatureItem(
            utterance_id=item.utterance_id,
            split=item.split,
            feature_path=relative.as_posix(),
            feature_sha256=feature_sha,
            frame_count=matrix.frame_count,
            feature_dimension=matrix.feature_dimension,
            feature_dtype=str(array.dtype),
            phone_targets=phone_targets,
            mora_targets=mora_targets,
            phone_inventory_digest=self.phone_inventory.digest,
            mora_inventory_digest=self.mora_inventory.digest,
            speaker_id=item.speaker_id,
            source_id=item.source_id,
            source_audio_sha256=clip_sha256,
            feature_revision=self.feature_revision,
            rights_decision=item.rights_decision,
            license_id=item.license_id,
        )
        receipt = PhoneticFeatureReceipt(
            utterance_id=item.utterance_id,
            source_item_digest=item.digest,
            source_manifest_digest=source.digest,
            source_recording_file_sha256=recording.file_sha256,
            source_clip_sha256=clip_sha256,
            sample_start=start,
            sample_end=end,
            sample_rate=recording.sample_rate,
            pronunciation_target_digest=target.digest,
            phone_inventory_digest=self.phone_inventory.digest,
            mora_inventory_digest=self.mora_inventory.digest,
            feature_backend_config_digest=self.feature_backend.config.digest,
            feature_matrix_digest=matrix.digest,
            feature_path=relative.as_posix(),
            feature_sha256=feature_sha,
            feature_revision=self.feature_revision,
            export_config_digest=self.config.digest,
        )
        sidecar = feature_path.with_suffix(".receipt.json")
        _atomic_write_text(
            sidecar,
            json.dumps(
                {**asdict(receipt), "receiptDigest": receipt.digest},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        return output_item, receipt

    def export(
        self,
        source: PhoneticSourceManifest,
        output_manifest: str | Path,
        *,
        allow_derived_export: bool,
        resume: bool = True,
    ) -> PhoneticFeatureExportResult:
        if not allow_derived_export:
            raise PermissionError("phonetic feature export requires allow_derived_export=True")
        output = Path(output_manifest)
        if output.suffix != ".jsonl":
            raise ValueError("output manifest must use the .jsonl suffix")
        output_root = output.resolve().parent
        if output_root == Path(output_root.anchor):
            raise ValueError("output manifest must not be written at filesystem root")
        output_root.mkdir(parents=True, exist_ok=True)
        run_digest = self._run_digest(source, output)
        partial = output.with_suffix(output.suffix + ".partial")
        checkpoint = output.with_suffix(output.suffix + ".partial.meta.json")
        final_meta = output.with_suffix(output.suffix + ".export.json")
        checkpoint_payload = {
            "schemaVersion": "1",
            "runDigest": run_digest,
            "sourceManifestDigest": source.digest,
            "featureBackendConfigDigest": self.feature_backend.config.digest,
            "pronunciationPolicyDigest": self.pronunciation_policy.digest,
            "phoneInventoryDigest": self.phone_inventory.digest,
            "moraInventoryDigest": self.mora_inventory.digest,
            "featureRevision": self.feature_revision,
            "exportConfigDigest": self.config.digest,
        }
        if output.exists():
            if not final_meta.exists():
                raise ValueError("completed output manifest is missing its export metadata")
            metadata = json.loads(final_meta.read_text(encoding="utf-8"))
            if metadata.get("runDigest") != run_digest:
                raise ValueError("completed output belongs to a different export run")
            if metadata.get("outputManifestSha256") != file_sha256(output):
                raise ValueError("completed output manifest SHA-256 mismatch")
            return PhoneticFeatureExportResult(
                output_manifest=output,
                output_manifest_sha256=metadata["outputManifestSha256"],
                item_count=int(metadata["itemCount"]),
                source_manifest_digest=source.digest,
                feature_backend_config_digest=self.feature_backend.config.digest,
                pronunciation_policy_digest=self.pronunciation_policy.digest,
                phone_inventory_digest=self.phone_inventory.digest,
                mora_inventory_digest=self.mora_inventory.digest,
                feature_revision=self.feature_revision,
                export_config_digest=self.config.digest,
                run_digest=run_digest,
                receipt_digests=tuple(metadata["receiptDigests"]),
            )
        completed: list[PhoneticFeatureItem] = []
        receipt_digests: list[str] = []
        if partial.exists():
            if not resume or not checkpoint.exists():
                raise ValueError("partial export exists without an enabled matching checkpoint")
            metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
            if metadata != checkpoint_payload:
                raise ValueError("partial export checkpoint belongs to a different run")
            with partial.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if not line.strip():
                        continue
                    if index >= len(source.items):
                        raise ValueError("partial export contains more rows than the source manifest")
                    item = _feature_item(json.loads(line))
                    source_item = source.items[index]
                    if item.utterance_id != source_item.utterance_id:
                        raise ValueError("partial export is not an exact source-manifest prefix")
                    feature = output_root / item.feature_path
                    if not feature.exists() or file_sha256(feature) != item.feature_sha256:
                        raise ValueError("partial export feature file is absent or corrupted")
                    sidecar = feature.with_suffix(".receipt.json")
                    payload = json.loads(sidecar.read_text(encoding="utf-8"))
                    receipt_digest = payload.pop("receiptDigest")
                    if sha256_json(payload) != receipt_digest:
                        raise ValueError("partial export receipt digest mismatch")
                    completed.append(item)
                    receipt_digests.append(receipt_digest)
        else:
            _atomic_write_text(
                checkpoint,
                json.dumps(checkpoint_payload, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
        mode = "a" if completed else "w"
        total_samples = 0
        with partial.open(mode, encoding="utf-8", newline="\n") as handle:
            for source_item in source.items[len(completed) :]:
                output_item, receipt = self._export_item(
                    source,
                    source_item,
                    output_root=output_root,
                )
                total_samples += receipt.sample_end - receipt.sample_start
                if total_samples > self.source_resources.maximum_total_audio_samples:
                    raise ValueError("feature export exceeds maximum_total_audio_samples")
                handle.write(
                    json.dumps(_feature_row(output_item), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                handle.flush()
                if self.config.fsync_each_row:
                    os.fsync(handle.fileno())
                completed.append(output_item)
                receipt_digests.append(receipt.digest)
        os.replace(partial, output)
        output_sha = file_sha256(output)
        result = PhoneticFeatureExportResult(
            output_manifest=output,
            output_manifest_sha256=output_sha,
            item_count=len(completed),
            source_manifest_digest=source.digest,
            feature_backend_config_digest=self.feature_backend.config.digest,
            pronunciation_policy_digest=self.pronunciation_policy.digest,
            phone_inventory_digest=self.phone_inventory.digest,
            mora_inventory_digest=self.mora_inventory.digest,
            feature_revision=self.feature_revision,
            export_config_digest=self.config.digest,
            run_digest=run_digest,
            receipt_digests=tuple(receipt_digests),
        )
        metadata = {
            "schemaVersion": "1",
            "runDigest": run_digest,
            "resultDigest": result.digest,
            "outputManifestSha256": output_sha,
            "itemCount": len(completed),
            "receiptDigests": receipt_digests,
            "phoneInventory": asdict(self.phone_inventory),
            "moraInventory": asdict(self.mora_inventory),
            "pronunciationPolicy": asdict(self.pronunciation_policy),
            "pronunciationPolicyDigest": self.pronunciation_policy.digest,
            "featureBackendConfig": asdict(self.feature_backend.config),
            "featureBackendConfigDigest": self.feature_backend.config.digest,
            "featureRevision": self.feature_revision,
            "sourceManifestDigest": source.digest,
            "exportConfig": asdict(self.config),
            "exportConfigDigest": self.config.digest,
            "claimBoundary": (
                "derived training features only; no model quality or runtime promotion claim"
            ),
        }
        _atomic_write_text(
            final_meta,
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        checkpoint.unlink(missing_ok=True)
        return result
