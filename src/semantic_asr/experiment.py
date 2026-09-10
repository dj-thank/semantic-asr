from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real
from statistics import fmean
from typing import Any, Literal, cast

from .audio import require_integer
from .rights import Operation, RightsRegistry

DatasetSplitRole = Literal["train", "dev", "calibration", "test", "regression-exposed"]
SplitName = DatasetSplitRole
ModelVisibleSplit = Literal["train", "dev", "calibration"]
ReferencePurpose = Literal["training", "context", "calibration"]
_SPLIT_NAMES = (
    "train",
    "dev",
    "calibration",
    "test",
    "regression-exposed",
)
_SPLIT_NAME_SET = frozenset(_SPLIT_NAMES)
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_REFERENCE_PURPOSE_SPLITS: dict[ReferencePurpose, frozenset[SplitName]] = {
    "training": frozenset(("train",)),
    "context": frozenset(("train", "dev")),
    "calibration": frozenset(("calibration",)),
}


def _require_split(value: Any, *, name: str = "split") -> SplitName:
    if not isinstance(value, str) or value not in _SPLIT_NAME_SET:
        raise ValueError(f"{name} must be exactly one of: {', '.join(_SPLIT_NAMES)}")
    return cast(SplitName, value)


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in _LOWERCASE_HEX for character in value
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _require_optional_identifier(value: Any, *, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string when provided")


def _canonical_string_tuple(
    value: Any,
    *,
    name: str,
    sha256: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    rows = tuple(value)
    for row in rows:
        if sha256:
            _require_sha256(row, name=f"{name} entry")
        else:
            _require_optional_identifier(row, name=f"{name} entry")
    if len(set(rows)) != len(rows):
        raise ValueError(f"{name} must not contain duplicates")
    return rows


def _normalized_reference_digest(value: str) -> str:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _identifier_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class UtteranceRecord:
    sample_id: str
    split: SplitName
    audio_sha256: str
    reference: str
    speaker_id: str | None = None
    source_recording_id: str | None = None
    duration_seconds: float | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pcm_sha256: str | None = None
    session_id: str | None = None
    reference_lineage_id: str | None = None
    derivation_group_id: str | None = None
    near_duplicate_id: str | None = None
    rights_asset_id: str | None = None
    source_dataset_revision: str | None = None
    parent_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id
            or self.sample_id != self.sample_id.strip()
            or not isinstance(self.reference, str)
            or not self.reference
        ):
            raise ValueError("sample_id, audio_sha256 and reference are required")
        _require_split(self.split)
        _require_sha256(self.audio_sha256, name="audio_sha256")
        if self.pcm_sha256 is not None:
            _require_sha256(self.pcm_sha256, name="pcm_sha256")
        for name in (
            "speaker_id",
            "source_recording_id",
            "session_id",
            "reference_lineage_id",
            "derivation_group_id",
            "near_duplicate_id",
            "rights_asset_id",
            "source_dataset_revision",
        ):
            _require_optional_identifier(getattr(self, name), name=name)
        parents = _canonical_string_tuple(self.parent_sample_ids, name="parent_sample_ids")
        if self.sample_id in parents:
            raise ValueError("parent_sample_ids cannot contain the sample itself")
        object.__setattr__(self, "parent_sample_ids", parents)
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be finite and positive")

    @property
    def reference_digest(self) -> str:
        return _normalized_reference_digest(self.reference)


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    kind: str
    value: str
    splits: tuple[SplitName, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    records: tuple[UtteranceRecord, ...]
    dataset_name: str
    dataset_revision: str
    rights_registry_digest: str | None = None
    split_policy: str = "semantic-asr-five-role-v1"
    split_seed: int = 0
    excluded_sample_ids: tuple[str, ...] = ()
    excluded_audio_sha256: tuple[str, ...] = ()
    excluded_reference_sha256: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.records
            or not isinstance(self.dataset_name, str)
            or not self.dataset_name
            or self.dataset_name != self.dataset_name.strip()
            or not isinstance(self.dataset_revision, str)
            or not self.dataset_revision
            or self.dataset_revision != self.dataset_revision.strip()
        ):
            raise ValueError("records, dataset_name and dataset_revision are required")
        if len({record.sample_id for record in self.records}) != len(self.records):
            raise ValueError("sample IDs must be globally unique")
        if self.rights_registry_digest is not None:
            _require_sha256(self.rights_registry_digest, name="rights_registry_digest")
        _require_optional_identifier(self.split_policy, name="split_policy")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise TypeError("split_seed must be an integer")
        object.__setattr__(
            self,
            "excluded_sample_ids",
            _canonical_string_tuple(self.excluded_sample_ids, name="excluded_sample_ids"),
        )
        object.__setattr__(
            self,
            "excluded_audio_sha256",
            _canonical_string_tuple(
                self.excluded_audio_sha256,
                name="excluded_audio_sha256",
                sha256=True,
            ),
        )
        object.__setattr__(
            self,
            "excluded_reference_sha256",
            _canonical_string_tuple(
                self.excluded_reference_sha256,
                name="excluded_reference_sha256",
                sha256=True,
            ),
        )

    @property
    def digest(self) -> str:
        payload = {
            "datasetName": self.dataset_name,
            "datasetRevision": self.dataset_revision,
            "rightsRegistryDigest": self.rights_registry_digest,
            "splitPolicy": self.split_policy,
            "splitSeed": self.split_seed,
            "exclusions": {
                "sampleIds": sorted(self.excluded_sample_ids),
                "audioSha256": sorted(self.excluded_audio_sha256),
                "referenceSha256": sorted(self.excluded_reference_sha256),
            },
            "records": [
                asdict(record) for record in sorted(self.records, key=lambda row: row.sample_id)
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def split(self, name: SplitName) -> tuple[UtteranceRecord, ...]:
        canonical = _require_split(name, name="requested split")
        return tuple(record for record in self.records if record.split == canonical)

    def model_visible_records(
        self,
        splits: Sequence[ModelVisibleSplit],
        *,
        purpose: ReferencePurpose,
    ) -> tuple[UtteranceRecord, ...]:
        if isinstance(splits, (str, bytes)):
            raise TypeError("model-visible splits must be a sequence")
        canonical = tuple(_require_split(value, name="model-visible split") for value in splits)
        if not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("model-visible splits must be non-empty and unique")
        allowed = _REFERENCE_PURPOSE_SPLITS.get(purpose)
        if allowed is None:
            raise ValueError(f"unknown reference purpose: {purpose!r}")
        forbidden = tuple(value for value in canonical if value not in allowed)
        if forbidden:
            raise PermissionError(
                f"reference purpose {purpose!r} forbids split(s): {', '.join(forbidden)}"
            )
        selected = tuple(record for record in self.records if record.split in canonical)
        if not selected:
            raise ValueError("requested model-visible cohort is empty")
        return selected

    def training_records(self) -> tuple[UtteranceRecord, ...]:
        return self.model_visible_records(("train",), purpose="training")

    def context_records(self) -> tuple[UtteranceRecord, ...]:
        return self.model_visible_records(("train", "dev"), purpose="context")

    def calibration_records(self) -> tuple[UtteranceRecord, ...]:
        return self.model_visible_records(("calibration",), purpose="calibration")

    def require_operation_rights(
        self,
        registry: RightsRegistry,
        operation: Operation,
        *,
        splits: Sequence[SplitName] | None = None,
    ) -> tuple[UtteranceRecord, ...]:
        if self.rights_registry_digest is None:
            raise ValueError("rights_registry_digest is required before checking an operation")
        if registry.digest != self.rights_registry_digest:
            raise ValueError("rights registry digest does not match the dataset manifest")
        if splits is None:
            selected = self.records
        else:
            if isinstance(splits, (str, bytes)):
                raise TypeError("rights-check splits must be a sequence")
            canonical = tuple(_require_split(value, name="rights-check split") for value in splits)
            if not canonical or len(set(canonical)) != len(canonical):
                raise ValueError("rights-check splits must be non-empty and unique")
            selected = tuple(record for record in self.records if record.split in canonical)
        if not selected:
            raise ValueError("rights check selected no records")
        asset_ids = set()
        for record in selected:
            if record.rights_asset_id is None:
                raise PermissionError(
                    f"unknown rights asset for sample {record.sample_id}; cannot {operation}"
                )
            asset_ids.add(record.rights_asset_id)
        for asset_id in sorted(asset_ids):
            registry.require(asset_id, operation)
        return selected

    def leakage_findings(
        self, *, reference_near_duplicate: bool = True
    ) -> tuple[LeakageFinding, ...]:
        findings: list[LeakageFinding] = []

        def collect(kind: str, values: Iterable[tuple[str | None, UtteranceRecord]]) -> None:
            groups: dict[str, list[UtteranceRecord]] = {}
            for value, record in values:
                if value:
                    groups.setdefault(value, []).append(record)
            for value, rows in groups.items():
                splits = tuple(sorted({row.split for row in rows}))
                if len(splits) > 1:
                    findings.append(
                        LeakageFinding(
                            kind=kind,
                            value=value,
                            splits=splits,
                            sample_ids=tuple(sorted(row.sample_id for row in rows)),
                        )
                    )

        collect("audio-sha256", ((record.audio_sha256, record) for record in self.records))
        collect("pcm-sha256", ((record.pcm_sha256, record) for record in self.records))
        collect("speaker-id", ((record.speaker_id, record) for record in self.records))
        collect("session-id", ((record.session_id, record) for record in self.records))
        collect(
            "source-recording-id",
            ((record.source_recording_id, record) for record in self.records),
        )
        collect(
            "reference-lineage-id",
            ((record.reference_lineage_id, record) for record in self.records),
        )
        collect(
            "derivation-group-id",
            ((record.derivation_group_id, record) for record in self.records),
        )
        collect(
            "near-duplicate-id",
            ((record.near_duplicate_id, record) for record in self.records),
        )
        if reference_near_duplicate:
            collect(
                "reference-digest",
                ((record.reference_digest, record) for record in self.records),
            )

        by_sample_id = {record.sample_id: record for record in self.records}
        for record in self.records:
            for parent_id in record.parent_sample_ids:
                parent = by_sample_id.get(parent_id)
                if parent is not None and parent.split != record.split:
                    findings.append(
                        LeakageFinding(
                            kind="parent-sample-id",
                            value=parent_id,
                            splits=tuple(sorted((parent.split, record.split))),
                            sample_ids=tuple(sorted((parent.sample_id, record.sample_id))),
                        )
                    )
        excluded_sample_ids = set(self.excluded_sample_ids)
        excluded_audio = set(self.excluded_audio_sha256)
        excluded_reference = set(self.excluded_reference_sha256)
        for record in self.split("test"):
            for kind, value, excluded in (
                ("excluded-sample-id", record.sample_id, excluded_sample_ids),
                ("excluded-audio-sha256", record.audio_sha256, excluded_audio),
                ("excluded-reference-sha256", record.reference_digest, excluded_reference),
            ):
                if value in excluded:
                    findings.append(
                        LeakageFinding(
                            kind=kind,
                            value=value,
                            splits=("test",),
                            sample_ids=(record.sample_id,),
                        )
                    )
        unique = {
            (finding.kind, finding.value, finding.splits, finding.sample_ids): finding
            for finding in findings
        }
        return tuple(
            sorted(
                unique.values(),
                key=lambda finding: (finding.kind, finding.value, finding.sample_ids),
            )
        )

    def _cyclic_sample_ids(self) -> tuple[str, ...]:
        records = {record.sample_id: record for record in self.records}
        state: dict[str, int] = {}
        cyclic: set[str] = set()
        for start in sorted(records):
            if state.get(start, 0):
                continue
            path: list[str] = []
            stack: list[tuple[str, Any]] = [(start, iter(records[start].parent_sample_ids))]
            while stack:
                node, parents = stack[-1]
                if state.get(node, 0) == 0:
                    state[node] = 1
                    path.append(node)
                try:
                    parent = next(parents)
                except StopIteration:
                    stack.pop()
                    state[node] = 2
                    path.pop()
                    continue
                if parent not in records:
                    continue
                parent_state = state.get(parent, 0)
                if parent_state == 0:
                    stack.append((parent, iter(records[parent].parent_sample_ids)))
                elif parent_state == 1:
                    cyclic.update(path[path.index(parent) :])
        return tuple(sorted(cyclic))

    def integrity_report(
        self, *, reference_near_duplicate: bool = True
    ) -> dict[str, Any]:
        sample_ids = {record.sample_id for record in self.records}
        missing_parents = sorted(
            (
                {"sampleId": record.sample_id, "parentSampleId": parent_id}
                for record in self.records
                for parent_id in record.parent_sample_ids
                if parent_id not in sample_ids
            ),
            key=lambda row: (row["sampleId"], row["parentSampleId"]),
        )
        missing_pcm = sorted(
            record.sample_id for record in self.records if record.pcm_sha256 is None
        )
        unknown_speakers = sorted(
            record.sample_id for record in self.records if record.speaker_id is None
        )
        unknown_rights = sorted(
            record.sample_id for record in self.records if record.rights_asset_id is None
        )
        unknown_source_revisions = sorted(
            record.sample_id
            for record in self.records
            if record.source_dataset_revision is None
        )
        revision_mismatches = sorted(
            record.sample_id
            for record in self.records
            if record.source_dataset_revision is not None
            and record.source_dataset_revision != self.dataset_revision
        )
        leakage = self.leakage_findings(
            reference_near_duplicate=reference_near_duplicate
        )
        speaker_leakage = any(finding.kind == "speaker-id" for finding in leakage)
        test_isolation = not any("test" in finding.splits for finding in leakage)
        cycles = self._cyclic_sample_ids()
        return {
            "manifestDigest": self.digest,
            "splitPolicy": self.split_policy,
            "splitSeed": self.split_seed,
            "counts": {name: len(self.split(cast(SplitName, name))) for name in _SPLIT_NAMES},
            "leakageFindings": [asdict(finding) for finding in leakage],
            "missingParentRelations": missing_parents,
            "cyclicSampleIds": list(cycles),
            "missingPcmSha256SampleIds": missing_pcm,
            "unknownSpeakerSampleIds": unknown_speakers,
            "unknownRightsSampleIds": unknown_rights,
            "unknownSourceDatasetRevisionSampleIds": unknown_source_revisions,
            "datasetRevisionMismatchSampleIds": revision_mismatches,
            "speakerDisjointGuaranteed": not unknown_speakers and not speaker_leakage,
            "testIsolationPassed": test_isolation,
            "lineagePassed": not missing_parents
            and not cycles
            and not missing_pcm
            and not unknown_source_revisions
            and not revision_mismatches
            and not leakage,
            "rightsComplete": not unknown_rights,
        }

    def assert_lineage_valid(
        self,
        *,
        reference_near_duplicate: bool = True,
        require_known_speakers: bool = False,
        require_known_rights: bool = False,
    ) -> None:
        report = self.integrity_report(
            reference_near_duplicate=reference_near_duplicate
        )
        hard_failures = (
            report["leakageFindings"],
            report["missingParentRelations"],
            report["cyclicSampleIds"],
            report["missingPcmSha256SampleIds"],
            report["unknownSourceDatasetRevisionSampleIds"],
            report["datasetRevisionMismatchSampleIds"],
        )
        if any(hard_failures):
            raise ValueError("dataset lineage integrity failed; inspect integrity_report()")
        if require_known_speakers and report["unknownSpeakerSampleIds"]:
            raise ValueError("speaker identity is unknown for one or more samples")
        if require_known_rights and report["unknownRightsSampleIds"]:
            raise PermissionError("rights asset is unknown for one or more samples")

    def public_redacted_payload(self) -> dict[str, Any]:
        records = []
        for record in sorted(self.records, key=lambda row: row.sample_id):
            row: dict[str, Any] = {
                "sampleIdSha256": _identifier_digest(record.sample_id),
                "split": record.split,
                "audioSha256": record.audio_sha256,
                "pcmSha256": record.pcm_sha256,
                "referenceSha256": record.reference_digest,
                "speakerKnown": record.speaker_id is not None,
                "sourceRecordingIdSha256": _identifier_digest(record.source_recording_id),
                "sessionIdSha256": _identifier_digest(record.session_id),
                "referenceLineageIdSha256": _identifier_digest(record.reference_lineage_id),
                "derivationGroupIdSha256": _identifier_digest(record.derivation_group_id),
                "nearDuplicateIdSha256": _identifier_digest(record.near_duplicate_id),
                "rightsAssetId": record.rights_asset_id,
                "sourceDatasetRevision": record.source_dataset_revision,
                "parentSampleIdSha256": [
                    _identifier_digest(parent_id) for parent_id in record.parent_sample_ids
                ],
                "durationSeconds": record.duration_seconds,
                "domain": record.domain,
            }
            records.append(row)
        return {
            "schemaVersion": "semantic-asr-public-redacted-manifest-v1",
            "datasetName": self.dataset_name,
            "datasetRevision": self.dataset_revision,
            "rightsRegistryDigest": self.rights_registry_digest,
            "splitPolicy": self.split_policy,
            "splitSeed": self.split_seed,
            "manifestDigest": self.digest,
            "records": records,
            "redaction": {
                "omitted": ["reference", "speakerId", "metadata"],
                "hashed": [
                    "sampleId",
                    "sourceRecordingId",
                    "sessionId",
                    "referenceLineageId",
                    "derivationGroupId",
                    "nearDuplicateId",
                    "parentSampleIds",
                ],
            },
        }

    def assert_leakage_free(self, *, reference_near_duplicate: bool = True) -> None:
        findings = self.leakage_findings(reference_near_duplicate=reference_near_duplicate)
        if findings:
            summary = "; ".join(
                f"{finding.kind}:{finding.value[:16]}:{','.join(finding.splits)}"
                for finding in findings[:10]
            )
            raise ValueError(f"dataset split leakage detected: {summary}")


def _finite_metric(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool or text")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _identifier(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SampleResult:
    sample_id: str
    system_id: str
    metrics: dict[str, float]
    latency_ms: float
    peak_memory_mb: float | None = None
    accepted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.sample_id, name="sample_id")
        _identifier(self.system_id, name="system_id")
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics are required")
        for name, value in self.metrics.items():
            _identifier(name, name="metric name")
            _finite_metric(value, name=f"metric {name}")
        if _finite_metric(self.latency_ms, name="latency_ms") < 0:
            raise ValueError("latency_ms must be non-negative")
        if self.peak_memory_mb is not None and (
            _finite_metric(self.peak_memory_mb, name="peak_memory_mb") < 0
        ):
            raise ValueError("peak_memory_mb must be non-negative")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")


@dataclass(frozen=True, slots=True)
class BootstrapComparison:
    metric: str
    baseline_system: str
    candidate_system: str
    samples: int
    baseline_mean: float
    candidate_mean: float
    mean_delta: float
    confidence: float
    lower_delta: float
    upper_delta: float
    probability_candidate_better: float
    lower_is_better: bool

    group_count: int | None = None
    resampling_unit: Literal["sample", "group"] = "sample"
    aggregation: Literal["utterance-mean", "corpus-error-rate"] = "utterance-mean"

    def __post_init__(self) -> None:
        for name in ("metric", "baseline_system", "candidate_system"):
            _identifier(getattr(self, name), name=name)
        require_integer(self.samples, name="samples", minimum=1)
        if self.group_count is None:
            if self.resampling_unit != "sample":
                raise ValueError("group resampling requires an explicit group_count")
            object.__setattr__(self, "group_count", self.samples)
        require_integer(self.group_count, name="group_count", minimum=1)
        if self.group_count > self.samples or (
            self.resampling_unit == "sample" and self.group_count != self.samples
        ):
            raise ValueError("group_count does not match resampling unit")
        if self.resampling_unit not in {"sample", "group"}:
            raise ValueError("unknown resampling unit")
        if self.aggregation not in {"utterance-mean", "corpus-error-rate"}:
            raise ValueError("unknown aggregation")
        if not isinstance(self.lower_is_better, bool):
            raise TypeError("lower_is_better must be a boolean")
        for name in (
            "baseline_mean",
            "candidate_mean",
            "mean_delta",
            "confidence",
            "lower_delta",
            "upper_delta",
            "probability_candidate_better",
        ):
            _finite_metric(getattr(self, name), name=name)
        if not 0 < self.confidence < 1 or not 0 <= self.probability_candidate_better <= 1:
            raise ValueError("invalid confidence or bootstrap fraction")
        if self.lower_delta > self.upper_delta:
            raise ValueError("confidence interval bounds are reversed")
        if not math.isclose(
            self.mean_delta,
            self.candidate_mean - self.baseline_mean,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("mean_delta differs from the paired means")


@dataclass(frozen=True, slots=True)
class PairedErrorCounts:
    """Exact edit counts for one fixed pair, not a correctness probability.

    Errors may exceed reference length (insertions). Zero-reference rows retain
    insertion counts, but each resampled group needs a positive total denominator.
    Unknown speaker/session independence must not be invented through group IDs.
    """

    sample_id: str
    group_id: str
    reference_units: int
    baseline_errors: int
    candidate_errors: int

    def __post_init__(self) -> None:
        _identifier(self.sample_id, name="sample_id")
        _identifier(self.group_id, name="group_id")
        for name in ("reference_units", "baseline_errors", "candidate_errors"):
            require_integer(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class SliceSummary:
    system_id: str
    metric: str
    slice_name: str
    slice_value: str
    samples: int
    mean: float


def _paired_resampling(
    components: Sequence[tuple[str, str, float, float, int]],
    *,
    baseline_system: str,
    candidate_system: str,
    metric: str,
    iterations: int,
    confidence: float,
    seed: int,
    lower_is_better: bool,
    resampling_unit: Literal["sample", "group"],
    aggregation: Literal["utterance-mean", "corpus-error-rate"],
    expected_sample_ids: Sequence[str] | None,
) -> BootstrapComparison:
    """Resample paired group totals; never rebuild per-sample lists in each draw."""
    require_integer(iterations, name="iterations", minimum=100)
    require_integer(seed, name="seed")
    if not 0 < _finite_metric(confidence, name="confidence") < 1:
        raise ValueError("confidence must be in (0, 1)")
    if not isinstance(lower_is_better, bool):
        raise TypeError("lower_is_better must be a boolean")
    for name, value in (
        ("baseline_system", baseline_system),
        ("candidate_system", candidate_system),
        ("metric", metric),
    ):
        _identifier(value, name=name)
    if not components:
        raise ValueError("paired evaluation requires observations")
    grouped: dict[str, list[tuple[float, float, int]]] = {}
    seen = set()
    for sample_id, group_id, left, right, denominator in sorted(components):
        if sample_id in seen:
            raise ValueError("duplicate sample ID in paired evaluation")
        seen.add(sample_id)
        grouped.setdefault(group_id, []).append((left, right, denominator))
    if expected_sample_ids is not None:
        if isinstance(expected_sample_ids, (str, bytes)):
            raise TypeError("expected_sample_ids must be a sequence of identifiers")
        expected = tuple(expected_sample_ids)
        for identifier in expected:
            _identifier(identifier, name="expected sample ID")
        if len(set(expected)) != len(expected) or set(expected) != seen:
            raise ValueError("expected cohort differs from supplied paired samples")
    totals = [
        (math.fsum(x[0] for x in rows), math.fsum(x[1] for x in rows), sum(x[2] for x in rows))
        for _, rows in sorted(grouped.items())
    ]
    if any(denominator <= 0 for _, _, denominator in totals):
        raise ValueError(
            "each resampled group needs a positive reference denominator; "
            "report silence-only groups separately, do not discard them"
        )

    def means(rows):
        denominator = sum(x[2] for x in rows)
        left = _finite_metric(math.fsum(x[0] for x in rows) / denominator, name="baseline mean")
        right = _finite_metric(math.fsum(x[1] for x in rows) / denominator, name="candidate mean")
        return left, right

    baseline_mean, candidate_mean = means(totals)
    rng = random.Random(seed)
    deltas, better = [], 0
    for _ in range(iterations):
        # Both systems and the denominator use exactly the same group draws.
        sampled = [totals[rng.randrange(len(totals))] for _ in totals]
        left, right = means(sampled)
        delta = _finite_metric(right - left, name="bootstrap delta")
        deltas.append(delta)
        better += int(delta < 0 if lower_is_better else delta > 0)
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(len(deltas) - 1, int(math.floor(tail * len(deltas)))))
    upper_index = max(0, min(len(deltas) - 1, int(math.ceil((1.0 - tail) * len(deltas))) - 1))
    return BootstrapComparison(
        metric=metric,
        baseline_system=baseline_system,
        candidate_system=candidate_system,
        samples=len(components),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=candidate_mean - baseline_mean,
        confidence=confidence,
        lower_delta=deltas[lower_index],
        upper_delta=deltas[upper_index],
        probability_candidate_better=better / iterations,
        lower_is_better=lower_is_better,
        group_count=len(totals),
        resampling_unit=resampling_unit,
        aggregation=aggregation,
    )


def paired_bootstrap_comparison(
    results: Sequence[SampleResult],
    *,
    baseline_system: str,
    candidate_system: str,
    metric: str,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    lower_is_better: bool = True,
    group_ids: Mapping[str, str] | None = None,
    expected_sample_ids: Sequence[str] | None = None,
) -> BootstrapComparison:
    """Compare a COMPLETE paired cohort using utterance-mean metrics.

    Duplicates, missing counterparts and missing metrics are errors, not exclusions.
    ``accepted=False`` rows remain included: selective evaluation requires its own
    preregistered cohort and coverage report. Optional group IDs resample whole
    speakers/sessions; the estimand still weights every utterance equally.
    A frozen expected_sample_ids list additionally detects samples missing from
    BOTH systems. Without it only mutual cohort equality is checked.
    The historical probability_candidate_better field is a bootstrap fraction,
    not a calibrated posterior probability that the candidate is better.
    """
    by_system: dict[str, dict[str, SampleResult]] = {}
    for result in results:
        if result.system_id not in {baseline_system, candidate_system}:
            continue
        result.__post_init__()  # metrics is a legacy mutable mapping; revalidate at use.
        samples = by_system.setdefault(result.system_id, {})
        if result.sample_id in samples:
            raise ValueError("duplicate (system_id, sample_id) in paired evaluation")
        samples[result.sample_id] = result
    if baseline_system not in by_system or candidate_system not in by_system:
        raise ValueError("both systems must be present")
    cohort = set(by_system[baseline_system])
    if cohort != set(by_system[candidate_system]):
        raise ValueError("paired cohort mismatch; missing system outputs cannot be discarded")
    if group_ids is not None and not isinstance(group_ids, Mapping):
        raise TypeError("group_ids must be an ID-to-group mapping")
    if group_ids is not None and set(group_ids) != cohort:
        raise ValueError("group IDs must cover exactly the paired cohort")
    components = []
    for sample_id in sorted(cohort):
        baseline, candidate = (
            by_system[baseline_system][sample_id],
            by_system[candidate_system][sample_id],
        )
        if metric not in baseline.metrics or metric not in candidate.metrics:
            raise ValueError("requested metric is missing on a paired sample")
        group_id = sample_id if group_ids is None else group_ids[sample_id]
        _identifier(group_id, name="group_id")
        components.append(
            (
                sample_id,
                group_id,
                float(baseline.metrics[metric]),
                float(candidate.metrics[metric]),
                1,
            )
        )
    return _paired_resampling(
        components,
        baseline_system=baseline_system,
        candidate_system=candidate_system,
        metric=metric,
        iterations=iterations,
        confidence=confidence,
        seed=seed,
        lower_is_better=lower_is_better,
        resampling_unit="sample" if group_ids is None else "group",
        aggregation="utterance-mean",
        expected_sample_ids=expected_sample_ids,
    )


def paired_error_rate_comparison(
    rows: Sequence[PairedErrorCounts],
    *,
    baseline_system: str,
    candidate_system: str,
    metric: str,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    expected_sample_ids: Sequence[str] | None = None,
) -> BootstrapComparison:
    """Corpus error-rate difference from integer counts, with paired group resampling.

    Divide the resampled total errors by the resampled reference length, NOT by
    utterance or group count. Phone/mora labels from G2P must be reported as proxy
    metrics by the caller. IDs are bookkeeping, not proof of independent speakers.
    """
    components = []
    for row in rows:
        row.__post_init__()
        components.append(
            (
                row.sample_id,
                row.group_id,
                row.baseline_errors,
                row.candidate_errors,
                row.reference_units,
            )
        )
    return _paired_resampling(
        components,
        baseline_system=baseline_system,
        candidate_system=candidate_system,
        metric=metric,
        iterations=iterations,
        confidence=confidence,
        seed=seed,
        lower_is_better=True,
        resampling_unit="group",
        aggregation="corpus-error-rate",
        expected_sample_ids=expected_sample_ids,
    )


def summarize_slices(
    results: Sequence[SampleResult],
    records: Mapping[str, UtteranceRecord],
    *,
    metric: str,
    slicers: Mapping[str, Callable[[UtteranceRecord], str]],
) -> tuple[SliceSummary, ...]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for result in results:
        record = records.get(result.sample_id)
        if record is None or metric not in result.metrics:
            continue
        for slice_name, slicer in slicers.items():
            slice_value = str(slicer(record))
            grouped.setdefault((result.system_id, slice_name, slice_value), []).append(
                result.metrics[metric]
            )
    return tuple(
        SliceSummary(
            system_id=system_id,
            metric=metric,
            slice_name=slice_name,
            slice_value=slice_value,
            samples=len(values),
            mean=fmean(values),
        )
        for (system_id, slice_name, slice_value), values in sorted(grouped.items())
    )


@dataclass(frozen=True, slots=True)
class QualityCostPoint:
    system_id: str
    quality_loss: float
    cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ValueError("system_id is required")
        if any(not math.isfinite(value) or value < 0 for value in (self.quality_loss, self.cost)):
            raise ValueError("quality_loss and cost must be finite and non-negative")


def quality_cost_frontier(
    points: Iterable[QualityCostPoint],
) -> tuple[QualityCostPoint, ...]:
    rows = tuple(points)
    frontier = [
        candidate
        for candidate in rows
        if not any(
            other.system_id != candidate.system_id
            and other.quality_loss <= candidate.quality_loss
            and other.cost <= candidate.cost
            and (other.quality_loss < candidate.quality_loss or other.cost < candidate.cost)
            for other in rows
        )
    ]
    return tuple(sorted(frontier, key=lambda row: (row.cost, row.quality_loss, row.system_id)))


@dataclass(frozen=True, slots=True)
class ClaimGate:
    passed: bool
    reasons: tuple[str, ...]
    comparison: BootstrapComparison
    maximum_critical_regression: float
    maximum_latency_ratio: float


def evaluate_claim_gate(
    comparison: BootstrapComparison,
    *,
    critical_metric_delta: float,
    latency_ratio: float,
    maximum_critical_regression: float = 0.0,
    maximum_latency_ratio: float = 2.0,
    minimum_better_probability: float = 0.95,
) -> ClaimGate:
    comparison.__post_init__()
    for name, value in (
        ("critical_metric_delta", critical_metric_delta),
        ("latency_ratio", latency_ratio),
        ("maximum_critical_regression", maximum_critical_regression),
        ("maximum_latency_ratio", maximum_latency_ratio),
        ("minimum_better_probability", minimum_better_probability),
    ):
        _finite_metric(value, name=name)
    if latency_ratio < 0 or maximum_critical_regression < 0 or maximum_latency_ratio <= 0:
        raise ValueError("invalid critical-regression or latency bounds")
    if not 0 <= minimum_better_probability <= 1:
        raise ValueError("minimum_better_probability must be in [0, 1]")
    reasons: list[str] = []
    if comparison.group_count < 2:
        reasons.append("insufficient-independent-resampling-units")
    improvement = (
        comparison.upper_delta < 0 if comparison.lower_is_better else comparison.lower_delta > 0
    )
    if not improvement:
        reasons.append("paired-confidence-interval-does-not-show-improvement")
    if comparison.probability_candidate_better < minimum_better_probability:
        reasons.append("bootstrap-better-probability-below-threshold")
    if critical_metric_delta > maximum_critical_regression:
        reasons.append("critical-metric-regression")
    if latency_ratio > maximum_latency_ratio:
        reasons.append("latency-ratio-exceeds-threshold")
    return ClaimGate(
        passed=not reasons,
        reasons=tuple(reasons),
        comparison=comparison,
        maximum_critical_regression=maximum_critical_regression,
        maximum_latency_ratio=maximum_latency_ratio,
    )
