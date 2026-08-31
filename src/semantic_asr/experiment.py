from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Literal

SplitName = Literal["train", "calibration", "test"]


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

    def __post_init__(self) -> None:
        if not self.sample_id or not self.audio_sha256 or not self.reference:
            raise ValueError("sample_id, audio_sha256 and reference are required")
        if len(self.audio_sha256) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.audio_sha256
        ):
            raise ValueError("audio_sha256 must be a hexadecimal SHA-256 digest")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be finite and positive")

    @property
    def reference_digest(self) -> str:
        return hashlib.sha256(self.reference.encode("utf-8")).hexdigest()


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

    def __post_init__(self) -> None:
        if not self.records or not self.dataset_name or not self.dataset_revision:
            raise ValueError("records, dataset_name and dataset_revision are required")
        if len({record.sample_id for record in self.records}) != len(self.records):
            raise ValueError("sample IDs must be globally unique")

    @property
    def digest(self) -> str:
        payload = {
            "datasetName": self.dataset_name,
            "datasetRevision": self.dataset_revision,
            "rightsRegistryDigest": self.rights_registry_digest,
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
        return tuple(record for record in self.records if record.split == name)

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
        collect("speaker-id", ((record.speaker_id, record) for record in self.records))
        collect(
            "source-recording-id",
            ((record.source_recording_id, record) for record in self.records),
        )
        if reference_near_duplicate:
            collect(
                "reference-digest",
                ((record.reference_digest, record) for record in self.records),
            )
        return tuple(
            sorted(
                findings,
                key=lambda finding: (finding.kind, finding.value, finding.sample_ids),
            )
        )

    def assert_leakage_free(self, *, reference_near_duplicate: bool = True) -> None:
        findings = self.leakage_findings(reference_near_duplicate=reference_near_duplicate)
        if findings:
            summary = "; ".join(
                f"{finding.kind}:{finding.value[:16]}:{','.join(finding.splits)}"
                for finding in findings[:10]
            )
            raise ValueError(f"dataset split leakage detected: {summary}")


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
        if not self.sample_id or not self.system_id or not self.metrics:
            raise ValueError("sample_id, system_id and metrics are required")
        for name, value in self.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name} must be finite")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        if self.peak_memory_mb is not None and (
            not math.isfinite(self.peak_memory_mb) or self.peak_memory_mb < 0
        ):
            raise ValueError("peak_memory_mb must be finite and non-negative")


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


@dataclass(frozen=True, slots=True)
class SliceSummary:
    system_id: str
    metric: str
    slice_name: str
    slice_value: str
    samples: int
    mean: float


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
) -> BootstrapComparison:
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    by_system: dict[str, dict[str, SampleResult]] = {}
    for result in results:
        by_system.setdefault(result.system_id, {})[result.sample_id] = result
    if baseline_system not in by_system or candidate_system not in by_system:
        raise ValueError("both systems must be present")
    shared = sorted(set(by_system[baseline_system]) & set(by_system[candidate_system]))
    if not shared:
        raise ValueError("systems have no shared samples")
    pairs = []
    for sample_id in shared:
        baseline = by_system[baseline_system][sample_id]
        candidate = by_system[candidate_system][sample_id]
        if metric not in baseline.metrics or metric not in candidate.metrics:
            continue
        pairs.append((baseline.metrics[metric], candidate.metrics[metric]))
    if not pairs:
        raise ValueError(f"metric {metric} is missing on shared samples")

    rng = random.Random(seed)
    deltas: list[float] = []
    better = 0
    for _ in range(iterations):
        sampled = [pairs[rng.randrange(len(pairs))] for _row in pairs]
        delta = fmean(candidate - baseline for baseline, candidate in sampled)
        deltas.append(delta)
        better += int(delta < 0 if lower_is_better else delta > 0)
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(len(deltas) - 1, int(math.floor(tail * len(deltas)))))
    upper_index = max(
        0,
        min(len(deltas) - 1, int(math.ceil((1.0 - tail) * len(deltas))) - 1),
    )
    baseline_mean = fmean(pair[0] for pair in pairs)
    candidate_mean = fmean(pair[1] for pair in pairs)
    return BootstrapComparison(
        metric=metric,
        baseline_system=baseline_system,
        candidate_system=candidate_system,
        samples=len(pairs),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=candidate_mean - baseline_mean,
        confidence=confidence,
        lower_delta=deltas[lower_index],
        upper_delta=deltas[upper_index],
        probability_candidate_better=better / iterations,
        lower_is_better=lower_is_better,
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
    reasons: list[str] = []
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
