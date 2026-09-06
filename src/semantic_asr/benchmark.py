from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from .audio import require_integer
from .cascade import CascadeConfig, run_candidate_cascade
from .contracts import CandidateEvidence
from .evaluation import (
    ContiguousAlignmentDiagnostic,
    best_contiguous_alignment,
    exact_cer,
    lenient_cer,
    normalize_characters,
    normalize_characters_lenient,
    reference_annotation_counts,
    spoken_reference_surface,
)
from .experiment import _finite_metric
from .mbr import critical_units

SplitName = Literal["train", "calibration", "test"]
MetricValue = float | None


@dataclass(frozen=True, slots=True)
class BenchmarkUtterance:
    sample_id: str
    group_id: str
    source_id: str
    split: SplitName
    reference: str
    candidates: tuple[CandidateEvidence, ...]
    domain: str = "unknown"
    near_duplicate_id: str | None = None
    annotated_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id or not self.source_id:
            raise ValueError("sample, group, and source IDs are required")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("unknown benchmark split")
        if not self.reference:
            raise ValueError("benchmark reference must not be empty")
        if not self.candidates:
            raise ValueError("benchmark utterance requires at least one candidate")
        identifiers = [candidate.candidate_id for candidate in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate IDs must be unique within an utterance")


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    sample_id: str
    group_id: str
    domain: str
    critical: bool
    baseline_candidate_id: str
    cascade_candidate_id: str
    mbr_candidate_id: str
    baseline_cer: MetricValue
    cascade_cer: MetricValue
    mbr_cer: MetricValue
    oracle_cer: dict[int, MetricValue]
    rank_regret: MetricValue
    adaptive_k: int
    requires_additional_evidence: bool
    reference_characters: int = 0
    reference_characters_lenient: int = 0
    baseline_lenient_cer: MetricValue = 0.0
    cascade_lenient_cer: MetricValue = 0.0
    mbr_lenient_cer: MetricValue = 0.0
    boundary_diagnostics: dict[str, ContiguousAlignmentDiagnostic] | None = None
    annotated_reference: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    iterations: int
    unit: str = "group"
    eligible_samples: int = 0
    group_count: int = 0
    excluded_samples: int = 0


@dataclass(frozen=True, slots=True)
class BenchmarkSlice:
    count: int
    baseline_cer: MetricValue
    cascade_cer: MetricValue
    mbr_cer: MetricValue
    rank_regret: MetricValue
    mean_adaptive_k: float
    corpus_cer: dict[str, MetricValue] | None = None
    lenient_corpus_cer: dict[str, MetricValue] | None = None
    boundary_diagnostics: dict[str, BoundaryDiagnosticAggregate] | None = None


@dataclass(frozen=True, slots=True)
class BoundaryDiagnosticAggregate:
    aligned_corpus_cer: MetricValue
    alignment_edit_reduction: int
    overrun_rows: int
    prefix_overrun_rows: int
    prefix_overrun_characters: int
    suffix_overrun_rows: int
    suffix_overrun_characters: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    sample_count: int
    group_count: int
    baseline_cer: MetricValue
    cascade_cer: MetricValue
    mbr_cer: MetricValue
    oracle_cer_at_k: dict[int, MetricValue]
    rank_regret: MetricValue
    mean_adaptive_k: float
    additional_evidence_rate: float
    cascade_improvement: BootstrapInterval | None
    slices: dict[str, BenchmarkSlice]
    rows: tuple[BenchmarkRow, ...]
    corpus_cer: dict[str, MetricValue] | None = None
    lenient_corpus_cer: dict[str, MetricValue] | None = None
    boundary_diagnostics: dict[str, BoundaryDiagnosticAggregate] | None = None
    metric_note: str = (
        "utterance-mean CER keeps punctuation (strict); corpus_cer weights utterances by "
        "reference length; lenient variants strip punctuation and symbols for comparison "
        "with published Japanese ASR numbers. Boundary alignment is diagnostic-only and never "
        "changes candidate selection or the primary strict CER. Rows with undefined annotated "
        "references are excluded from exact CER aggregates."
    )
    claim_boundary: str = (
        "A report is evidence only for the exact immutable manifest, models, "
        "runtime, split policy, and configuration used to produce it."
    )


def verify_split_isolation(records: Sequence[BenchmarkUtterance]) -> None:
    if not records:
        raise ValueError("benchmark records must not be empty")
    seen_samples: set[str] = set()
    indices: dict[str, dict[str, set[str]]] = {
        "group": defaultdict(set),
        "source": defaultdict(set),
        "near-duplicate": defaultdict(set),
        "reference": defaultdict(set),
    }
    for record in records:
        if record.sample_id in seen_samples:
            raise ValueError(f"duplicate sample ID: {record.sample_id}")
        seen_samples.add(record.sample_id)
        indices["group"][record.group_id].add(record.split)
        indices["source"][record.source_id].add(record.split)
        if record.near_duplicate_id:
            indices["near-duplicate"][record.near_duplicate_id].add(record.split)
        reference_digest = hashlib.sha256(
            "".join(normalize_characters(record.reference)).encode("utf-8")
        ).hexdigest()
        indices["reference"][reference_digest].add(record.split)
    for kind, rows in indices.items():
        leaking = {
            identifier: sorted(splits) for identifier, splits in rows.items() if len(splits) > 1
        }
        if leaking:
            first_id = sorted(leaking)[0]
            raise ValueError(f"{kind} leakage across splits: {first_id} -> {leaking[first_id]}")


def _ordered_candidates(record: BenchmarkUtterance) -> list[CandidateEvidence]:
    return sorted(
        record.candidates,
        key=lambda candidate: (
            candidate.rank is None,
            candidate.rank if candidate.rank is not None else 10**9,
            -(
                candidate.avg_logprob
                if candidate.avg_logprob is not None
                else candidate.acoustic
                if candidate.acoustic is not None
                else -math.inf
            ),
            candidate.candidate_id,
        ),
    )


def _cer(
    reference: str,
    hypothesis: str,
    *,
    annotated_reference: str | None = None,
) -> MetricValue:
    return exact_cer(
        reference,
        hypothesis,
        annotated_reference=annotated_reference,
    )


def evaluate_utterance(
    record: BenchmarkUtterance,
    *,
    ks: Sequence[int],
    cascade_config: CascadeConfig | None = None,
) -> BenchmarkRow:
    ordered = _ordered_candidates(record)
    baseline = ordered[0]
    decision = run_candidate_cascade(
        ordered,
        cascade_config=cascade_config or CascadeConfig(selection_policy="fusion"),
    )
    by_id = {ranked.candidate.candidate_id: ranked for ranked in decision.ranked}
    cascade_candidate = by_id[decision.selected_candidate_id].candidate
    mbr_candidate = next(
        candidate
        for candidate in decision.ranked
        if candidate.candidate.candidate_id == decision.mbr.selected_candidate_id
    ).candidate
    metric_reference = (
        spoken_reference_surface(record.annotated_reference)
        if record.annotated_reference is not None
        else record.reference
    )
    baseline_cer = _cer(
        record.reference,
        baseline.text,
        annotated_reference=record.annotated_reference,
    )
    cascade_cer = _cer(
        record.reference,
        cascade_candidate.text,
        annotated_reference=record.annotated_reference,
    )
    mbr_cer = _cer(
        record.reference,
        mbr_candidate.text,
        annotated_reference=record.annotated_reference,
    )
    exact_reference_safe = (
        record.annotated_reference is None
        or reference_annotation_counts(record.annotated_reference).exact_cer_safe
    )
    oracle: dict[int, MetricValue] = {}
    previous: MetricValue = None
    normalized_ks = sorted(set(int(value) for value in ks))
    if not normalized_ks or any(value < 1 for value in normalized_ks):
        raise ValueError("oracle K values must be positive")
    for raw_k in normalized_ks:
        subset = ordered[: min(raw_k, len(ordered))]
        values = [
            value
            for candidate in subset
            if (
                value := _cer(
                    record.reference,
                    candidate.text,
                    annotated_reference=record.annotated_reference,
                )
            )
            is not None
        ]
        value = min(values) if values else previous
        if value is not None and previous is not None:
            value = min(previous, value)
        oracle[raw_k] = value
        previous = value
    maximum_oracle = oracle[max(oracle)]
    rank_regret = (
        None
        if cascade_cer is None or maximum_oracle is None
        else max(0.0, cascade_cer - maximum_oracle)
    )
    lenient_reference = normalize_characters_lenient(metric_reference)

    def _lenient(text: str) -> MetricValue:
        if not lenient_reference:
            return None
        if not exact_reference_safe:
            return None
        return lenient_cer(metric_reference, text)

    boundary_diagnostics = None
    if exact_reference_safe:
        boundary_diagnostics = {
            system: diagnostic
            for system, text in (
                ("baseline", baseline.text),
                ("cascade", cascade_candidate.text),
                ("mbr", mbr_candidate.text),
            )
            if (diagnostic := best_contiguous_alignment(metric_reference, text)) is not None
        }

    return BenchmarkRow(
        sample_id=record.sample_id,
        group_id=record.group_id,
        domain=record.domain,
        critical=bool(critical_units(metric_reference)),
        baseline_candidate_id=baseline.candidate_id,
        cascade_candidate_id=cascade_candidate.candidate_id,
        mbr_candidate_id=mbr_candidate.candidate_id,
        baseline_cer=baseline_cer,
        cascade_cer=cascade_cer,
        mbr_cer=mbr_cer,
        oracle_cer=oracle,
        rank_regret=rank_regret,
        adaptive_k=decision.adaptive_k.k,
        requires_additional_evidence=decision.requires_additional_evidence,
        reference_characters=len(normalize_characters(metric_reference)),
        reference_characters_lenient=len(lenient_reference),
        baseline_lenient_cer=_lenient(baseline.text),
        cascade_lenient_cer=_lenient(cascade_candidate.text),
        mbr_lenient_cer=_lenient(mbr_candidate.text),
        boundary_diagnostics=boundary_diagnostics,
        annotated_reference=record.annotated_reference,
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0.0, fraction * (len(ordered) - 1)))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean_defined(values: Iterable[MetricValue]) -> MetricValue:
    defined = [value for value in values if value is not None]
    return fmean(defined) if defined else None


def paired_group_bootstrap(
    rows: Sequence[BenchmarkRow],
    *,
    left: Callable[[BenchmarkRow], MetricValue],
    right: Callable[[BenchmarkRow], MetricValue],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> BootstrapInterval | None:
    """Paired utterance-mean difference (left minus right), clustered by group.

    Jointly undefined annotated rows remain explicitly excluded; asymmetric
    missingness or non-finite metrics are errors. Evaluate each callback once.
    """
    require_integer(iterations, name="iterations", minimum=1)
    require_integer(seed, name="seed")
    if not rows or not 0 < _finite_metric(confidence, name="confidence") < 1:
        raise ValueError("invalid paired bootstrap configuration")
    by_group: dict[str, list[float]] = defaultdict(list)
    seen: set[str] = set()
    excluded = 0
    for row in rows:
        if row.sample_id in seen:
            raise ValueError("duplicate sample ID in paired group bootstrap")
        seen.add(row.sample_id)
        if not isinstance(row.group_id, str) or not row.group_id.strip():
            raise ValueError("group_id is required")
        left_value, right_value = left(row), right(row)
        if left_value is None and right_value is None:
            excluded += 1
            continue
        if left_value is None or right_value is None:
            raise ValueError("asymmetric missing metric in paired group bootstrap")
        difference = _finite_metric(left_value, name="left") - _finite_metric(
            right_value, name="right"
        )
        by_group[row.group_id].append(_finite_metric(difference, name="difference"))
    if not by_group:
        return None
    # Summarize each group once. Resampling no longer rebuilds complete row lists
    # or repeatedly invokes potentially expensive metric callbacks.
    totals = [(math.fsum(values), len(values)) for _, values in sorted(by_group.items())]
    count = sum(n for _, n in totals)
    observed = math.fsum(value for value, _ in totals) / count
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        drawn = [rng.choice(totals) for _ in totals]
        samples.append(math.fsum(value for value, _ in drawn) / sum(n for _, n in drawn))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=observed,
        lower=_percentile(samples, alpha),
        upper=_percentile(samples, 1.0 - alpha),
        confidence=confidence,
        iterations=iterations,
        eligible_samples=count,
        group_count=len(totals),
        excluded_samples=excluded,
    )


def _corpus_cer(rows: Sequence[BenchmarkRow], *, lenient: bool) -> dict[str, MetricValue]:
    """Length-weighted CER over rows with a defined metric and non-empty reference."""

    length_attribute = "reference_characters_lenient" if lenient else "reference_characters"
    suffix = "_lenient_cer" if lenient else "_cer"
    output: dict[str, MetricValue] = {}
    for system in ("baseline", "cascade", "mbr"):
        values = [
            row
            for row in rows
            if getattr(row, length_attribute) > 0
            if getattr(row, f"{system}{suffix}") is not None
        ]
        denominator = sum(getattr(row, length_attribute) for row in values)
        if not denominator:
            output[system] = None
            continue
        edits = sum(
            getattr(row, f"{system}{suffix}") * getattr(row, length_attribute) for row in values
        )
        output[system] = edits / denominator
    return output


def _boundary_diagnostics(
    rows: Sequence[BenchmarkRow],
) -> dict[str, BoundaryDiagnosticAggregate]:
    output: dict[str, BoundaryDiagnosticAggregate] = {}
    for system in ("baseline", "cascade", "mbr"):
        diagnostics = [
            row.boundary_diagnostics[system]
            for row in rows
            if row.boundary_diagnostics is not None and system in row.boundary_diagnostics
        ]
        total_reference = sum(value.reference_characters for value in diagnostics)
        aligned_edits = sum(value.edits for value in diagnostics)
        strict_edits = round(
            sum(
                getattr(row, f"{system}_cer") * row.reference_characters
                for row in rows
                if getattr(row, f"{system}_cer") is not None and row.reference_characters > 0
            )
        )
        output[system] = BoundaryDiagnosticAggregate(
            aligned_corpus_cer=aligned_edits / total_reference if total_reference else None,
            alignment_edit_reduction=max(0, strict_edits - aligned_edits),
            overrun_rows=sum(
                value.prefix_overrun_characters > 0 or value.suffix_overrun_characters > 0
                for value in diagnostics
            ),
            prefix_overrun_rows=sum(value.prefix_overrun_characters > 0 for value in diagnostics),
            prefix_overrun_characters=sum(value.prefix_overrun_characters for value in diagnostics),
            suffix_overrun_rows=sum(value.suffix_overrun_characters > 0 for value in diagnostics),
            suffix_overrun_characters=sum(value.suffix_overrun_characters for value in diagnostics),
        )
    return output


def _slice(rows: Sequence[BenchmarkRow]) -> BenchmarkSlice:
    if not rows:
        raise ValueError("benchmark slice must not be empty")
    return BenchmarkSlice(
        count=len(rows),
        baseline_cer=_mean_defined(row.baseline_cer for row in rows),
        cascade_cer=_mean_defined(row.cascade_cer for row in rows),
        mbr_cer=_mean_defined(row.mbr_cer for row in rows),
        rank_regret=_mean_defined(row.rank_regret for row in rows),
        mean_adaptive_k=fmean(row.adaptive_k for row in rows),
        corpus_cer=_corpus_cer(rows, lenient=False),
        lenient_corpus_cer=_corpus_cer(rows, lenient=True),
        boundary_diagnostics=_boundary_diagnostics(rows),
    )


def run_benchmark(
    records: Sequence[BenchmarkUtterance],
    *,
    ks: Sequence[int] = (1, 3, 5, 8, 12, 16, 25, 50),
    bootstrap_iterations: int = 2000,
    seed: int = 17,
    require_test_split: bool = True,
) -> BenchmarkReport:
    verify_split_isolation(records)
    if require_test_split and any(record.split != "test" for record in records):
        raise ValueError("final benchmark may consume only the locked test split")
    normalized_ks = sorted(set(int(value) for value in ks))
    if not normalized_ks or any(value < 1 for value in normalized_ks):
        raise ValueError("oracle K values must be positive")
    rows = tuple(evaluate_utterance(record, ks=normalized_ks) for record in records)
    oracle = {int(k): _mean_defined(row.oracle_cer[int(k)] for row in rows) for k in normalized_ks}
    slices: dict[str, BenchmarkSlice] = {"all": _slice(rows)}
    for domain in sorted({row.domain for row in rows}):
        domain_rows = [row for row in rows if row.domain == domain]
        slices[f"domain:{domain}"] = _slice(domain_rows)
    for label, predicate in (
        ("critical", lambda row: row.critical),
        ("non-critical", lambda row: not row.critical),
    ):
        selected = [row for row in rows if predicate(row)]
        if selected:
            slices[label] = _slice(selected)
    for label, lower, upper in (
        ("length:short<0.75", 0.0, 0.75),
        ("length:near[0.75,1.25)", 0.75, 1.25),
        ("length:long>=1.25", 1.25, math.inf),
    ):
        selected = [
            row
            for row in rows
            if row.boundary_diagnostics is not None
            and (diagnostic := row.boundary_diagnostics.get("baseline")) is not None
            and diagnostic.reference_characters > 0
            and lower <= diagnostic.hypothesis_characters / diagnostic.reference_characters < upper
        ]
        if selected:
            slices[label] = _slice(selected)
    return BenchmarkReport(
        sample_count=len(rows),
        group_count=len({row.group_id for row in rows}),
        baseline_cer=_mean_defined(row.baseline_cer for row in rows),
        cascade_cer=_mean_defined(row.cascade_cer for row in rows),
        mbr_cer=_mean_defined(row.mbr_cer for row in rows),
        oracle_cer_at_k=oracle,
        rank_regret=_mean_defined(row.rank_regret for row in rows),
        mean_adaptive_k=fmean(row.adaptive_k for row in rows),
        additional_evidence_rate=fmean(float(row.requires_additional_evidence) for row in rows),
        cascade_improvement=paired_group_bootstrap(
            rows,
            left=lambda row: row.baseline_cer,
            right=lambda row: row.cascade_cer,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        slices=slices,
        rows=rows,
        corpus_cer=_corpus_cer(rows, lenient=False),
        lenient_corpus_cer=_corpus_cer(rows, lenient=True),
        boundary_diagnostics=_boundary_diagnostics(rows),
    )


def _candidate_from_row(row: Mapping[str, Any]) -> CandidateEvidence:
    aliases = {
        "candidateId": "candidate_id",
        "tokenIds": "token_ids",
        "crossModel": "cross_model",
        "moraUnits": "mora_units",
        "hypothesisCount": "hypothesis_count",
        "sequenceScore": "sequence_score",
        "avgLogprob": "avg_logprob",
        "beamConfidence": "beam_confidence",
    }
    return CandidateEvidence.from_dict(
        {aliases.get(str(key), str(key)): value for key, value in row.items()}
    )


def benchmark_utterance_from_row(
    row: Mapping[str, Any], *, line_number: int = 0
) -> BenchmarkUtterance:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"benchmark row {line_number} has no candidates array")
    annotated_reference = row.get("annotatedReference", row.get("annotated_reference"))
    return BenchmarkUtterance(
        sample_id=str(row.get("sampleId") or row.get("sample_id") or line_number),
        group_id=str(row.get("groupId") or row.get("group_id") or ""),
        source_id=str(row.get("sourceId") or row.get("source_id") or ""),
        split=str(row.get("split") or "test"),
        reference=str(row.get("reference") or ""),
        candidates=tuple(_candidate_from_row(dict(value)) for value in candidates),
        domain=str(row.get("domain") or "unknown"),
        near_duplicate_id=(
            str(row.get("nearDuplicateId") or row.get("near_duplicate_id"))
            if row.get("nearDuplicateId") or row.get("near_duplicate_id")
            else None
        ),
        annotated_reference=(None if annotated_reference is None else str(annotated_reference)),
    )


def load_benchmark_jsonl(path: str | Path) -> list[BenchmarkUtterance]:
    output: list[BenchmarkUtterance] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"benchmark row {line_number} must be an object")
        output.append(benchmark_utterance_from_row(payload, line_number=line_number))
    if not output:
        raise ValueError("benchmark dataset is empty")
    return output


def write_benchmark_report(report: BenchmarkReport, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
