from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from .cascade import CascadeConfig, run_candidate_cascade
from .contracts import CandidateEvidence
from .evaluation import cer
from .mbr import critical_units

SplitName = Literal["train", "calibration", "test"]


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
    baseline_cer: float
    cascade_cer: float
    mbr_cer: float
    oracle_cer: dict[int, float]
    rank_regret: float
    adaptive_k: int
    requires_additional_evidence: bool


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    iterations: int
    unit: str = "group"


@dataclass(frozen=True, slots=True)
class BenchmarkSlice:
    count: int
    baseline_cer: float
    cascade_cer: float
    mbr_cer: float
    rank_regret: float
    mean_adaptive_k: float


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    sample_count: int
    group_count: int
    baseline_cer: float
    cascade_cer: float
    mbr_cer: float
    oracle_cer_at_k: dict[int, float]
    rank_regret: float
    mean_adaptive_k: float
    additional_evidence_rate: float
    cascade_improvement: BootstrapInterval
    slices: dict[str, BenchmarkSlice]
    rows: tuple[BenchmarkRow, ...]
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
    }
    for record in records:
        if record.sample_id in seen_samples:
            raise ValueError(f"duplicate sample ID: {record.sample_id}")
        seen_samples.add(record.sample_id)
        indices["group"][record.group_id].add(record.split)
        indices["source"][record.source_id].add(record.split)
        if record.near_duplicate_id:
            indices["near-duplicate"][record.near_duplicate_id].add(record.split)
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


def _cer(reference: str, hypothesis: str) -> float:
    value = cer(reference, hypothesis)
    if value is None:
        raise ValueError("benchmark CER is undefined for an empty reference")
    return float(value)


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
    oracle: dict[int, float] = {}
    previous = math.inf
    for raw_k in sorted(set(int(value) for value in ks)):
        if raw_k < 1:
            raise ValueError("oracle K values must be positive")
        subset = ordered[: min(raw_k, len(ordered))]
        value = min(_cer(record.reference, candidate.text) for candidate in subset)
        value = min(previous, value)
        oracle[raw_k] = value
        previous = value
    maximum_oracle = oracle[max(oracle)]
    cascade_value = _cer(record.reference, cascade_candidate.text)
    return BenchmarkRow(
        sample_id=record.sample_id,
        group_id=record.group_id,
        domain=record.domain,
        critical=bool(critical_units(record.reference)),
        baseline_candidate_id=baseline.candidate_id,
        cascade_candidate_id=cascade_candidate.candidate_id,
        mbr_candidate_id=mbr_candidate.candidate_id,
        baseline_cer=_cer(record.reference, baseline.text),
        cascade_cer=cascade_value,
        mbr_cer=_cer(record.reference, mbr_candidate.text),
        oracle_cer=oracle,
        rank_regret=max(0.0, cascade_value - maximum_oracle),
        adaptive_k=decision.adaptive_k.k,
        requires_additional_evidence=decision.requires_additional_evidence,
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


def paired_group_bootstrap(
    rows: Sequence[BenchmarkRow],
    *,
    left: Callable[[BenchmarkRow], float],
    right: Callable[[BenchmarkRow], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 17,
) -> BootstrapInterval:
    if not rows or iterations < 1 or not 0 < confidence < 1:
        raise ValueError("invalid paired bootstrap configuration")
    by_group: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        by_group[row.group_id].append(row)
    group_ids = sorted(by_group)
    observed = fmean(left(row) - right(row) for row in rows)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_rows: list[BenchmarkRow] = []
        for _group in group_ids:
            sampled_id = rng.choice(group_ids)
            sampled_rows.extend(by_group[sampled_id])
        samples.append(fmean(left(row) - right(row) for row in sampled_rows))
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=observed,
        lower=_percentile(samples, alpha),
        upper=_percentile(samples, 1.0 - alpha),
        confidence=confidence,
        iterations=iterations,
    )


def _slice(rows: Sequence[BenchmarkRow]) -> BenchmarkSlice:
    if not rows:
        raise ValueError("benchmark slice must not be empty")
    return BenchmarkSlice(
        count=len(rows),
        baseline_cer=fmean(row.baseline_cer for row in rows),
        cascade_cer=fmean(row.cascade_cer for row in rows),
        mbr_cer=fmean(row.mbr_cer for row in rows),
        rank_regret=fmean(row.rank_regret for row in rows),
        mean_adaptive_k=fmean(row.adaptive_k for row in rows),
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
    rows = tuple(evaluate_utterance(record, ks=ks) for record in records)
    oracle = {
        int(k): fmean(row.oracle_cer[int(k)] for row in rows)
        for k in sorted(set(int(value) for value in ks))
    }
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
    return BenchmarkReport(
        sample_count=len(rows),
        group_count=len({row.group_id for row in rows}),
        baseline_cer=fmean(row.baseline_cer for row in rows),
        cascade_cer=fmean(row.cascade_cer for row in rows),
        mbr_cer=fmean(row.mbr_cer for row in rows),
        oracle_cer_at_k=oracle,
        rank_regret=fmean(row.rank_regret for row in rows),
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
