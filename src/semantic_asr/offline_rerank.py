from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .benchmark import BenchmarkUtterance, load_benchmark_jsonl
from .calibration import CalibrationProfile
from .contracts import CandidateEvidence
from .mbr import semantic_loss
from .ranker_calibration import RankerCalibrationProfile, RankerCalibrationSample
from .rerankers import CandidateRanker


class ScoreCalibration(Protocol):
    @property
    def digest(self) -> str: ...

    def transform(self, value: float | None) -> float | None: ...


def _ranker_scores(
    record: BenchmarkUtterance,
    ranker: CandidateRanker,
) -> dict[str, float]:
    scores = dict(ranker.score(record.candidates, context=""))
    identifiers = {candidate.candidate_id for candidate in record.candidates}
    if set(scores) != identifiers:
        raise ValueError("ranker must score every candidate ID exactly once")
    output = {candidate_id: float(value) for candidate_id, value in scores.items()}
    if any(not math.isfinite(value) for value in output.values()):
        raise ValueError("ranker returned a non-finite score")
    return output


def build_calibration_samples(
    records: Sequence[BenchmarkUtterance],
    ranker: CandidateRanker,
    *,
    oracle_tolerance: float = 1e-12,
) -> list[RankerCalibrationSample]:
    if not records:
        raise ValueError("calibration records must not be empty")
    if any(record.split != "calibration" for record in records):
        raise ValueError("ranker calibration samples require calibration split only")
    output: list[RankerCalibrationSample] = []
    for record in records:
        reference = CandidateEvidence("__reference__", record.reference)
        losses = {
            candidate.candidate_id: semantic_loss(candidate, reference)[0]
            for candidate in record.candidates
        }
        oracle_loss = min(losses.values())
        scores = _ranker_scores(record, ranker)
        for candidate in record.candidates:
            output.append(
                RankerCalibrationSample(
                    sample_id=f"{record.sample_id}:{candidate.candidate_id}",
                    group_id=record.group_id,
                    score=scores[candidate.candidate_id],
                    correct=losses[candidate.candidate_id] <= oracle_loss + oracle_tolerance,
                )
            )
    return output


def rerank_record(
    record: BenchmarkUtterance,
    ranker: CandidateRanker,
    *,
    calibration: ScoreCalibration | CalibrationProfile | RankerCalibrationProfile | None = None,
    lexical_blend: float = 0.65,
) -> BenchmarkUtterance:
    if not 0 <= lexical_blend <= 1:
        raise ValueError("lexical_blend must be in [0, 1]")
    scores = _ranker_scores(record, ranker)
    ordered = sorted(
        record.candidates,
        key=lambda candidate: (-scores[candidate.candidate_id], candidate.candidate_id),
    )
    reranked: list[CandidateEvidence] = []
    for rank, candidate in enumerate(ordered, 1):
        raw_score = scores[candidate.candidate_id]
        calibrated = calibration.transform(raw_score) if calibration is not None else None
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "originalRank": candidate.rank,
                "offlineReranker": ranker.name,
                "offlineRerankerRawScore": raw_score,
                "offlineRerankerRank": rank,
                "offlineRerankerCalibrationDigest": (
                    calibration.digest if calibration is not None else None
                ),
                "offlineRerankerCalibratedProbability": calibrated,
                "offlineRerankerEvidenceInjected": calibrated is not None,
            }
        )
        lexical = candidate.lexical
        if calibrated is not None:
            lexical = (
                float(calibrated)
                if lexical is None
                else (1.0 - lexical_blend) * float(lexical) + lexical_blend * float(calibrated)
            )
        reranked.append(
            replace(
                candidate,
                lexical=lexical,
                metadata=metadata,
            )
        )
    return BenchmarkUtterance(
        sample_id=record.sample_id,
        group_id=record.group_id,
        source_id=record.source_id,
        split=record.split,
        reference=record.reference,
        candidates=tuple(reranked),
        domain=record.domain,
        near_duplicate_id=record.near_duplicate_id,
        annotated_reference=record.annotated_reference,
    )


def rerank_records(
    records: Sequence[BenchmarkUtterance],
    ranker: CandidateRanker,
    *,
    calibration: ScoreCalibration | CalibrationProfile | RankerCalibrationProfile | None = None,
    lexical_blend: float = 0.65,
) -> list[BenchmarkUtterance]:
    return [
        rerank_record(
            record,
            ranker,
            calibration=calibration,
            lexical_blend=lexical_blend,
        )
        for record in records
    ]


def write_calibration_samples(samples: Sequence[RankerCalibrationSample], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            json.dumps(
                {
                    "sampleId": sample.sample_id,
                    "groupId": sample.group_id,
                    "score": sample.score,
                    "correct": sample.correct,
                    "split": sample.split,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for sample in samples
        )
        + ("\n" if samples else ""),
        encoding="utf-8",
    )


def write_reranked_benchmark(records: Sequence[BenchmarkUtterance], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = {
            "sampleId": record.sample_id,
            "groupId": record.group_id,
            "sourceId": record.source_id,
            "split": record.split,
            "reference": record.reference,
            "domain": record.domain,
            "nearDuplicateId": record.near_duplicate_id,
            "candidates": [candidate.as_dict() for candidate in record.candidates],
        }
        if record.annotated_reference is not None:
            row["annotatedReference"] = record.annotated_reference
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def load_records(path: str | Path) -> list[BenchmarkUtterance]:
    return load_benchmark_jsonl(path)
