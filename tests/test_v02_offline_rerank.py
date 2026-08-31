from __future__ import annotations

import pytest

from semantic_asr.benchmark import BenchmarkUtterance, run_benchmark
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.offline_rerank import (
    build_calibration_samples,
    rerank_record,
)
from semantic_asr.ranker_calibration import RankerCalibrationProfile
from semantic_asr.rerankers import StaticCandidateRanker


def _record(split: str) -> BenchmarkUtterance:
    return BenchmarkUtterance(
        sample_id=f"sample-{split}",
        group_id=f"speaker-{split}",
        source_id=f"source-{split}",
        split=split,
        reference="料金は3000円です",
        candidates=(
            CandidateEvidence(
                "raw-top",
                "料金は30000円です",
                acoustic=0.60,
                mora=0.60,
                rank=1,
                hypothesis_count=2,
            ),
            CandidateEvidence(
                "correct",
                "料金は3000円です",
                acoustic=0.59,
                mora=0.59,
                rank=2,
                hypothesis_count=2,
            ),
        ),
    )


def _ranker() -> StaticCandidateRanker:
    return StaticCandidateRanker({"raw-top": -2.0, "correct": 3.0})


def test_calibration_sample_builder_accepts_only_calibration_split() -> None:
    samples = build_calibration_samples([_record("calibration")], _ranker())
    by_id = {sample.sample_id: sample for sample in samples}
    assert by_id["sample-calibration:correct"].correct
    assert not by_id["sample-calibration:raw-top"].correct
    with pytest.raises(ValueError, match="calibration split only"):
        build_calibration_samples([_record("test")], _ranker())


def test_rerank_preserves_raw_asr_rank_and_injects_only_calibrated_probability() -> None:
    profile = RankerCalibrationProfile(
        name="fixture",
        source_ranker="static",
        slope=1.0,
        intercept=0.0,
        sample_count=20,
        group_count=4,
        calibration_manifest_sha256="a" * 64,
    )
    reranked = rerank_record(_record("test"), _ranker(), calibration=profile)
    by_id = {candidate.candidate_id: candidate for candidate in reranked.candidates}
    assert by_id["raw-top"].rank == 1
    assert by_id["correct"].rank == 2
    assert by_id["correct"].metadata["offlineRerankerRank"] == 1
    assert by_id["raw-top"].metadata["offlineRerankerRank"] == 2
    assert by_id["correct"].lexical is not None
    assert by_id["correct"].metadata["offlineRerankerEvidenceInjected"] is True
    report = run_benchmark([reranked], ks=(1, 2), bootstrap_iterations=20)
    assert report.baseline_cer > 0
    assert report.oracle_cer_at_k[2] == 0


def test_uncalibrated_offline_reranker_does_not_enter_fusion_stream() -> None:
    reranked = rerank_record(_record("test"), _ranker(), calibration=None)
    by_id = {candidate.candidate_id: candidate for candidate in reranked.candidates}
    assert by_id["correct"].lexical is None
    assert by_id["correct"].metadata["offlineRerankerEvidenceInjected"] is False
