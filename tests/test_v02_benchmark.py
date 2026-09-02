from __future__ import annotations

import pytest

from semantic_asr.benchmark import (
    BenchmarkUtterance,
    run_benchmark,
    verify_split_isolation,
)
from semantic_asr.contracts import CandidateEvidence


def _record(
    sample_id: str,
    group_id: str,
    source_id: str,
    reference: str,
    texts: list[str],
    *,
    split: str = "test",
    domain: str = "meeting",
) -> BenchmarkUtterance:
    return BenchmarkUtterance(
        sample_id=sample_id,
        group_id=group_id,
        source_id=source_id,
        split=split,
        reference=reference,
        candidates=tuple(
            CandidateEvidence(
                candidate_id=f"{sample_id}-{index}",
                text=text,
                acoustic=1.0 - index * 0.1,
                mora=1.0 - index * 0.1,
                rank=index + 1,
                hypothesis_count=len(texts),
            )
            for index, text in enumerate(texts)
        ),
        domain=domain,
    )


def test_split_isolation_rejects_speaker_leakage() -> None:
    records = [
        _record("train", "speaker-a", "source-a", "東京です", ["東京です"], split="train"),
        _record("test", "speaker-a", "source-b", "大阪です", ["大阪です"], split="test"),
    ]
    with pytest.raises(ValueError, match="group leakage"):
        verify_split_isolation(records)


def test_split_isolation_rejects_source_leakage() -> None:
    records = [
        _record("train", "speaker-a", "source-a", "東京です", ["東京です"], split="train"),
        _record("test", "speaker-b", "source-a", "大阪です", ["大阪です"], split="test"),
    ]
    with pytest.raises(ValueError, match="source leakage"):
        verify_split_isolation(records)


def test_benchmark_reports_monotonic_oracle_curve_and_group_bootstrap() -> None:
    records = [
        _record(
            "one",
            "speaker-a",
            "source-a",
            "料金は3000円です",
            ["料金は30000円です", "料金は3000円です", "料金は3000円でした"],
        ),
        _record(
            "two",
            "speaker-b",
            "source-b",
            "明日は行きません",
            ["明日は行きます", "明日は行きません", "昨日は行きません"],
        ),
        _record(
            "three",
            "speaker-c",
            "source-c",
            "東京へ行きます",
            ["東京に行きます", "東京へ行きます", "京都へ行きます"],
            domain="travel",
        ),
    ]
    report = run_benchmark(
        records,
        ks=(1, 2, 3),
        bootstrap_iterations=200,
        seed=5,
    )
    assert report.sample_count == 3
    assert report.group_count == 3
    assert report.oracle_cer_at_k[2] <= report.oracle_cer_at_k[1]
    assert report.oracle_cer_at_k[3] <= report.oracle_cer_at_k[2]
    assert report.cascade_improvement.iterations == 200
    assert report.cascade_improvement.lower <= report.cascade_improvement.upper
    assert "critical" in report.slices
    assert "domain:meeting" in report.slices
    assert report.mean_adaptive_k >= 1
    assert set(report.corpus_cer) == {"baseline", "cascade", "mbr"}
    assert report.slices["all"].corpus_cer == report.corpus_cer
    assert 0.0 <= report.lenient_corpus_cer["baseline"] <= report.corpus_cer["baseline"] + 1e-9
    assert set(report.boundary_diagnostics) == {"baseline", "cascade", "mbr"}
    assert report.boundary_diagnostics["baseline"].aligned_corpus_cer >= 0.0


def test_boundary_diagnostics_are_report_only_and_use_fixed_length_slices() -> None:
    records = [
        _record(
            "overrun",
            "speaker-o",
            "source-o",
            "中央区です",
            ["前置き中央区です後", "中央区です"],
        )
    ]
    report = run_benchmark(records, ks=(1, 2), bootstrap_iterations=10, seed=1)
    row = report.rows[0]
    diagnostic = row.boundary_diagnostics["baseline"]
    assert diagnostic.edits == 0
    assert diagnostic.prefix_overrun_characters == 3
    assert diagnostic.suffix_overrun_characters == 1
    assert report.boundary_diagnostics["baseline"].overrun_rows == 1
    assert "length:long>=1.25" in report.slices
    # The diagnostic must not alter the selected candidate or strict primary score.
    assert row.baseline_candidate_id == "overrun-0"
    assert row.baseline_cer > 0.0


def test_lenient_cer_ignores_punctuation_but_strict_does_not() -> None:
    records = [
        _record(
            "punct",
            "speaker-p",
            "source-p",
            "はい、そうです。",
            ["はいそうです", "はい、そうです。"],
        )
    ]
    report = run_benchmark(records, ks=(1, 2), bootstrap_iterations=10, seed=1)
    row = report.rows[0]
    assert row.baseline_cer > 0.0
    assert row.baseline_lenient_cer == 0.0
    assert row.reference_characters == 8
    assert row.reference_characters_lenient == 6


def test_final_benchmark_rejects_non_test_split() -> None:
    records = [
        _record(
            "calibration",
            "speaker-a",
            "source-a",
            "東京です",
            ["東京です", "京都です"],
            split="calibration",
        )
    ]
    with pytest.raises(ValueError, match="locked test split"):
        run_benchmark(records, ks=(1, 2), bootstrap_iterations=10)
