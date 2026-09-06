"""Count, cohort and resampling contracts, independent of any acoustic model."""

import math
import random
from dataclasses import replace

import pytest

from semantic_asr.benchmark import BenchmarkRow, paired_group_bootstrap
from semantic_asr.evaluation import (
    edit_distance,
    normalize_characters,
    normalize_characters_lenient,
)
from semantic_asr.experiment import (
    PairedErrorCounts,
    SampleResult,
    evaluate_claim_gate,
    paired_bootstrap_comparison,
    paired_error_rate_comparison,
)


def compare(rows, **kwargs):
    return paired_error_rate_comparison(
        rows,
        baseline_system="base",
        candidate_system="new",
        metric="cer",
        iterations=1000,
        seed=17,
        **kwargs,
    )


def test_corpus_and_utterance_mean_can_have_opposite_directions():
    counts = [PairedErrorCounts("short", "s", 1, 1, 0), PairedErrorCounts("long", "l", 100, 0, 2)]
    corpus = compare(counts)
    rates = [
        SampleResult(r.sample_id, name, {"cer": err / r.reference_units}, 0)
        for r in counts
        for name, err in (("base", r.baseline_errors), ("new", r.candidate_errors))
    ]
    macro = paired_bootstrap_comparison(
        rates, baseline_system="base", candidate_system="new", metric="cer", iterations=100
    )
    assert corpus.mean_delta == pytest.approx(1 / 101)  # Worse by corpus CER.
    assert macro.mean_delta == pytest.approx(-0.49)  # Better by utterance mean.
    assert corpus.aggregation == "corpus-error-rate" and macro.aggregation == "utterance-mean"


def test_cluster_totals_match_independently_resampled_corpus_counts():
    rows = [
        PairedErrorCounts("a1", "a", 10, 3, 1),
        PairedErrorCounts("a2", "a", 5, 1, 1),
        PairedErrorCounts("b1", "b", 40, 1, 2),
    ]
    result = compare(rows)
    assert result.samples == 3 and result.group_count == 2
    rng = random.Random(17)
    draws = []
    grouped = [[r for r in rows if r.group_id == g] for g in ("a", "b")]
    for _ in range(1000):
        chosen = [r for _ in grouped for r in grouped[rng.randrange(2)]]
        draws.append(
            sum(r.candidate_errors - r.baseline_errors for r in chosen)
            / sum(r.reference_units for r in chosen)
        )
    draws.sort()
    assert result.lower_delta == pytest.approx(draws[25])
    assert result.upper_delta == pytest.approx(draws[974])
    assert result == compare(list(reversed(rows)))


def test_identical_system_and_ties_never_count_as_improvement():
    r = compare([PairedErrorCounts("a", "g1", 10, 1, 1), PairedErrorCounts("b", "g2", 3, 2, 2)])
    assert (r.mean_delta, r.lower_delta, r.upper_delta, r.probability_candidate_better) == (
        0,
        0,
        0,
        0,
    )
    assert not evaluate_claim_gate(r, critical_metric_delta=0, latency_ratio=1).passed


def test_zero_reference_insertions_are_retained_inside_positive_reference_group():
    r = compare(
        [PairedErrorCounts("speech", "g1", 10, 0, 0), PairedErrorCounts("silence", "g1", 0, 0, 3)]
    )
    assert r.candidate_mean == 0.3 and r.samples == 2
    assert not evaluate_claim_gate(r, critical_metric_delta=0, latency_ratio=1).passed


@pytest.mark.parametrize("rows", [[], [PairedErrorCounts("silent", "g", 0, 0, 4)]])
def test_undefined_silence_only_denominators_are_never_zero_quality(rows):
    with pytest.raises(ValueError):
        compare(rows)


def test_insertions_allow_error_rate_greater_than_one():
    assert compare([PairedErrorCounts("s", "g", 1, 3, 2)]).baseline_mean == 3.0


@pytest.mark.parametrize("name", ["reference_units", "baseline_errors", "candidate_errors"])
@pytest.mark.parametrize("value", [-1, True, 1.2, float("nan")])
def test_counts_are_nonnegative_integers(name, value):
    with pytest.raises((ValueError, TypeError)):
        replace(PairedErrorCounts("x", "g", 1, 0, 0), **{name: value})


def test_duplicate_count_rows_are_rejected():
    row = PairedErrorCounts("s", "g", 3, 1, 0)
    with pytest.raises(ValueError, match="duplicate"):
        compare([row, row])


@pytest.mark.parametrize(
    "groups", [{}, {"a": "x"}, {"a": "x", "b": ""}, {"a": "x", "b": "y", "extra": "z"}]
)
def test_group_mapping_must_match_complete_cohort(groups):
    rows = [
        SampleResult(i, s, {"cer": v}, 0)
        for i in ("a", "b")
        for s, v in (("base", 0.2), ("new", 0.1))
    ]
    with pytest.raises(ValueError):
        paired_bootstrap_comparison(
            rows,
            baseline_system="base",
            candidate_system="new",
            metric="cer",
            iterations=100,
            group_ids=groups,
        )


def test_correlated_clips_do_not_manufacture_independent_groups():
    rows = [
        SampleResult(i, s, {"cer": v}, 0)
        for i in ("a", "b", "c")
        for s, v in (("base", 0.2), ("new", 0.1))
    ]
    r = paired_bootstrap_comparison(
        rows,
        baseline_system="base",
        candidate_system="new",
        metric="cer",
        iterations=100,
        group_ids={i: "same" for i in ("a", "b", "c")},
    )
    assert r.group_count == 1 and r.samples == 3
    assert not evaluate_claim_gate(r, critical_metric_delta=0, latency_ratio=1).passed


def test_abstained_rows_are_not_silently_removed():
    rows = [
        SampleResult(i, s, {"cer": v}, 0, accepted=False)
        for i in ("a", "b")
        for s, v in (("base", 0.2), ("new", 0.1))
    ]
    r = paired_bootstrap_comparison(
        rows, baseline_system="base", candidate_system="new", metric="cer", iterations=100
    )
    assert r.samples == 2


@pytest.mark.parametrize(
    "ref,hyp,strict,lenient",
    [
        ("50m", "50マイル", 3, 3),
        ("ない", "ある", 2, 2),
        ("つながる", "繋がる", 2, 2),
        ("えー", "え", 1, 1),
        ("がっこう", "がこう", 1, 1),
        ("はい。", "はい", 1, 0),
        ("", "あ", 1, 1),
        ("", "", 0, 0),
    ],
)
def test_fixed_japanese_metric_golden(ref, hyp, strict, lenient):
    assert edit_distance(normalize_characters(ref), normalize_characters(hyp)) == strict
    assert (
        edit_distance(normalize_characters_lenient(ref), normalize_characters_lenient(hyp))
        == lenient
    )


def benchmark_row(sample="x", left=0.2, right=0.1):
    return BenchmarkRow(
        sample, "g", "fixture", False, "base", "new", "new", left, right, right, {}, 0, 1, False
    )


def test_benchmark_metric_callbacks_are_evaluated_once_per_row():
    calls = {"left": 0, "right": 0}

    def metric(row, side):
        calls[side] += 1
        return row.baseline_cer if side == "left" else row.cascade_cer

    r = paired_group_bootstrap(
        [benchmark_row()],
        left=lambda r: metric(r, "left"),
        right=lambda r: metric(r, "right"),
        iterations=10,
    )
    assert calls == {"left": 1, "right": 1}
    assert r.eligible_samples == 1 and r.group_count == 1


@pytest.mark.parametrize("left,right", [(None, 0.1), (0.1, None), (math.nan, 0.1), (0.1, True)])
def test_benchmark_cannot_hide_asymmetric_or_nonfinite_values(left, right):
    with pytest.raises((TypeError, ValueError)):
        paired_group_bootstrap(
            [benchmark_row(left=left, right=right)],
            left=lambda r: r.baseline_cer,
            right=lambda r: r.cascade_cer,
            iterations=10,
        )


def test_jointly_undefined_annotation_exclusions_are_counted():
    r = paired_group_bootstrap(
        [benchmark_row(), benchmark_row("uncertain", None, None)],
        left=lambda r: r.baseline_cer,
        right=lambda r: r.cascade_cer,
        iterations=10,
    )
    assert r.excluded_samples == 1 and r.eligible_samples == 1


def test_benchmark_duplicate_rows_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        paired_group_bootstrap(
            [benchmark_row(), benchmark_row()],
            left=lambda r: r.baseline_cer,
            right=lambda r: r.cascade_cer,
            iterations=10,
        )


@pytest.mark.parametrize("expected", [["a", "missing"], ["a", "a"], [""], []])
def test_frozen_expected_cohort_rejects_both_systems_missing_same_sample(expected):
    with pytest.raises(ValueError):
        compare([PairedErrorCounts("a", "g", 1, 1, 0)], expected_sample_ids=expected)


@pytest.mark.parametrize(
    "control,value",
    [
        ("iterations", True),
        ("iterations", 100.5),
        ("iterations", 0),
        ("confidence", True),
        ("confidence", float("nan")),
        ("confidence", "0.95"),
        ("seed", True),
        ("seed", 1.5),
        ("lower_is_better", "yes"),
    ],
)
def test_bootstrap_control_validation(control, value):
    rows = [SampleResult("a", s, {"cer": v}, 0) for s, v in (("base", 0.2), ("new", 0.1))]
    kwargs = {"iterations": 100, control: value}
    with pytest.raises((ValueError, TypeError)):
        paired_bootstrap_comparison(
            rows, baseline_system="base", candidate_system="new", metric="cer", **kwargs
        )


def test_candidate_equals_baseline_system_is_zero_difference():
    rows = [SampleResult("a", "same", {"cer": 0.2}, 0), SampleResult("b", "same", {"cer": 0.1}, 0)]
    c = paired_bootstrap_comparison(
        rows, baseline_system="same", candidate_system="same", metric="cer", iterations=100
    )
    assert c.mean_delta == c.lower_delta == c.upper_delta == 0
