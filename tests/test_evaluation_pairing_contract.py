"""Evaluation must not turn missing, duplicated or invalid results into success."""

from dataclasses import replace

import pytest

from semantic_asr.experiment import (
    SampleResult,
    evaluate_claim_gate,
    paired_bootstrap_comparison,
)


def row(sample, system, score):
    return SampleResult(sample, system, {"cer": score}, 1.0)


def pair():
    return [
        row("a", "base", 0.2),
        row("a", "new", 0.1),
        row("b", "base", 0.2),
        row("b", "new", 0.1),
    ]


def compare(rows, **kwargs):
    return paired_bootstrap_comparison(
        rows,
        baseline_system="base",
        candidate_system="new",
        metric="cer",
        iterations=100,
        **kwargs,
    )


def test_missing_output_is_not_silently_dropped():
    with pytest.raises(ValueError, match="cohort"):
        compare(pair()[:-1])


@pytest.mark.parametrize("first", [True, False])
def test_duplicate_sample_system_pair_is_never_last_write_wins(first):
    extra = row("a", "base", 0.9)
    with pytest.raises(ValueError, match="duplicate"):
        compare([extra, *pair()] if first else [*pair(), extra])


def test_missing_metric_is_not_silently_dropped():
    rows = pair()
    rows[-1] = SampleResult("b", "new", {"other": 0.0}, 1.0)
    with pytest.raises(ValueError, match="metric"):
        compare(rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.2"])
def test_mutated_metric_is_revalidated_at_comparison(value):
    rows = pair()
    rows[-1].metrics["cer"] = value
    with pytest.raises((ValueError, TypeError)):
        compare(rows)


@pytest.mark.parametrize(
    "name",
    [
        "critical_metric_delta",
        "latency_ratio",
        "maximum_critical_regression",
        "maximum_latency_ratio",
        "minimum_better_probability",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_claim_gate_rejects_invalid_controls(name, value):
    controls = {"critical_metric_delta": 0.0, "latency_ratio": 1.0, name: value}
    with pytest.raises((ValueError, TypeError)):
        evaluate_claim_gate(compare(pair()), **controls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("probability_candidate_better", float("nan")),
        ("probability_candidate_better", 1.1),
        ("samples", True),
        ("samples", 0),
        ("upper_delta", float("nan")),
        ("lower_is_better", "yes"),
        ("mean_delta", -42.0),
    ],
)
def test_comparison_receipt_cannot_claim_invalid_evidence(field, value):
    with pytest.raises((ValueError, TypeError)):
        changed = replace(compare(pair()), **{field: value})
        evaluate_claim_gate(changed, critical_metric_delta=0.0, latency_ratio=1.0)


def test_one_observation_is_not_a_publication_quality_gate():
    gate = evaluate_claim_gate(compare(pair()[:2]), critical_metric_delta=0.0, latency_ratio=1.0)
    assert not gate.passed


def test_legitimate_complete_pair_retains_legacy_result():
    result = compare(pair())
    assert result.mean_delta == pytest.approx(-0.1)
    assert result.probability_candidate_better == 1.0
    assert compare(list(reversed(pair()))) == result
