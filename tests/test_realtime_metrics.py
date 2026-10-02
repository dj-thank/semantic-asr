import json
import math
import subprocess
import sys
import tracemalloc

import pytest

from semantic_asr.realtime_metrics import TimingSummary


def test_empty_timing_summary_has_no_fabricated_duration():
    assert TimingSummary().as_dict() == {
        "count": 0,
        "retained": 0,
        "complete": True,
        "minMs": None,
        "maxMs": None,
        "meanMs": None,
        "p50Ms": None,
        "p95Ms": None,
    }


def test_timing_quantiles_use_existing_interpolation_and_ignore_input_order():
    forward, reverse = TimingSummary(), TimingSummary()
    for duration in (0, 10, 20, 30):
        forward.add(duration)
    for duration in (30, 20, 10, 0):
        reverse.add(duration)
    expected = {
        "count": 4,
        "retained": 4,
        "complete": True,
        "minMs": 0.0,
        "maxMs": 30.0,
        "meanMs": 15.0,
        "p50Ms": 15.0,
        "p95Ms": pytest.approx(28.5),
    }
    assert forward.as_dict() == expected
    assert reverse.as_dict() == expected


def test_zero_and_single_sample_are_real_measurements():
    summary = TimingSummary(max_samples=1)
    summary.add(0)
    assert summary.as_dict() == {
        "count": 1,
        "retained": 1,
        "complete": True,
        "minMs": 0.0,
        "maxMs": 0.0,
        "meanMs": 0.0,
        "p50Ms": 0.0,
        "p95Ms": 0.0,
    }


def test_overflow_preserves_full_stream_aggregates_but_removes_prefix_quantiles():
    summary = TimingSummary(max_samples=2)
    summary.add(10)
    summary.add(20)
    before = summary.as_dict()
    assert before["complete"] is True
    assert before["p50Ms"] == 15.0
    summary.add(300)
    summary.add(0)
    assert summary.as_dict() == {
        "count": 4,
        "retained": 2,
        "complete": False,
        "minMs": 0.0,
        "maxMs": 300.0,
        "meanMs": 82.5,
        "p50Ms": None,
        "p95Ms": None,
    }
    assert before["count"] == 2


@pytest.mark.parametrize("invalid", [True, False, "1", None, object(), complex(1)])
def test_non_numeric_durations_fail_without_mutating_summary(invalid):
    summary = TimingSummary()
    summary.add(10)
    before = summary.as_dict()
    with pytest.raises(TypeError, match="duration_ms"):
        summary.add(invalid)
    assert summary.as_dict() == before


@pytest.mark.parametrize("invalid", [-1, math.nan, math.inf, -math.inf, 10**400])
def test_invalid_numeric_durations_fail_without_mutating_summary(invalid):
    summary = TimingSummary()
    summary.add(10)
    before = summary.as_dict()
    with pytest.raises(ValueError, match="duration_ms"):
        summary.add(invalid)
    assert summary.as_dict() == before


@pytest.mark.parametrize("invalid", [True, False, 1.0, "2", None])
def test_non_integer_sample_budget_is_rejected(invalid):
    with pytest.raises(TypeError, match="max_samples"):
        TimingSummary(max_samples=invalid)


@pytest.mark.parametrize("invalid", [0, -1, 100_001])
def test_out_of_range_sample_budget_is_rejected(invalid):
    with pytest.raises(ValueError, match="max_samples"):
        TimingSummary(max_samples=invalid)


def test_largest_allowed_sample_budget_is_accepted():
    assert TimingSummary(max_samples=100_000).as_dict()["complete"] is True


def test_finite_large_durations_do_not_overflow_mean_or_json():
    summary = TimingSummary(max_samples=2)
    for duration in (1e308, 1e308, 0):
        summary.add(duration)
    result = summary.as_dict()
    assert result["meanMs"] == pytest.approx(1e308 * (2 / 3))
    assert math.isfinite(result["meanMs"])
    json.dumps(result, allow_nan=False)


def test_retained_memory_stays_bounded_after_budget_is_exhausted():
    summary = TimingSummary(max_samples=2)
    summary.add(0)
    summary.add(1)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for index in range(20_000):
            summary.add(float(index))
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert after - before < 32_768
    result = summary.as_dict()
    assert result["count"] == 20_002
    assert result["retained"] == 2
    assert result["complete"] is False


def test_timing_module_and_summary_need_no_optional_model_dependencies():
    program = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'numpy', 'torch', 'sherpa_onnx', 'faster_whisper', 'transformers'
        }:
            raise ImportError('optional dependency blocked: ' + fullname)

sys.meta_path.insert(0, BlockOptional())
from semantic_asr.realtime_metrics import TimingSummary
summary = TimingSummary()
summary.add(7)
assert summary.as_dict()['p95Ms'] == 7
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
