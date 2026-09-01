from __future__ import annotations

from semantic_asr.pipeline import effort_profile
from semantic_asr.throttling import (
    AdaptiveThrottleConfig,
    RuntimePressure,
    ThrottleState,
    throttle_effort,
    update_throttle_state,
)


def test_high_pressure_disables_expensive_components() -> None:
    profile = effort_profile("research")
    throttled = throttle_effort(
        profile,
        RuntimePressure(
            latency_ratio=2.0,
            memory_pressure=1.0,
            queue_pressure=1.0,
            thermal_pressure=1.0,
            battery_saver=True,
        ),
    )
    assert throttled.level == 3
    assert throttled.maximum_candidates < profile.maximum_candidates
    assert throttled.evidence_budget_ms == 0
    assert throttled.maximum_evidence_actions == 0
    assert not throttled.enable_neural_reranker
    assert not throttled.enable_acoustic_verifier
    assert not throttled.enable_second_ear
    assert not throttled.enable_offline_teacher


def test_medium_pressure_preserves_cheap_ranker_but_sheds_second_ear() -> None:
    profile = effort_profile("edge-gpu")
    throttled = throttle_effort(
        profile,
        RuntimePressure(
            latency_ratio=1.4,
            memory_pressure=0.70,
            queue_pressure=0.50,
            thermal_pressure=0.60,
        ),
    )
    assert throttled.level >= 2
    assert throttled.enable_neural_reranker
    assert not throttled.enable_acoustic_verifier
    assert not throttled.enable_second_ear


def test_hysteresis_requires_stable_low_pressure_before_release() -> None:
    config = AdaptiveThrottleConfig(release_steps=3)
    state = ThrottleState(level=2)
    low = RuntimePressure()
    first = update_throttle_state(low, previous=state, config=config)
    second = update_throttle_state(low, previous=first, config=config)
    third = update_throttle_state(low, previous=second, config=config)
    assert first.level == 2 and first.stable_steps == 1
    assert second.level == 2 and second.stable_steps == 2
    assert third.level == 1 and third.stable_steps == 0


def test_pressure_score_is_bounded() -> None:
    pressure = RuntimePressure(
        latency_ratio=100.0,
        memory_pressure=100.0,
        queue_pressure=100.0,
        thermal_pressure=100.0,
        battery_saver=True,
    )
    assert pressure.score == 1.0
