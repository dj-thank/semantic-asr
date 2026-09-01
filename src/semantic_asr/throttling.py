from __future__ import annotations

import math
from dataclasses import dataclass

from .pipeline import ComputeEffortProfile


@dataclass(frozen=True, slots=True)
class RuntimePressure:
    latency_ratio: float = 0.0
    memory_pressure: float = 0.0
    queue_pressure: float = 0.0
    thermal_pressure: float = 0.0
    battery_saver: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("latency_ratio", self.latency_ratio),
            ("memory_pressure", self.memory_pressure),
            ("queue_pressure", self.queue_pressure),
            ("thermal_pressure", self.thermal_pressure),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def score(self) -> float:
        latency = min(2.0, self.latency_ratio) / 2.0
        memory = min(1.0, self.memory_pressure)
        queue = min(1.0, self.queue_pressure)
        thermal = min(1.0, self.thermal_pressure)
        battery = 1.0 if self.battery_saver else 0.0
        return min(
            1.0,
            0.34 * latency + 0.26 * memory + 0.18 * queue + 0.16 * thermal + 0.06 * battery,
        )


@dataclass(frozen=True, slots=True)
class ThrottleState:
    level: int = 0
    stable_steps: int = 0

    def __post_init__(self) -> None:
        if self.level not in {0, 1, 2, 3}:
            raise ValueError("throttle level must be 0, 1, 2, or 3")
        if self.stable_steps < 0:
            raise ValueError("stable_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class AdaptiveThrottleConfig:
    level_one_threshold: float = 0.32
    level_two_threshold: float = 0.56
    level_three_threshold: float = 0.78
    release_hysteresis: float = 0.10
    release_steps: int = 3

    def __post_init__(self) -> None:
        thresholds = (
            self.level_one_threshold,
            self.level_two_threshold,
            self.level_three_threshold,
        )
        if not 0 <= thresholds[0] < thresholds[1] < thresholds[2] <= 1:
            raise ValueError("throttle thresholds must be ordered inside [0, 1]")
        if not 0 <= self.release_hysteresis < 1:
            raise ValueError("release_hysteresis must be in [0, 1)")
        if self.release_steps < 1:
            raise ValueError("release_steps must be positive")


@dataclass(frozen=True, slots=True)
class ThrottledEffort:
    source_profile: str
    pressure_score: float
    level: int
    maximum_candidates: int
    evidence_budget_ms: int
    maximum_evidence_actions: int
    enable_neural_reranker: bool
    enable_acoustic_verifier: bool
    enable_second_ear: bool
    enable_offline_teacher: bool
    state: ThrottleState
    reasons: tuple[str, ...]


def _target_level(score: float, config: AdaptiveThrottleConfig) -> int:
    if score >= config.level_three_threshold:
        return 3
    if score >= config.level_two_threshold:
        return 2
    if score >= config.level_one_threshold:
        return 1
    return 0


def _release_boundary(level: int, config: AdaptiveThrottleConfig) -> float:
    threshold = {
        1: config.level_one_threshold,
        2: config.level_two_threshold,
        3: config.level_three_threshold,
    }.get(level, 0.0)
    return max(0.0, threshold - config.release_hysteresis)


def update_throttle_state(
    pressure: RuntimePressure,
    *,
    previous: ThrottleState | None = None,
    config: AdaptiveThrottleConfig | None = None,
) -> ThrottleState:
    config = config or AdaptiveThrottleConfig()
    previous = previous or ThrottleState()
    target = _target_level(pressure.score, config)
    if target > previous.level:
        return ThrottleState(level=target, stable_steps=0)
    if target == previous.level:
        return ThrottleState(level=previous.level, stable_steps=0)
    if pressure.score > _release_boundary(previous.level, config):
        return ThrottleState(level=previous.level, stable_steps=0)
    stable_steps = previous.stable_steps + 1
    if stable_steps < config.release_steps:
        return ThrottleState(level=previous.level, stable_steps=stable_steps)
    return ThrottleState(level=max(target, previous.level - 1), stable_steps=0)


def throttle_effort(
    profile: ComputeEffortProfile,
    pressure: RuntimePressure,
    *,
    previous: ThrottleState | None = None,
    config: AdaptiveThrottleConfig | None = None,
) -> ThrottledEffort:
    state = update_throttle_state(pressure, previous=previous, config=config)
    level = state.level
    candidate_scale = (1.0, 0.75, 0.50, 0.25)[level]
    budget_scale = (1.0, 0.60, 0.25, 0.0)[level]
    action_scale = (1.0, 0.75, 0.50, 0.0)[level]
    maximum_candidates = max(1, round(profile.maximum_candidates * candidate_scale))
    evidence_budget_ms = max(0, round(profile.evidence_budget_ms * budget_scale))
    maximum_actions = max(0, round(profile.maximum_evidence_actions * action_scale))

    reasons: list[str] = []
    if pressure.latency_ratio >= 1.0:
        reasons.append("latency-over-target")
    if pressure.memory_pressure >= 0.75:
        reasons.append("memory-pressure")
    if pressure.queue_pressure >= 0.75:
        reasons.append("queue-pressure")
    if pressure.thermal_pressure >= 0.75:
        reasons.append("thermal-pressure")
    if pressure.battery_saver:
        reasons.append("battery-saver")
    if level:
        reasons.append(f"throttle-level-{level}")

    return ThrottledEffort(
        source_profile=profile.name,
        pressure_score=pressure.score,
        level=level,
        maximum_candidates=maximum_candidates,
        evidence_budget_ms=evidence_budget_ms,
        maximum_evidence_actions=maximum_actions,
        enable_neural_reranker=profile.enable_neural_reranker and level < 3,
        enable_acoustic_verifier=profile.enable_acoustic_verifier and level < 2,
        enable_second_ear=profile.enable_second_ear and level < 2,
        enable_offline_teacher=profile.enable_offline_teacher and level < 1,
        state=state,
        reasons=tuple(reasons),
    )
