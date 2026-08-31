from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from .contracts import CandidateEvidence


@dataclass(frozen=True, slots=True)
class AdaptiveKConfig:
    minimum_k: int = 2
    maximum_k: int = 12
    posterior_mass_target: float = 0.94
    minimum_incremental_mass: float = 0.015
    high_risk_threshold: float = 0.28
    high_risk_extra: int = 3
    criticality_threshold: float = 0.72
    criticality_extra: int = 2
    minimum_diverse_surfaces: int = 2

    def __post_init__(self) -> None:
        if self.minimum_k < 1 or self.maximum_k < self.minimum_k:
            raise ValueError("invalid adaptive K bounds")
        for value in (
            self.posterior_mass_target,
            self.minimum_incremental_mass,
            self.high_risk_threshold,
            self.criticality_threshold,
        ):
            if not 0 <= value <= 1:
                raise ValueError("adaptive K thresholds must be in [0, 1]")
        if self.high_risk_extra < 0 or self.criticality_extra < 0:
            raise ValueError("adaptive K extras must be non-negative")
        if self.minimum_diverse_surfaces < 1:
            raise ValueError("minimum_diverse_surfaces must be positive")


@dataclass(frozen=True, slots=True)
class AdaptiveKDecision:
    selected_candidate_ids: tuple[str, ...]
    k: int
    cumulative_posterior: float
    selected_mass: tuple[float, ...]
    reason: str
    risk: float
    semantic_criticality: float
    available_candidates: int


@dataclass(frozen=True, slots=True)
class RiskControlProfile:
    """Held-out policy thresholds.

    `verified` means the profile was produced from a declared calibration
    manifest. It does not independently prove a theorem or benchmark result.
    """

    name: str
    maximum_risk: float
    minimum_coverage: float
    minimum_samples: int
    calibration_manifest_sha256: str
    verified: bool = False
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("risk-control profile name is required")
        if not 0 <= self.maximum_risk <= 1 or not 0 <= self.minimum_coverage <= 1:
            raise ValueError("risk-control thresholds must be in [0, 1]")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if len(self.calibration_manifest_sha256) != 64:
            raise ValueError("calibration manifest digest must be SHA-256 hex")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _normalize_posterior(
    candidates: Sequence[CandidateEvidence],
    posterior: Mapping[str, float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for candidate in candidates:
        value = float(posterior.get(candidate.candidate_id, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError("posterior values must be finite and non-negative")
        values[candidate.candidate_id] = value
    total = sum(values.values())
    if total <= 0:
        probability = 1.0 / len(candidates)
        return {candidate.candidate_id: probability for candidate in candidates}
    return {candidate_id: value / total for candidate_id, value in values.items()}


def select_adaptive_k(
    candidates: Sequence[CandidateEvidence],
    posterior: Mapping[str, float],
    *,
    selective_risk: float = 0.0,
    semantic_criticality: float = 0.0,
    config: AdaptiveKConfig | None = None,
) -> AdaptiveKDecision:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    if not 0 <= selective_risk <= 1 or not 0 <= semantic_criticality <= 1:
        raise ValueError("risk and criticality must be in [0, 1]")
    config = config or AdaptiveKConfig()
    probabilities = _normalize_posterior(candidates, posterior)
    ordered = sorted(
        candidates,
        key=lambda candidate: (-probabilities[candidate.candidate_id], candidate.candidate_id),
    )
    maximum = min(config.maximum_k, len(ordered))
    target_k = min(config.minimum_k, maximum)
    if selective_risk >= config.high_risk_threshold:
        target_k = min(maximum, target_k + config.high_risk_extra)
    if semantic_criticality >= config.criticality_threshold:
        target_k = min(maximum, target_k + config.criticality_extra)

    selected: list[CandidateEvidence] = []
    masses: list[float] = []
    cumulative = 0.0
    seen_surfaces: set[str] = set()
    stopping_reason = "maximum-k"
    for candidate in ordered[:maximum]:
        probability = probabilities[candidate.candidate_id]
        selected.append(candidate)
        masses.append(probability)
        cumulative += probability
        seen_surfaces.add(candidate.text)
        enough_count = len(selected) >= target_k
        enough_diversity = len(seen_surfaces) >= min(
            config.minimum_diverse_surfaces, len(ordered)
        )
        enough_mass = cumulative >= config.posterior_mass_target
        negligible_tail = probability < config.minimum_incremental_mass
        if enough_count and enough_diversity and (enough_mass or negligible_tail):
            stopping_reason = (
                "posterior-mass-target" if enough_mass else "negligible-incremental-mass"
            )
            break

    return AdaptiveKDecision(
        selected_candidate_ids=tuple(candidate.candidate_id for candidate in selected),
        k=len(selected),
        cumulative_posterior=min(1.0, cumulative),
        selected_mass=tuple(masses),
        reason=stopping_reason,
        risk=selective_risk,
        semantic_criticality=semantic_criticality,
        available_candidates=len(candidates),
    )


def apply_risk_control(
    decision: AdaptiveKDecision,
    *,
    empirical_risk: float,
    empirical_coverage: float,
    sample_count: int,
    profile: RiskControlProfile,
) -> bool:
    """Return whether a held-out risk-control gate permits automatic acceptance."""

    if not profile.verified:
        return False
    if sample_count < profile.minimum_samples:
        return False
    if not 0 <= empirical_risk <= 1 or not 0 <= empirical_coverage <= 1:
        raise ValueError("empirical risk and coverage must be in [0, 1]")
    return (
        empirical_risk <= profile.maximum_risk
        and empirical_coverage >= profile.minimum_coverage
        and decision.cumulative_posterior >= profile.minimum_coverage
    )
