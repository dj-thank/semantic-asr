"""Deterministic selection for one context × phonetic factorial arm."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..phonetic_experiment.planner import FrozenPhoneticCandidate
from .planner import PreparedContextPhoneticCase
from .protocol import ContextPhoneticArm, ContextPhoneticProtocol


@dataclass(frozen=True, slots=True)
class ScoredContextPhoneticCandidate:
    candidate_id: str
    phonetic_score: float
    context_score: float
    final_score: float
    phonetic_channel_values: tuple[tuple[str, float], ...]
    context_condition: str
    first_pass: bool
    first_pass_selected: bool

    def __post_init__(self) -> None:
        if not self.candidate_id or self.context_condition not in {
            "none",
            "ordered",
            "shuffled",
        }:
            raise ValueError("scored factorial candidate identity is invalid")
        for name in ("phonetic_score", "context_score", "final_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ContextPhoneticDecision:
    arm_name: str
    arm_digest: str
    prepared_case_digest: str
    proposed_candidate_id: str
    effective_candidate_id: str
    first_pass_selected_candidate_id: str
    status: str
    margin: float
    ranked: tuple[ScoredContextPhoneticCandidate, ...]
    reason: str
    selection_latency_ms: float

    def __post_init__(self) -> None:
        if not self.arm_name or self.status not in {"accepted", "provisional"}:
            raise ValueError("factorial decision arm/status is invalid")
        if not self.ranked or self.ranked[0].candidate_id != self.proposed_candidate_id:
            raise ValueError("factorial decision ranking does not match proposed candidate")
        margin = float(self.margin)
        latency = float(self.selection_latency_ms)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("factorial decision margin must be finite and non-negative")
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("factorial selection latency must be finite and non-negative")
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "selection_latency_ms", latency)

    @property
    def changed_proposal(self) -> bool:
        return self.proposed_candidate_id != self.first_pass_selected_candidate_id

    @property
    def changed_effective(self) -> bool:
        return self.effective_candidate_id != self.first_pass_selected_candidate_id

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "armName": self.arm_name,
                "armDigest": self.arm_digest,
                "preparedCaseDigest": self.prepared_case_digest,
                "proposedCandidateId": self.proposed_candidate_id,
                "effectiveCandidateId": self.effective_candidate_id,
                "firstPassSelectedCandidateId": self.first_pass_selected_candidate_id,
                "status": self.status,
                "margin": self.margin,
                "rankedDigests": [row.digest for row in self.ranked],
                "reason": self.reason,
            }
        )


def _phonetic_channel_value(
    candidate: FrozenPhoneticCandidate,
    channel: str,
) -> float | None:
    if channel == "first_pass":
        return 2.0 * candidate.first_pass_posterior - 1.0
    if channel == "phone":
        return candidate.phone_utility
    if channel == "mora":
        return candidate.mora_utility
    if channel == "discrete_unit":
        return candidate.discrete_unit_utility
    raise ValueError(f"unknown factorial phonetic channel: {channel}")


def _context_value(
    prepared: PreparedContextPhoneticCase,
    candidate_id: str,
    condition: str,
) -> float:
    if condition == "none":
        return 0.0
    if condition == "ordered":
        return prepared.ordered.score(candidate_id).value
    if condition == "shuffled":
        return prepared.shuffled.score(candidate_id).value
    raise ValueError(f"unknown context condition: {condition}")


def select_context_phonetic_arm(
    prepared: PreparedContextPhoneticCase,
    arm: ContextPhoneticArm,
    protocol: ContextPhoneticProtocol,
) -> ContextPhoneticDecision:
    """Apply one registered arm without regenerating candidates or context scores."""

    started = time.perf_counter_ns()
    phonetic_arm = arm.resolve_phonetic_arm(protocol.phonetic_protocol)
    active_independent = {
        channel
        for channel, weight in phonetic_arm.channel_weights
        if channel in {"phone", "mora", "discrete_unit"} and weight > 0.0
    }
    ranked: list[ScoredContextPhoneticCandidate] = []
    for candidate in prepared.pool.candidates:
        if not phonetic_arm.allow_outside_first_pass and not candidate.is_first_pass:
            continue
        if not candidate.is_first_pass and not active_independent:
            continue
        channel_values: list[tuple[str, float]] = []
        phonetic_score = 0.0
        eligible = True
        for channel, weight in phonetic_arm.channel_weights:
            if weight == 0.0:
                continue
            value = _phonetic_channel_value(candidate, channel)
            if value is None:
                eligible = False
                break
            channel_values.append((channel, value))
            phonetic_score += weight * value
        if not eligible:
            continue
        context_value = _context_value(
            prepared,
            candidate.candidate_id,
            arm.context_condition,
        )
        context_score = arm.context_weight * context_value
        final_score = phonetic_score + context_score
        if candidate.first_pass_selected:
            final_score += phonetic_arm.retention_bonus
        ranked.append(
            ScoredContextPhoneticCandidate(
                candidate_id=candidate.candidate_id,
                phonetic_score=phonetic_score,
                context_score=context_score,
                final_score=final_score,
                phonetic_channel_values=tuple(channel_values),
                context_condition=arm.context_condition,
                first_pass=candidate.is_first_pass,
                first_pass_selected=candidate.first_pass_selected,
            )
        )
    if not ranked:
        raise ValueError(f"factorial arm {arm.name!r} has no eligible candidates")
    ranked.sort(
        key=lambda row: (
            -row.final_score,
            not row.first_pass_selected,
            row.candidate_id,
        )
    )
    proposed = ranked[0]
    has_runner_up = len(ranked) > 1
    margin = proposed.final_score - ranked[1].final_score if has_runner_up else 0.0
    threshold = phonetic_arm.minimum_margin if arm.minimum_margin is None else arm.minimum_margin
    apply_provisional = (
        phonetic_arm.apply_provisional if arm.apply_provisional is None else arm.apply_provisional
    )
    status = "accepted"
    reason = "margin-accepted"
    if has_runner_up and margin < threshold:
        status = "provisional"
        reason = "below-minimum-margin"
    effective = proposed.candidate_id
    if status == "provisional" and not apply_provisional:
        effective = prepared.pool.first_pass_selected_candidate_id
        reason = "provisional-fallback-to-first-pass"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return ContextPhoneticDecision(
        arm_name=arm.name,
        arm_digest=arm.digest,
        prepared_case_digest=prepared.digest,
        proposed_candidate_id=proposed.candidate_id,
        effective_candidate_id=effective,
        first_pass_selected_candidate_id=prepared.pool.first_pass_selected_candidate_id,
        status=status,
        margin=margin,
        ranked=tuple(ranked),
        reason=reason,
        selection_latency_ms=latency_ms,
    )
