"""Deterministic selection over one shared frozen phonetic candidate pool."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from .planner import FrozenPhoneticCandidate, FrozenPhoneticCandidatePool
from .protocol import PhoneticAblationArm


@dataclass(frozen=True, slots=True)
class ScoredPhoneticCandidate:
    candidate_id: str
    score: float
    channel_values: tuple[tuple[str, float], ...]
    first_pass: bool
    first_pass_selected: bool

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("phonetic candidate score must be finite")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class PhoneticAblationDecision:
    arm_name: str
    arm_digest: str
    pool_digest: str
    proposed_candidate_id: str
    effective_candidate_id: str
    first_pass_selected_candidate_id: str
    status: str
    margin: float
    proposed_score: float
    ranked: tuple[ScoredPhoneticCandidate, ...]
    selection_latency_ms: float
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "provisional"}:
            raise ValueError("ablation decision status must be accepted or provisional")
        if not self.ranked or self.ranked[0].candidate_id != self.proposed_candidate_id:
            raise ValueError("ranked candidates do not match the proposed candidate")
        margin = float(self.margin)
        latency = float(self.selection_latency_ms)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("decision margin must be finite and non-negative")
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("selection latency must be finite and non-negative")
        if self.status == "provisional" and self.effective_candidate_id not in {
            self.proposed_candidate_id,
            self.first_pass_selected_candidate_id,
        }:
            raise ValueError("provisional effective candidate has an invalid fallback")
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
                **asdict(self),
                "ranked": [asdict(row) for row in self.ranked],
            }
        )


def _channel_value(
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
    raise ValueError(f"unknown phonetic evidence channel: {channel}")


def select_phonetic_arm(
    pool: FrozenPhoneticCandidatePool,
    arm: PhoneticAblationArm,
) -> PhoneticAblationDecision:
    started = time.perf_counter_ns()
    rows: list[ScoredPhoneticCandidate] = []
    for candidate in pool.candidates:
        if not arm.allow_outside_first_pass and not candidate.is_first_pass:
            continue
        values: list[tuple[str, float]] = []
        score = 0.0
        eligible = True
        for channel, weight in arm.channel_weights:
            if weight == 0.0:
                continue
            value = _channel_value(candidate, channel)
            if value is None:
                eligible = False
                break
            values.append((channel, value))
            score += weight * value
        if not eligible:
            continue
        if candidate.first_pass_selected:
            score += arm.retention_bonus
        rows.append(
            ScoredPhoneticCandidate(
                candidate_id=candidate.candidate_id,
                score=score,
                channel_values=tuple(values),
                first_pass=candidate.is_first_pass,
                first_pass_selected=candidate.first_pass_selected,
            )
        )
    if not rows:
        raise ValueError(f"ablation arm {arm.name!r} has no eligible candidates")
    rows.sort(
        key=lambda row: (
            -row.score,
            not row.first_pass_selected,
            row.candidate_id,
        )
    )
    proposed = rows[0]
    has_runner_up = len(rows) > 1
    margin = proposed.score - rows[1].score if has_runner_up else 0.0
    status = "accepted"
    reason = "margin-accepted"
    if has_runner_up and margin < arm.minimum_margin:
        status = "provisional"
        reason = "below-minimum-margin"
    effective = proposed.candidate_id
    if status == "provisional" and not arm.apply_provisional:
        effective = pool.first_pass_selected_candidate_id
        reason = "provisional-fallback-to-first-pass"
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return PhoneticAblationDecision(
        arm_name=arm.name,
        arm_digest=arm.digest,
        pool_digest=pool.digest,
        proposed_candidate_id=proposed.candidate_id,
        effective_candidate_id=effective,
        first_pass_selected_candidate_id=pool.first_pass_selected_candidate_id,
        status=status,
        margin=margin,
        proposed_score=proposed.score,
        ranked=tuple(rows),
        selection_latency_ms=latency_ms,
        reason=reason,
    )
