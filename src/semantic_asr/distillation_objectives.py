"""Candidate-level distillation objectives for edge ASR rerankers.

The objectives translate compact-model training ideas into the N-best domain:
Top-K membership mass is matched separately from the conditional distribution
inside Top-K, and pairwise alignment is normalized by candidate length so a
student does not learn a blanket preference for short transcripts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class DecoupledTopKLoss:
    total: float
    membership_kl: float
    conditional_kl: float
    teacher_topk_mass: float
    student_topk_mass: float
    topk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LengthNormalizedPairwiseLoss:
    total: float
    relative: float
    absolute_anchor: float
    chosen_per_token: float
    rejected_per_token: float


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution must not be empty")
    numeric = {str(key): float(value) for key, value in values.items()}
    if any(not math.isfinite(value) or value < 0 for value in numeric.values()):
        raise ValueError("distribution values must be finite and non-negative")
    total = sum(numeric.values())
    if total <= 0:
        raise ValueError("distribution must contain positive mass")
    return {key: value / total for key, value in numeric.items()}


def _binary_kl(left: float, right: float) -> float:
    left = min(1.0 - _EPSILON, max(_EPSILON, left))
    right = min(1.0 - _EPSILON, max(_EPSILON, right))
    return left * math.log(left / right) + (1.0 - left) * math.log((1.0 - left) / (1.0 - right))


def _tempered(values: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return _normalize(
        {key: max(_EPSILON, value) ** (1.0 / temperature) for key, value in values.items()}
    )


def _kl(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if set(left) != set(right):
        raise ValueError("KL distributions must have identical support")
    return sum(
        probability * math.log((probability + _EPSILON) / (right[key] + _EPSILON))
        for key, probability in left.items()
    )


def decoupled_topk_kl(
    teacher: Mapping[str, float],
    student: Mapping[str, float],
    *,
    k: int,
    temperature: float = 1.0,
) -> DecoupledTopKLoss:
    """Match retained mass and relative preference inside the teacher Top-K."""

    teacher_distribution = _normalize(teacher)
    student_distribution = _normalize(student)
    if set(teacher_distribution) != set(student_distribution):
        raise ValueError("teacher and student must score identical candidate IDs")
    if k < 1:
        raise ValueError("k must be positive")
    ordered = sorted(
        teacher_distribution,
        key=lambda candidate_id: (-teacher_distribution[candidate_id], candidate_id),
    )
    topk_ids = tuple(ordered[: min(k, len(ordered))])
    teacher_mass = sum(teacher_distribution[candidate_id] for candidate_id in topk_ids)
    student_mass = sum(student_distribution[candidate_id] for candidate_id in topk_ids)
    membership = _binary_kl(teacher_mass, student_mass)
    teacher_conditional = _tempered(
        _normalize({candidate_id: teacher_distribution[candidate_id] for candidate_id in topk_ids}),
        temperature,
    )
    student_conditional = _tempered(
        _normalize({candidate_id: student_distribution[candidate_id] for candidate_id in topk_ids}),
        temperature,
    )
    conditional = (
        teacher_mass
        * (temperature**2)
        * _kl(
            teacher_conditional,
            student_conditional,
        )
    )
    return DecoupledTopKLoss(
        total=membership + conditional,
        membership_kl=membership,
        conditional_kl=conditional,
        teacher_topk_mass=teacher_mass,
        student_topk_mass=student_mass,
        topk_ids=topk_ids,
    )


def _softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))


def length_normalized_pairwise_alignment(
    *,
    chosen_logprob: float,
    rejected_logprob: float,
    chosen_length: int,
    rejected_length: int,
    reference_chosen_logprob: float = 0.0,
    reference_rejected_logprob: float = 0.0,
    beta: float = 1.0,
    margin: float = 0.0,
    absolute_weight: float = 0.2,
) -> LengthNormalizedPairwiseLoss:
    """Pairwise direct alignment with per-token normalization and an anchor."""

    if chosen_length < 1 or rejected_length < 1:
        raise ValueError("candidate lengths must be positive")
    values = (
        chosen_logprob,
        rejected_logprob,
        reference_chosen_logprob,
        reference_rejected_logprob,
        beta,
        margin,
        absolute_weight,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("alignment inputs must be finite")
    if beta <= 0 or absolute_weight < 0:
        raise ValueError("beta must be positive and absolute_weight non-negative")
    chosen = (chosen_logprob - reference_chosen_logprob) / chosen_length
    rejected = (rejected_logprob - reference_rejected_logprob) / rejected_length
    relative = _softplus(-beta * ((chosen - rejected) - margin))
    anchor = _softplus(-beta * chosen)
    return LengthNormalizedPairwiseLoss(
        total=relative + absolute_weight * anchor,
        relative=relative,
        absolute_anchor=anchor,
        chosen_per_token=chosen,
        rejected_per_token=rejected,
    )
