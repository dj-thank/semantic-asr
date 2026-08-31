from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import CandidateEvidence, canonical_json
from .ranker_training import RankerExample

TeacherScoreKind = Literal["logit", "log_likelihood", "preference", "probability"]


def candidate_set_digest(candidates: Sequence[CandidateEvidence]) -> str:
    if len(candidates) < 2:
        raise ValueError("distillation requires at least two candidates")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")
    payload = [
        {
            "candidateId": candidate.candidate_id,
            "text": candidate.text,
            "tokenIds": list(candidate.token_ids),
            "source": candidate.evidence_source,
        }
        for candidate in sorted(candidates, key=lambda row: row.candidate_id)
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TeacherJudgment:
    teacher: str
    candidate_set_sha256: str
    scores: dict[str, float]
    score_kind: TeacherScoreKind
    teacher_revision: str | None = None
    reliability: float = 1.0
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.teacher:
            raise ValueError("teacher name is required")
        if len(self.candidate_set_sha256) != 64:
            raise ValueError("candidate set digest must be SHA-256 hex")
        if self.score_kind not in {
            "logit",
            "log_likelihood",
            "preference",
            "probability",
        }:
            raise ValueError("unknown teacher score kind")
        if not 0 <= self.reliability <= 1:
            raise ValueError("teacher reliability must be in [0, 1]")
        if not self.scores:
            raise ValueError("teacher judgment requires scores")
        if any(not math.isfinite(float(value)) for value in self.scores.values()):
            raise ValueError("teacher scores must be finite")
        if self.score_kind == "probability" and any(
            not 0 <= float(value) <= 1 for value in self.scores.values()
        ):
            raise ValueError("teacher probabilities must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MultiTeacherConfig:
    temperature: float = 1.0
    minimum_active_teachers: int = 1
    maximum_teacher_share: float = 0.60
    entropy_reliability_floor: float = 0.10
    maximum_disagreement: float = 0.42

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("teacher temperature must be positive")
        if self.minimum_active_teachers < 1:
            raise ValueError("minimum_active_teachers must be positive")
        if not 0 < self.maximum_teacher_share <= 1:
            raise ValueError("maximum_teacher_share must be in (0, 1]")
        if not 0 <= self.entropy_reliability_floor <= 1:
            raise ValueError("entropy_reliability_floor must be in [0, 1]")
        if not 0 <= self.maximum_disagreement <= 1:
            raise ValueError("maximum_disagreement must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TeacherConsensus:
    candidate_set_sha256: str
    preference_distribution: dict[str, float]
    teacher_weights: dict[str, float]
    teacher_entropies: dict[str, float]
    disagreement: float
    active_teachers: tuple[str, ...]
    abstained_teachers: tuple[str, ...]
    usable_for_distillation: bool
    reasons: tuple[str, ...]


def _softmax(scores: Sequence[float], temperature: float) -> list[float]:
    maximum = max(scores)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (score - maximum) / temperature))) for score in scores
    ]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _normalize_judgment(
    judgment: TeacherJudgment,
    candidate_ids: Sequence[str],
    *,
    temperature: float,
) -> list[float]:
    values = [float(judgment.scores[candidate_id]) for candidate_id in candidate_ids]
    if judgment.score_kind == "probability":
        total = sum(max(0.0, value) for value in values)
        if total <= 0:
            raise ValueError(f"teacher {judgment.teacher} probability mass is zero")
        return [max(0.0, value) / total for value in values]
    return _softmax(values, temperature)


def _entropy(distribution: Sequence[float]) -> float:
    if len(distribution) <= 1:
        return 0.0
    raw = -sum(value * math.log(value + 1e-12) for value in distribution)
    return min(1.0, max(0.0, raw / math.log(len(distribution))))


def _kl(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        value * math.log((value + 1e-12) / (other + 1e-12))
        for value, other in zip(left, right, strict=True)
        if value > 0
    )


def _jensen_shannon(distributions: Sequence[Sequence[float]]) -> float:
    if len(distributions) <= 1:
        return 0.0
    mixture = [
        sum(distribution[index] for distribution in distributions) / len(distributions)
        for index in range(len(distributions[0]))
    ]
    raw = sum(_kl(distribution, mixture) for distribution in distributions)
    raw /= len(distributions)
    return min(1.0, max(0.0, raw / math.log(max(2, len(mixture)))))


def _bounded_weights(raw_weights: Mapping[str, float], maximum_share: float) -> dict[str, float]:
    if not raw_weights:
        return {}
    weights = {name: max(0.0, float(value)) for name, value in raw_weights.items()}
    if sum(weights.values()) <= 0:
        uniform = 1.0 / len(weights)
        return {name: uniform for name in weights}
    for _iteration in range(32):
        total = sum(weights.values()) or 1.0
        normalized = {name: value / total for name, value in weights.items()}
        over = {name for name, value in normalized.items() if value > maximum_share}
        if not over:
            return normalized
        if len(over) * maximum_share >= 1.0:
            uniform = 1.0 / len(weights)
            return {name: uniform for name in weights}
        remaining_names = [name for name in weights if name not in over]
        remaining_mass = 1.0 - maximum_share * len(over)
        remaining_total = sum(weights[name] for name in remaining_names) or 1.0
        weights = {
            name: (
                maximum_share if name in over else remaining_mass * weights[name] / remaining_total
            )
            for name in weights
        }
    total = sum(weights.values()) or 1.0
    return {name: value / total for name, value in weights.items()}


def aggregate_teacher_judgments(
    candidates: Sequence[CandidateEvidence],
    judgments: Sequence[TeacherJudgment],
    *,
    config: MultiTeacherConfig | None = None,
) -> TeacherConsensus:
    config = config or MultiTeacherConfig()
    digest = candidate_set_digest(candidates)
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    active: list[TeacherJudgment] = []
    abstained: list[str] = []
    distributions: dict[str, list[float]] = {}
    entropies: dict[str, float] = {}
    raw_weights: dict[str, float] = {}

    seen_teachers: set[str] = set()
    for judgment in judgments:
        if judgment.teacher in seen_teachers:
            raise ValueError(f"duplicate teacher judgment: {judgment.teacher}")
        seen_teachers.add(judgment.teacher)
        if judgment.candidate_set_sha256 != digest:
            raise ValueError(
                f"teacher {judgment.teacher} was evaluated on a different candidate set"
            )
        if set(judgment.scores) != set(candidate_ids):
            raise ValueError(
                f"teacher {judgment.teacher} must score every candidate ID exactly once"
            )
        if judgment.abstained:
            abstained.append(judgment.teacher)
            continue
        distribution = _normalize_judgment(
            judgment,
            candidate_ids,
            temperature=config.temperature,
        )
        entropy = _entropy(distribution)
        information = max(config.entropy_reliability_floor, 1.0 - entropy)
        distributions[judgment.teacher] = distribution
        entropies[judgment.teacher] = entropy
        raw_weights[judgment.teacher] = judgment.reliability * information
        active.append(judgment)

    weights = _bounded_weights(raw_weights, config.maximum_teacher_share)
    consensus_values = [0.0] * len(candidate_ids)
    for teacher, distribution in distributions.items():
        for index, value in enumerate(distribution):
            consensus_values[index] += weights.get(teacher, 0.0) * value
    total = sum(consensus_values)
    if total > 0:
        consensus_values = [value / total for value in consensus_values]
    elif candidate_ids:
        consensus_values = [1.0 / len(candidate_ids)] * len(candidate_ids)
    disagreement = _jensen_shannon(list(distributions.values()))
    reasons: list[str] = []
    if len(active) < config.minimum_active_teachers:
        reasons.append("insufficient-active-teachers")
    if disagreement > config.maximum_disagreement:
        reasons.append("teacher-disagreement")
    if not active:
        reasons.append("all-teachers-abstained")
    return TeacherConsensus(
        candidate_set_sha256=digest,
        preference_distribution=dict(zip(candidate_ids, consensus_values, strict=True)),
        teacher_weights=weights,
        teacher_entropies=entropies,
        disagreement=disagreement,
        active_teachers=tuple(sorted(judgment.teacher for judgment in active)),
        abstained_teachers=tuple(sorted(abstained)),
        usable_for_distillation=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def consensus_to_ranker_example(
    *,
    example_id: str,
    candidates: Sequence[CandidateEvidence],
    consensus: TeacherConsensus,
    context: str = "",
    minimum_probability: float = 1e-8,
) -> RankerExample:
    digest = candidate_set_digest(candidates)
    if consensus.candidate_set_sha256 != digest:
        raise ValueError("teacher consensus does not match the candidate set")
    if not consensus.usable_for_distillation:
        raise ValueError("teacher consensus is not usable: " + ",".join(consensus.reasons))
    losses = {
        candidate.candidate_id: -math.log(
            max(
                minimum_probability,
                consensus.preference_distribution[candidate.candidate_id],
            )
        )
        for candidate in candidates
    }
    return RankerExample(
        example_id=example_id,
        candidates=tuple(candidates),
        losses=losses,
        context=context,
    )


def judgment_from_row(
    row: Mapping[str, Any], *, candidate_set_sha256: str | None = None
) -> TeacherJudgment:
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("teacher judgment row requires a scores object")
    return TeacherJudgment(
        teacher=str(row.get("teacher") or ""),
        teacher_revision=(str(row["teacherRevision"]) if row.get("teacherRevision") else None),
        candidate_set_sha256=str(row.get("candidateSetSha256") or candidate_set_sha256 or ""),
        scores={str(key): float(value) for key, value in scores.items()},
        score_kind=str(row.get("scoreKind") or "preference"),
        reliability=float(row.get("reliability", 1.0)),
        abstained=bool(row.get("abstained", False)),
        metadata=dict(row.get("metadata") or {}),
    )


def consensus_json(consensus: TeacherConsensus) -> str:
    return json.dumps(
        {
            "candidateSetSha256": consensus.candidate_set_sha256,
            "preferenceDistribution": consensus.preference_distribution,
            "teacherWeights": consensus.teacher_weights,
            "teacherEntropies": consensus.teacher_entropies,
            "disagreement": consensus.disagreement,
            "activeTeachers": list(consensus.active_teachers),
            "abstainedTeachers": list(consensus.abstained_teachers),
            "usableForDistillation": consensus.usable_for_distillation,
            "reasons": list(consensus.reasons),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
