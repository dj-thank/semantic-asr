from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Literal, Self

from .candidate_pool import SurfaceCandidate
from .contracts import CandidateEvidence

Monotonicity = Literal[-1, 0, 1]
Objective = Literal["pairwise", "listwise", "hybrid"]


def _finite(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(80.0, value))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(-80.0, value))
    return exponent / (1.0 + exponent)


def _softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not values:
        raise ValueError("softmax requires values")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    maximum = max(values)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (value - maximum) / temperature))) for value in values
    ]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    names: tuple[str, ...]
    monotonicity: dict[str, Monotonicity] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique and non-empty")
        unknown = set(self.monotonicity) - set(self.names)
        if unknown:
            raise ValueError(f"monotonicity references unknown features: {unknown}")
        if any(value not in {-1, 0, 1} for value in self.monotonicity.values()):
            raise ValueError("monotonicity values must be -1, 0 or 1")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "names": self.names,
                    "monotonicity": self.monotonicity,
                    "version": self.version,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


DEFAULT_FEATURE_SCHEMA = FeatureSchema(
    names=(
        "aggregate_acoustic_log_likelihood",
        "best_path_log_likelihood",
        "path_mass_bonus",
        "beam_confidence",
        "beam_rank_fraction",
        "mora_score",
        "lexical_score",
        "preservation_score",
        "cross_model_score",
        "source_count",
        "path_count",
        "candidate_length",
        "number_flag",
        "negation_flag",
        "entity_flag",
        "teacher_preference",
        "missing_evidence_fraction",
    ),
    monotonicity={
        "aggregate_acoustic_log_likelihood": 1,
        "best_path_log_likelihood": 1,
        "path_mass_bonus": 1,
        "beam_confidence": 1,
        "mora_score": 1,
        "preservation_score": 1,
        "cross_model_score": 1,
        "source_count": 1,
        "path_count": 1,
        "missing_evidence_fraction": -1,
        # Lexical and teacher evidence intentionally have no positive monotonic
        # constraint: they are language priors, not acoustic proof.
        "lexical_score": 0,
        "teacher_preference": 0,
    },
)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    values: dict[str, float]
    schema_digest: str

    @classmethod
    def create(cls, schema: FeatureSchema, values: Mapping[str, float]) -> Self:
        if set(values) != set(schema.names):
            missing = set(schema.names) - set(values)
            extra = set(values) - set(schema.names)
            raise ValueError(f"feature mismatch; missing={missing}, extra={extra}")
        return cls(
            values={name: _finite(values[name], name=name) for name in schema.names},
            schema_digest=schema.digest,
        )


@dataclass(frozen=True, slots=True)
class TrainingCandidate:
    candidate_id: str
    features: FeatureVector
    target_loss: float
    critical_loss: float = 0.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        for name, value in (
            ("target_loss", self.target_loss),
            ("critical_loss", self.critical_loss),
            ("weight", self.weight),
        ):
            _finite(value, name=name)
        if not 0 <= self.target_loss <= 1 or not 0 <= self.critical_loss <= 1:
            raise ValueError("losses must be in [0, 1]")
        if self.weight <= 0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True, slots=True)
class RankingGroup:
    group_id: str
    candidates: tuple[TrainingCandidate, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id or len(self.candidates) < 2:
            raise ValueError("ranking group requires an ID and at least two candidates")
        if len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate IDs must be unique within a ranking group")
        schema_digests = {candidate.features.schema_digest for candidate in self.candidates}
        if len(schema_digests) != 1:
            raise ValueError("all candidates in a group must share one feature schema")


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    means: dict[str, float]
    scales: dict[str, float]
    schema_digest: str

    @classmethod
    def fit(cls, schema: FeatureSchema, groups: Sequence[RankingGroup]) -> Self:
        rows = [candidate.features.values for group in groups for candidate in group.candidates]
        if not rows:
            raise ValueError("normalizer requires training rows")
        if any(
            candidate.features.schema_digest != schema.digest
            for group in groups
            for candidate in group.candidates
        ):
            raise ValueError("training features do not match schema")
        means = {name: fmean(row[name] for row in rows) for name in schema.names}
        scales: dict[str, float] = {}
        for name in schema.names:
            variance = fmean((row[name] - means[name]) ** 2 for row in rows)
            scales[name] = max(math.sqrt(variance), 1e-6)
        return cls(means=means, scales=scales, schema_digest=schema.digest)

    def transform(self, vector: FeatureVector) -> dict[str, float]:
        if vector.schema_digest != self.schema_digest:
            raise ValueError("feature vector uses a different schema")
        return {
            name: (value - self.means[name]) / self.scales[name]
            for name, value in vector.values.items()
        }


@dataclass(frozen=True, slots=True)
class ConstrainedLinearReranker:
    schema: FeatureSchema
    normalizer: FeatureNormalizer
    weights: dict[str, float]
    bias: float
    objective: Objective
    training_digest: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.weights) != set(self.schema.names):
            raise ValueError("weights must contain every schema feature")
        if self.normalizer.schema_digest != self.schema.digest:
            raise ValueError("normalizer and schema mismatch")
        _finite(self.bias, name="bias")
        for name, value in self.weights.items():
            _finite(value, name=f"weight:{name}")
            monotonicity = self.schema.monotonicity.get(name, 0)
            if monotonicity > 0 and value < 0:
                raise ValueError(f"weight for {name} violates positive monotonicity")
            if monotonicity < 0 and value > 0:
                raise ValueError(f"weight for {name} violates negative monotonicity")

    def score(self, vector: FeatureVector) -> float:
        transformed = self.normalizer.transform(vector)
        return self.bias + sum(self.weights[name] * transformed[name] for name in self.schema.names)

    def rank(self, vectors: Mapping[str, FeatureVector]) -> tuple[tuple[str, float], ...]:
        return tuple(
            sorted(
                ((candidate_id, self.score(vector)) for candidate_id, vector in vectors.items()),
                key=lambda item: (-item[1], item[0]),
            )
        )

    @property
    def digest(self) -> str:
        payload = {
            "schemaDigest": self.schema.digest,
            "means": self.normalizer.means,
            "scales": self.normalizer.scales,
            "weights": self.weights,
            "bias": self.bias,
            "objective": self.objective,
            "trainingDigest": self.training_digest,
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingReport:
    objective: Objective
    epochs: int
    pairwise_loss: float
    listwise_loss: float
    pairwise_accuracy: float
    groups: int
    candidates: int
    training_digest: str


def _training_digest(groups: Sequence[RankingGroup]) -> str:
    payload = [
        {
            "groupId": group.group_id,
            "candidates": [
                {
                    "candidateId": candidate.candidate_id,
                    "features": candidate.features.values,
                    "targetLoss": candidate.target_loss,
                    "criticalLoss": candidate.critical_loss,
                    "weight": candidate.weight,
                }
                for candidate in group.candidates
            ],
        }
        for group in groups
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _project_weights(schema: FeatureSchema, weights: dict[str, float]) -> None:
    for name in schema.names:
        monotonicity = schema.monotonicity.get(name, 0)
        if monotonicity > 0:
            weights[name] = max(0.0, weights[name])
        elif monotonicity < 0:
            weights[name] = min(0.0, weights[name])


def _pairwise_metrics(
    groups: Sequence[RankingGroup],
    model: ConstrainedLinearReranker,
) -> tuple[float, float]:
    losses: list[float] = []
    correct = 0
    total = 0
    for group in groups:
        for index, left in enumerate(group.candidates):
            for right in group.candidates[index + 1 :]:
                if math.isclose(left.target_loss, right.target_loss, abs_tol=1e-12):
                    continue
                preferred, other = (
                    (left, right) if left.target_loss < right.target_loss else (right, left)
                )
                margin = model.score(preferred.features) - model.score(other.features)
                probability = _sigmoid(margin)
                losses.append(-math.log(max(1e-12, probability)))
                correct += int(margin > 0)
                total += 1
    return (fmean(losses) if losses else 0.0, correct / total if total else 1.0)


def _listwise_metric(groups: Sequence[RankingGroup], model: ConstrainedLinearReranker) -> float:
    losses: list[float] = []
    for group in groups:
        target = _softmax(
            [-candidate.target_loss for candidate in group.candidates], temperature=0.15
        )
        predicted = _softmax([model.score(candidate.features) for candidate in group.candidates])
        losses.append(
            -sum(p * math.log(max(1e-12, q)) for p, q in zip(target, predicted, strict=True))
        )
    return fmean(losses) if losses else 0.0


def train_constrained_linear_reranker(
    groups: Sequence[RankingGroup],
    *,
    schema: FeatureSchema = DEFAULT_FEATURE_SCHEMA,
    objective: Objective = "hybrid",
    epochs: int = 250,
    learning_rate: float = 0.03,
    l2: float = 1e-4,
    critical_loss_weight: float = 0.35,
    listwise_weight: float = 0.50,
    seed: int = 0,
) -> tuple[ConstrainedLinearReranker, TrainingReport]:
    """Train an interpretable constrained stacker.

    This baseline intentionally uses deterministic projected SGD. It is not intended
    to replace a cross-encoder; it provides a reproducible lower-cost model and a
    safety reference for larger rerankers.
    """

    if not groups:
        raise ValueError("training groups are required")
    if objective not in {"pairwise", "listwise", "hybrid"}:
        raise ValueError(f"unsupported objective: {objective}")
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("invalid optimization settings")
    if not 0 <= critical_loss_weight <= 1 or not 0 <= listwise_weight <= 1:
        raise ValueError("loss weights must be in [0, 1]")
    if any(
        candidate.features.schema_digest != schema.digest
        for group in groups
        for candidate in group.candidates
    ):
        raise ValueError("training data does not match schema")

    normalizer = FeatureNormalizer.fit(schema, groups)
    transformed = {
        (group.group_id, candidate.candidate_id): normalizer.transform(candidate.features)
        for group in groups
        for candidate in group.candidates
    }
    weights = {name: 0.0 for name in schema.names}
    bias = 0.0
    rng = random.Random(seed)

    for epoch in range(epochs):
        order = list(groups)
        rng.shuffle(order)
        rate = learning_rate / math.sqrt(1.0 + epoch / 40.0)
        for group in order:
            if objective in {"pairwise", "hybrid"}:
                pairs: list[tuple[TrainingCandidate, TrainingCandidate]] = []
                for index, left in enumerate(group.candidates):
                    for right in group.candidates[index + 1 :]:
                        combined_left = (
                            1 - critical_loss_weight
                        ) * left.target_loss + critical_loss_weight * left.critical_loss
                        combined_right = (
                            1 - critical_loss_weight
                        ) * right.target_loss + critical_loss_weight * right.critical_loss
                        if math.isclose(combined_left, combined_right, abs_tol=1e-12):
                            continue
                        pairs.append(
                            (left, right) if combined_left < combined_right else (right, left)
                        )
                rng.shuffle(pairs)
                for preferred, other in pairs:
                    preferred_x = transformed[(group.group_id, preferred.candidate_id)]
                    other_x = transformed[(group.group_id, other.candidate_id)]
                    margin = sum(
                        weights[name] * (preferred_x[name] - other_x[name]) for name in schema.names
                    )
                    # d[-log(sigmoid(margin))]/d margin
                    gradient_factor = _sigmoid(margin) - 1.0
                    gap = max(0.02, abs(preferred.target_loss - other.target_loss))
                    example_weight = math.sqrt(preferred.weight * other.weight) * gap
                    pair_scale = 1.0 if objective == "pairwise" else 1.0 - listwise_weight
                    for name in schema.names:
                        gradient = (
                            pair_scale
                            * example_weight
                            * gradient_factor
                            * (preferred_x[name] - other_x[name])
                            + l2 * weights[name]
                        )
                        weights[name] -= rate * gradient
                    _project_weights(schema, weights)

            if objective in {"listwise", "hybrid"}:
                candidates = group.candidates
                logits = [
                    bias
                    + sum(
                        weights[name] * transformed[(group.group_id, candidate.candidate_id)][name]
                        for name in schema.names
                    )
                    for candidate in candidates
                ]
                predicted = _softmax(logits)
                combined_losses = [
                    (1 - critical_loss_weight) * candidate.target_loss
                    + critical_loss_weight * candidate.critical_loss
                    for candidate in candidates
                ]
                target = _softmax([-loss for loss in combined_losses], temperature=0.15)
                list_scale = 1.0 if objective == "listwise" else listwise_weight
                for candidate, predicted_probability, target_probability in zip(
                    candidates, predicted, target, strict=True
                ):
                    gradient_factor = (
                        list_scale * candidate.weight * (predicted_probability - target_probability)
                    )
                    x = transformed[(group.group_id, candidate.candidate_id)]
                    for name in schema.names:
                        weights[name] -= rate * (gradient_factor * x[name] + l2 * weights[name])
                    bias -= rate * gradient_factor
                _project_weights(schema, weights)

    digest = _training_digest(groups)
    model = ConstrainedLinearReranker(
        schema=schema,
        normalizer=normalizer,
        weights=weights,
        bias=bias,
        objective=objective,
        training_digest=digest,
        metadata={
            "epochs": epochs,
            "learningRate": learning_rate,
            "l2": l2,
            "criticalLossWeight": critical_loss_weight,
            "listwiseWeight": listwise_weight,
            "seed": seed,
        },
    )
    pairwise_loss, pairwise_accuracy = _pairwise_metrics(groups, model)
    listwise_loss = _listwise_metric(groups, model)
    report = TrainingReport(
        objective=objective,
        epochs=epochs,
        pairwise_loss=pairwise_loss,
        listwise_loss=listwise_loss,
        pairwise_accuracy=pairwise_accuracy,
        groups=len(groups),
        candidates=sum(len(group.candidates) for group in groups),
        training_digest=digest,
    )
    return model, report


def _flag(text: str, characters: str) -> float:
    return float(any(character in text for character in characters))


def features_from_candidate_evidence(
    candidate: CandidateEvidence,
    *,
    aggregate_acoustic_log_likelihood: float | None = None,
    best_path_log_likelihood: float | None = None,
    path_mass_bonus: float = 0.0,
    path_count: int = 1,
    source_count: int | None = None,
    teacher_preference: float | None = None,
    schema: FeatureSchema = DEFAULT_FEATURE_SCHEMA,
) -> FeatureVector:
    hypothesis_count = candidate.hypothesis_count or max(1, candidate.rank or 1)
    rank = candidate.rank or hypothesis_count
    rank_fraction = (
        1.0 if hypothesis_count <= 1 else (hypothesis_count - rank) / (hypothesis_count - 1)
    )
    streams = (
        candidate.acoustic,
        candidate.mora,
        candidate.lexical,
        candidate.preservation,
        candidate.cross_model,
    )
    missing = sum(value is None for value in streams) / len(streams)
    values = {
        "aggregate_acoustic_log_likelihood": float(
            aggregate_acoustic_log_likelihood
            if aggregate_acoustic_log_likelihood is not None
            else candidate.avg_logprob or 0.0
        ),
        "best_path_log_likelihood": float(
            best_path_log_likelihood
            if best_path_log_likelihood is not None
            else candidate.avg_logprob or 0.0
        ),
        "path_mass_bonus": float(path_mass_bonus),
        "beam_confidence": float(candidate.beam_confidence or 0.0),
        "beam_rank_fraction": float(rank_fraction),
        "mora_score": float(candidate.mora or 0.0),
        "lexical_score": float(candidate.lexical or 0.0),
        "preservation_score": float(candidate.preservation or 0.0),
        "cross_model_score": float(candidate.cross_model or 0.0),
        "source_count": float(source_count or len(candidate.source_support)),
        "path_count": float(path_count),
        "candidate_length": float(len(candidate.text)),
        "number_flag": float(any(character.isdigit() for character in candidate.text)),
        "negation_flag": _flag(candidate.text, "無未非不")
        + float(any(value in candidate.text for value in ("ない", "ません", "ず", "ぬ"))),
        "entity_flag": float(any("ァ" <= character <= "ヶ" for character in candidate.text)),
        "teacher_preference": float(
            teacher_preference if teacher_preference is not None else candidate.teacher or 0.0
        ),
        "missing_evidence_fraction": float(missing),
    }
    # Keep binary-like flags bounded after combining multiple indicators.
    values["negation_flag"] = min(1.0, values["negation_flag"])
    return FeatureVector.create(schema, values)


def features_from_surface_candidate(
    candidate: SurfaceCandidate,
    *,
    legacy_evidence: CandidateEvidence | None = None,
    teacher_preference: float | None = None,
    schema: FeatureSchema = DEFAULT_FEATURE_SCHEMA,
) -> FeatureVector:
    evidence = legacy_evidence or CandidateEvidence(
        candidate_id=candidate.candidate_id,
        text=candidate.text,
        source="surface-candidate-pool",
        metadata={"sourceSupport": candidate.source_support},
    )
    return features_from_candidate_evidence(
        evidence,
        aggregate_acoustic_log_likelihood=candidate.aggregate_log_likelihood,
        best_path_log_likelihood=candidate.best_path_log_likelihood,
        path_mass_bonus=candidate.path_mass_bonus,
        path_count=candidate.path_count,
        source_count=len(candidate.source_support),
        teacher_preference=teacher_preference,
        schema=schema,
    )
