from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev

from .contracts import canonical_json
from .ranker_training import RankerExample
from .rerankers import FEATURE_NAMES, LinearRankerProfile, candidate_features


@dataclass(frozen=True, slots=True)
class ListwiseTrainingConfig:
    epochs: int = 160
    learning_rate: float = 0.05
    l2: float = 0.002
    temperature: float = 1.0
    gradient_clip: float = 5.0
    seed: int = 29
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("invalid listwise optimizer configuration")
        if self.l2 < 0 or self.temperature <= 0 or self.gradient_clip <= 0:
            raise ValueError("invalid listwise regularization configuration")
        if not self.feature_names:
            raise ValueError("at least one listwise feature is required")
        unknown = set(self.feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown listwise features: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class ListwiseMetrics:
    mean_expected_loss: float
    mean_top1_loss: float
    mean_oracle_loss: float
    mean_rank_regret: float
    top1_oracle_rate: float


@dataclass(frozen=True, slots=True)
class ListwiseTrainingResult:
    profile: LinearRankerProfile
    before: ListwiseMetrics
    after: ListwiseMetrics
    epoch_losses: tuple[float, ...]
    example_count: int
    training_manifest_sha256: str


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    maximum = max(values)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (value - maximum) / temperature)))
        for value in values
    ]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _manifest_digest(examples: Sequence[RankerExample]) -> str:
    payload = [
        {
            "exampleId": example.example_id,
            "context": example.context,
            "candidates": [candidate.as_dict() for candidate in example.candidates],
            "losses": example.losses,
        }
        for example in examples
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _statistics(
    examples: Sequence[RankerExample], feature_names: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
    rows: dict[str, list[float]] = {name: [] for name in feature_names}
    for example in examples:
        for candidate in example.candidates:
            features = candidate_features(candidate, context=example.context)
            for name in feature_names:
                rows[name].append(features[name])
    means = {name: fmean(values) for name, values in rows.items()}
    scales = {
        name: max(1e-6, pstdev(values) if len(values) > 1 else 1.0)
        for name, values in rows.items()
    }
    return means, scales


def _vectors(
    example: RankerExample,
    *,
    feature_names: Sequence[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[tuple[float, ...]]:
    output: list[tuple[float, ...]] = []
    for candidate in example.candidates:
        features = candidate_features(candidate, context=example.context)
        output.append(
            tuple(
                (features[name] - means[name]) / scales[name]
                for name in feature_names
            )
        )
    return output


def _scores(vectors: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    return [
        sum(weight * value for weight, value in zip(weights, vector, strict=True))
        for vector in vectors
    ]


def _metrics(
    examples: Sequence[RankerExample],
    vectors: Sequence[Sequence[Sequence[float]]],
    weights: Sequence[float],
    *,
    temperature: float,
) -> ListwiseMetrics:
    expected_losses: list[float] = []
    top1_losses: list[float] = []
    oracle_losses: list[float] = []
    regrets: list[float] = []
    oracle_hits = 0
    for example, example_vectors in zip(examples, vectors, strict=True):
        score_values = _scores(example_vectors, weights)
        probabilities = _softmax(score_values, temperature)
        losses = [example.losses[candidate.candidate_id] for candidate in example.candidates]
        expected = sum(
            probability * loss
            for probability, loss in zip(probabilities, losses, strict=True)
        )
        selected_index = max(range(len(score_values)), key=lambda index: score_values[index])
        oracle = min(losses)
        top1 = losses[selected_index]
        expected_losses.append(expected)
        top1_losses.append(top1)
        oracle_losses.append(oracle)
        regrets.append(max(0.0, top1 - oracle))
        oracle_hits += math.isclose(top1, oracle, rel_tol=0.0, abs_tol=1e-12)
    return ListwiseMetrics(
        mean_expected_loss=fmean(expected_losses),
        mean_top1_loss=fmean(top1_losses),
        mean_oracle_loss=fmean(oracle_losses),
        mean_rank_regret=fmean(regrets),
        top1_oracle_rate=oracle_hits / len(examples),
    )


def train_listwise_semantic_mwer(
    examples: Sequence[RankerExample],
    *,
    name: str = "semantic-asr-listwise-mwer-v0.2",
    config: ListwiseTrainingConfig | None = None,
) -> ListwiseTrainingResult:
    if not examples:
        raise ValueError("at least one listwise training example is required")
    config = config or ListwiseTrainingConfig()
    means, scales = _statistics(examples, config.feature_names)
    vectors = [
        _vectors(
            example,
            feature_names=config.feature_names,
            means=means,
            scales=scales,
        )
        for example in examples
    ]
    weights = [0.0] * len(config.feature_names)
    before = _metrics(
        examples,
        vectors,
        weights,
        temperature=config.temperature,
    )
    rng = random.Random(config.seed)
    epoch_losses: list[float] = []
    for epoch in range(config.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        learning_rate = config.learning_rate / math.sqrt(1.0 + epoch * 0.04)
        epoch_expected: list[float] = []
        for example_index in order:
            example = examples[example_index]
            example_vectors = vectors[example_index]
            score_values = _scores(example_vectors, weights)
            probabilities = _softmax(score_values, config.temperature)
            losses = [
                example.losses[candidate.candidate_id]
                for candidate in example.candidates
            ]
            expected = sum(
                probability * loss
                for probability, loss in zip(probabilities, losses, strict=True)
            )
            epoch_expected.append(expected)
            gradient = [0.0] * len(weights)
            for probability, loss, vector in zip(
                probabilities, losses, example_vectors, strict=True
            ):
                coefficient = probability * (loss - expected) / config.temperature
                for index, value in enumerate(vector):
                    gradient[index] += coefficient * value
            norm = math.sqrt(sum(value * value for value in gradient))
            scale = min(1.0, config.gradient_clip / max(1e-12, norm))
            for index in range(len(weights)):
                regularized = gradient[index] * scale + config.l2 * weights[index]
                weights[index] -= learning_rate * regularized
        epoch_losses.append(fmean(epoch_expected))
    after = _metrics(
        examples,
        vectors,
        weights,
        temperature=config.temperature,
    )
    manifest_digest = _manifest_digest(examples)
    profile = LinearRankerProfile(
        name=name,
        weights=dict(zip(config.feature_names, weights, strict=True)),
        bias=0.0,
        feature_mean=means,
        feature_scale=scales,
        training_manifest_sha256=manifest_digest,
        version="listwise-mwer-1",
    )
    return ListwiseTrainingResult(
        profile=profile,
        before=before,
        after=after,
        epoch_losses=tuple(epoch_losses),
        example_count=len(examples),
        training_manifest_sha256=manifest_digest,
    )
