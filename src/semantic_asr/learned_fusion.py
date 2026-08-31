from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Literal

from .contracts import CandidateEvidence, canonical_json
from .fusion import ACOUSTIC_FAMILY, STREAMS, FusionConfig

TrainingSplit = Literal["train"]


@dataclass(frozen=True, slots=True)
class FusionTrainingExample:
    example_id: str
    group_id: str
    candidates: tuple[CandidateEvidence, ...]
    target_distribution: dict[str, float]
    split: TrainingSplit = "train"

    def __post_init__(self) -> None:
        if not self.example_id or not self.group_id:
            raise ValueError("fusion training example and group IDs are required")
        if self.split != "train":
            raise ValueError("fusion weights may be optimized only on the training split")
        if len(self.candidates) < 2:
            raise ValueError("fusion training requires at least two candidates")
        identifiers = {candidate.candidate_id for candidate in self.candidates}
        if len(identifiers) != len(self.candidates):
            raise ValueError("fusion candidate IDs must be unique")
        if set(self.target_distribution) != identifiers:
            raise ValueError("fusion target distribution must contain every candidate ID")
        values = [float(value) for value in self.target_distribution.values()]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("fusion target values must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("fusion target distribution has zero mass")
        for candidate in self.candidates:
            for stream in STREAMS:
                value = candidate.score(stream)
                if value is not None and (
                    not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                ):
                    raise ValueError(
                        "learned fusion requires held-out calibrated [0, 1] stream values"
                    )


@dataclass(frozen=True, slots=True)
class LearnedFusionConfig:
    epochs: int = 200
    learning_rate: float = 0.08
    l2_to_initial: float = 0.02
    temperature: float = 0.18
    acoustic_family_floor: float = 0.72
    seed: int = 37

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.learning_rate <= 0 or self.temperature <= 0:
            raise ValueError("invalid learned-fusion optimizer configuration")
        if self.l2_to_initial < 0:
            raise ValueError("fusion regularization must be non-negative")
        if not 0 <= self.acoustic_family_floor <= 1:
            raise ValueError("acoustic family floor must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FusionTrainingMetrics:
    cross_entropy: float
    top1_accuracy: float


@dataclass(frozen=True, slots=True)
class LearnedFusionProfile:
    name: str
    weights: dict[str, float]
    acoustic_family_floor: float
    training_manifest_sha256: str
    example_count: int
    group_count: int
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("learned fusion profile name is required")
        if set(self.weights) != set(STREAMS):
            raise ValueError("learned fusion profile must contain every evidence stream")
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("learned fusion weights must be finite and non-negative")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-8):
            raise ValueError("learned fusion weights must sum to one")
        acoustic = sum(self.weights[stream] for stream in ACOUSTIC_FAMILY)
        if acoustic + 1e-9 < self.acoustic_family_floor:
            raise ValueError("learned fusion violates its acoustic family floor")
        if len(self.training_manifest_sha256) != 64:
            raise ValueError("training manifest digest must be SHA-256 hex")
        if self.example_count < 1 or self.group_count < 1:
            raise ValueError("learned fusion counts must be positive")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def to_fusion_config(self, **overrides: object) -> FusionConfig:
        return FusionConfig(
            priors={stream: self.weights[stream] for stream in STREAMS},
            acoustic_family_floor=self.acoustic_family_floor,
            **overrides,
        )


@dataclass(frozen=True, slots=True)
class LearnedFusionResult:
    profile: LearnedFusionProfile
    before: FusionTrainingMetrics
    after: FusionTrainingMetrics
    epoch_losses: tuple[float, ...]


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    maximum = max(values)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (value - maximum) / temperature))) for value in values
    ]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def _project_weights(values: Mapping[str, float], *, acoustic_floor: float) -> dict[str, float]:
    output = {stream: max(0.0, float(values.get(stream, 0.0))) for stream in STREAMS}
    total = sum(output.values())
    if total <= 0:
        output = {stream: 1.0 / len(STREAMS) for stream in STREAMS}
    else:
        output = {stream: value / total for stream, value in output.items()}
    acoustic_total = sum(output[stream] for stream in ACOUSTIC_FAMILY)
    language_streams = [stream for stream in STREAMS if stream not in ACOUSTIC_FAMILY]
    language_total = sum(output[stream] for stream in language_streams)
    if acoustic_total + 1e-12 < acoustic_floor:
        if acoustic_total <= 0:
            acoustic_value = acoustic_floor / len(ACOUSTIC_FAMILY)
            for stream in ACOUSTIC_FAMILY:
                output[stream] = acoustic_value
        else:
            scale = acoustic_floor / acoustic_total
            for stream in ACOUSTIC_FAMILY:
                output[stream] *= scale
        remaining = 1.0 - acoustic_floor
        if language_total <= 0:
            language_value = remaining / max(1, len(language_streams))
            for stream in language_streams:
                output[stream] = language_value
        else:
            scale = remaining / language_total
            for stream in language_streams:
                output[stream] *= scale
    total = sum(output.values()) or 1.0
    return {stream: output[stream] / total for stream in STREAMS}


def _normalized_target(example: FusionTrainingExample) -> dict[str, float]:
    total = sum(example.target_distribution.values())
    return {
        candidate_id: float(value) / total
        for candidate_id, value in example.target_distribution.items()
    }


def _feature(candidate: CandidateEvidence, stream: str) -> float:
    value = candidate.score(stream)
    return 0.0 if value is None else float(value)


def _scores(example: FusionTrainingExample, weights: Mapping[str, float]) -> list[float]:
    return [
        sum(weights[stream] * _feature(candidate, stream) for stream in STREAMS)
        for candidate in example.candidates
    ]


def _metrics(
    examples: Sequence[FusionTrainingExample],
    weights: Mapping[str, float],
    *,
    temperature: float,
) -> FusionTrainingMetrics:
    losses: list[float] = []
    correct = 0
    for example in examples:
        target = _normalized_target(example)
        score_values = _scores(example, weights)
        probabilities = _softmax(score_values, temperature)
        losses.append(
            -sum(
                target[candidate.candidate_id] * math.log(probability + 1e-12)
                for candidate, probability in zip(example.candidates, probabilities, strict=True)
            )
        )
        selected = example.candidates[
            max(range(len(score_values)), key=lambda index: score_values[index])
        ].candidate_id
        oracle = max(target, key=target.get)
        correct += selected == oracle
    return FusionTrainingMetrics(
        cross_entropy=fmean(losses),
        top1_accuracy=correct / len(examples),
    )


def _manifest_digest(examples: Sequence[FusionTrainingExample]) -> str:
    payload = [
        {
            "exampleId": example.example_id,
            "groupId": example.group_id,
            "split": example.split,
            "candidates": [candidate.as_dict() for candidate in example.candidates],
            "targetDistribution": example.target_distribution,
        }
        for example in examples
    ]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def train_constrained_fusion(
    examples: Sequence[FusionTrainingExample],
    *,
    name: str = "semantic-asr-constrained-fusion-v0.2",
    initial_weights: Mapping[str, float] | None = None,
    config: LearnedFusionConfig | None = None,
) -> LearnedFusionResult:
    if not examples:
        raise ValueError("at least one learned-fusion example is required")
    config = config or LearnedFusionConfig()
    initial = _project_weights(
        initial_weights
        or {
            "acoustic": 0.42,
            "mora": 0.23,
            "lexical": 0.08,
            "preservation": 0.12,
            "cross_model": 0.15,
        },
        acoustic_floor=config.acoustic_family_floor,
    )
    weights = dict(initial)
    before = _metrics(examples, weights, temperature=config.temperature)
    rng = random.Random(config.seed)
    epoch_losses: list[float] = []
    for epoch in range(config.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        learning_rate = config.learning_rate / math.sqrt(1.0 + epoch * 0.04)
        losses: list[float] = []
        for index in order:
            example = examples[index]
            target = _normalized_target(example)
            score_values = _scores(example, weights)
            probabilities = _softmax(score_values, config.temperature)
            losses.append(
                -sum(
                    target[candidate.candidate_id] * math.log(probability + 1e-12)
                    for candidate, probability in zip(
                        example.candidates, probabilities, strict=True
                    )
                )
            )
            gradient = {stream: 0.0 for stream in STREAMS}
            for candidate, probability in zip(example.candidates, probabilities, strict=True):
                error = probability - target[candidate.candidate_id]
                for stream in STREAMS:
                    gradient[stream] += error * _feature(candidate, stream) / config.temperature
            proposed = {
                stream: weights[stream]
                - learning_rate
                * (gradient[stream] + config.l2_to_initial * (weights[stream] - initial[stream]))
                for stream in STREAMS
            }
            weights = _project_weights(
                proposed,
                acoustic_floor=config.acoustic_family_floor,
            )
        epoch_losses.append(fmean(losses))
    after = _metrics(examples, weights, temperature=config.temperature)
    profile = LearnedFusionProfile(
        name=name,
        weights=weights,
        acoustic_family_floor=config.acoustic_family_floor,
        training_manifest_sha256=_manifest_digest(examples),
        example_count=len(examples),
        group_count=len({example.group_id for example in examples}),
    )
    return LearnedFusionResult(
        profile=profile,
        before=before,
        after=after,
        epoch_losses=tuple(epoch_losses),
    )
