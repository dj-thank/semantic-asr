from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .contracts import CandidateEvidence, canonical_json
from .mbr import semantic_loss
from .rerankers import (
    FEATURE_NAMES,
    LinearCandidateRanker,
    LinearRankerProfile,
    candidate_features,
)


@dataclass(frozen=True, slots=True)
class RankerExample:
    example_id: str
    candidates: tuple[CandidateEvidence, ...]
    losses: dict[str, float]
    context: str = ""

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id is required")
        if len(self.candidates) < 2:
            raise ValueError("ranker example requires at least two candidates")
        identifiers = {candidate.candidate_id for candidate in self.candidates}
        if len(identifiers) != len(self.candidates):
            raise ValueError("candidate IDs must be unique within an example")
        if set(self.losses) != identifiers:
            raise ValueError("losses must contain every candidate ID exactly once")
        if any(not math.isfinite(value) or value < 0 for value in self.losses.values()):
            raise ValueError("training losses must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RankerTrainingConfig:
    epochs: int = 80
    learning_rate: float = 0.08
    l2: float = 0.002
    minimum_loss_gap: float = 1e-6
    maximum_pairs_per_example: int = 128
    seed: int = 17
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("invalid optimizer configuration")
        if self.minimum_loss_gap < 0 or self.maximum_pairs_per_example < 1:
            raise ValueError("invalid pair sampling configuration")
        if not self.feature_names:
            raise ValueError("at least one feature is required")
        unknown = set(self.feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown training features: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class RankerMetrics:
    pair_count: int
    pairwise_accuracy: float
    mean_logistic_loss: float
    mean_positive_margin: float


@dataclass(frozen=True, slots=True)
class RankerTrainingResult:
    profile: LinearRankerProfile
    before: RankerMetrics
    after: RankerMetrics
    epoch_losses: tuple[float, ...]
    example_count: int
    training_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _Pair:
    positive: tuple[float, ...]
    negative: tuple[float, ...]
    weight: float


def _sigmoid_negative_margin(margin: float) -> float:
    if margin >= 0:
        exp_value = math.exp(-margin)
        return exp_value / (1.0 + exp_value)
    exp_value = math.exp(margin)
    return 1.0 / (1.0 + exp_value)


def _logistic_loss(margin: float) -> float:
    if margin >= 0:
        return math.log1p(math.exp(-margin))
    return -margin + math.log1p(math.exp(margin))


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


def _feature_statistics(
    examples: Sequence[RankerExample],
    feature_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    values: dict[str, list[float]] = {name: [] for name in feature_names}
    for example in examples:
        for candidate in example.candidates:
            features = candidate_features(candidate, context=example.context)
            for name in feature_names:
                values[name].append(features[name])
    means = {name: fmean(rows) for name, rows in values.items()}
    scales = {
        name: max(1e-6, pstdev(rows) if len(rows) > 1 else 1.0) for name, rows in values.items()
    }
    return means, scales


def _vector(
    candidate: CandidateEvidence,
    *,
    context: str,
    feature_names: Sequence[str],
    means: Mapping[str, float],
    scales: Mapping[str, float],
) -> tuple[float, ...]:
    features = candidate_features(candidate, context=context)
    return tuple(
        (features[name] - means.get(name, 0.0)) / scales.get(name, 1.0) for name in feature_names
    )


def _pairs(
    examples: Sequence[RankerExample],
    *,
    config: RankerTrainingConfig,
    means: Mapping[str, float],
    scales: Mapping[str, float],
) -> list[_Pair]:
    rng = random.Random(config.seed)
    output: list[_Pair] = []
    for example in examples:
        vectors = {
            candidate.candidate_id: _vector(
                candidate,
                context=example.context,
                feature_names=config.feature_names,
                means=means,
                scales=scales,
            )
            for candidate in example.candidates
        }
        pairs: list[_Pair] = []
        rows = list(example.candidates)
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                left_loss = example.losses[left.candidate_id]
                right_loss = example.losses[right.candidate_id]
                gap = abs(left_loss - right_loss)
                if gap <= config.minimum_loss_gap:
                    continue
                positive, negative = (left, right) if left_loss < right_loss else (right, left)
                pairs.append(
                    _Pair(
                        positive=vectors[positive.candidate_id],
                        negative=vectors[negative.candidate_id],
                        weight=max(config.minimum_loss_gap, gap),
                    )
                )
        rng.shuffle(pairs)
        output.extend(pairs[: config.maximum_pairs_per_example])
    if not output:
        raise ValueError("training set produced no preference pairs")
    return output


def _metrics(pairs: Sequence[_Pair], weights: Sequence[float]) -> RankerMetrics:
    margins: list[float] = []
    losses: list[float] = []
    correct = 0
    for pair in pairs:
        difference = [
            positive - negative
            for positive, negative in zip(pair.positive, pair.negative, strict=True)
        ]
        margin = sum(weight * value for weight, value in zip(weights, difference, strict=True))
        margins.append(margin)
        losses.append(pair.weight * _logistic_loss(margin))
        correct += margin > 0
    return RankerMetrics(
        pair_count=len(pairs),
        pairwise_accuracy=correct / len(pairs),
        mean_logistic_loss=fmean(losses),
        mean_positive_margin=fmean(margins),
    )


def train_pairwise_ranker(
    examples: Sequence[RankerExample],
    *,
    name: str = "semantic-asr-linear-v0.2",
    config: RankerTrainingConfig | None = None,
) -> RankerTrainingResult:
    if not examples:
        raise ValueError("at least one training example is required")
    config = config or RankerTrainingConfig()
    manifest_digest = _manifest_digest(examples)
    means, scales = _feature_statistics(examples, config.feature_names)
    pairs = _pairs(examples, config=config, means=means, scales=scales)
    weights = [0.0] * len(config.feature_names)
    before = _metrics(pairs, weights)
    rng = random.Random(config.seed)
    epoch_losses: list[float] = []
    for epoch in range(config.epochs):
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        learning_rate = config.learning_rate / math.sqrt(1.0 + epoch * 0.05)
        losses: list[float] = []
        for index in indices:
            pair = pairs[index]
            difference = [
                positive - negative
                for positive, negative in zip(pair.positive, pair.negative, strict=True)
            ]
            margin = sum(weight * value for weight, value in zip(weights, difference, strict=True))
            error = _sigmoid_negative_margin(margin)
            losses.append(pair.weight * _logistic_loss(margin))
            for feature_index, value in enumerate(difference):
                gradient = -pair.weight * error * value + config.l2 * weights[feature_index]
                weights[feature_index] -= learning_rate * gradient
        epoch_losses.append(fmean(losses))
    after = _metrics(pairs, weights)
    profile = LinearRankerProfile(
        name=name,
        weights=dict(zip(config.feature_names, weights, strict=True)),
        bias=0.0,
        feature_mean=means,
        feature_scale=scales,
        training_manifest_sha256=manifest_digest,
        version="2",
    )
    return RankerTrainingResult(
        profile=profile,
        before=before,
        after=after,
        epoch_losses=tuple(epoch_losses),
        example_count=len(examples),
        training_manifest_sha256=manifest_digest,
    )


def evaluate_ranker(
    ranker: LinearCandidateRanker,
    examples: Sequence[RankerExample],
) -> RankerMetrics:
    pairs: list[_Pair] = []
    for example in examples:
        scores = ranker.score(example.candidates, context=example.context)
        rows = list(example.candidates)
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                left_loss = example.losses[left.candidate_id]
                right_loss = example.losses[right.candidate_id]
                if left_loss == right_loss:
                    continue
                positive, negative = (left, right) if left_loss < right_loss else (right, left)
                pairs.append(
                    _Pair(
                        positive=(float(scores[positive.candidate_id]),),
                        negative=(float(scores[negative.candidate_id]),),
                        weight=abs(left_loss - right_loss),
                    )
                )
    return _metrics(pairs, [1.0])


def _candidate_from_row(row: Mapping[str, Any]) -> CandidateEvidence:
    aliases = {
        "candidateId": "candidate_id",
        "tokenIds": "token_ids",
        "crossModel": "cross_model",
        "moraUnits": "mora_units",
        "hypothesisCount": "hypothesis_count",
        "sequenceScore": "sequence_score",
        "avgLogprob": "avg_logprob",
        "beamConfidence": "beam_confidence",
    }
    return CandidateEvidence.from_dict(
        {aliases.get(str(key), str(key)): value for key, value in row.items()}
    )


def example_from_row(row: Mapping[str, Any], *, line_number: int = 0) -> RankerExample:
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"ranker row {line_number} has no candidates array")
    candidates = tuple(_candidate_from_row(value) for value in raw_candidates)
    raw_losses = row.get("losses")
    if isinstance(raw_losses, Mapping):
        losses = {str(key): float(value) for key, value in raw_losses.items()}
    elif isinstance(row.get("reference"), str):
        reference = CandidateEvidence("__reference__", str(row["reference"]))
        losses = {
            candidate.candidate_id: semantic_loss(candidate, reference)[0]
            for candidate in candidates
        }
    else:
        raise ValueError(f"ranker row {line_number} requires losses or reference")
    return RankerExample(
        example_id=str(row.get("exampleId") or row.get("example_id") or line_number),
        candidates=candidates,
        losses=losses,
        context=str(row.get("context") or ""),
    )


def load_jsonl_examples(path: str | Path) -> list[RankerExample]:
    """Load ranker examples, skipping utterances that carry no ranking signal.

    Real N-best lists frequently collapse to one surface after path aggregation and
    loop-guard rejection. Such rows cannot form a pair or a list and are skipped;
    the count is reported on stderr so a dataset dominated by them is visible.
    """

    output: list[RankerExample] = []
    skipped = 0
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"ranker row {line_number} must be an object")
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and len(candidates) < 2:
            skipped += 1
            continue
        output.append(example_from_row(payload, line_number=line_number))
    if skipped:
        print(
            json.dumps({"skippedSingleCandidateRows": skipped, "path": str(path)}),
            file=sys.stderr,
        )
    if not output:
        raise ValueError("ranker dataset is empty")
    return output


def write_training_result(result: RankerTrainingResult, path: str | Path) -> None:
    payload = {
        "schemaVersion": "2.0.0",
        "profile": result.profile.as_dict(),
        "before": asdict(result.before),
        "after": asdict(result.after),
        "epochLosses": list(result.epoch_losses),
        "exampleCount": result.example_count,
        "trainingManifestSha256": result.training_manifest_sha256,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
