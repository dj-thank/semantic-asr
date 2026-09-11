"""Dependency-free complete-document ranker with immutable training and calibration artifacts.

The model is a conservative baseline for document-lattice path ranking. It combines signed hashed
Japanese character n-grams over the candidate and bidirectional context with explicit dense evidence
features. Training is pairwise within one recording/document group. It never generates text and its
output remains a bounded preference rather than a correctness probability.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .contracts import canonical_json, sha256_json
from .deliberation_lattice import DocumentContext, LatticeArc, path_digest
from .global_scorer import GlobalPathScore

DENSE_FEATURE_NAMES = (
    "local_score",
    "overlap_score",
    "mean_audio_support",
    "changed_window_fraction",
    "generated_window_fraction",
    "ambiguous_overlap_fraction",
    "candidate_length_log",
    "left_context_overlap",
    "right_context_overlap",
    "topic_overlap",
    "retained_path",
)


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _surface(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _character_set(text: str) -> set[str]:
    return {value for value in _surface(text) if not value.isspace()}


def _overlap(left: str, right: str) -> float:
    left_set = _character_set(left)
    right_set = _character_set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set.intersection(right_set)) / len(left_set.union(right_set))


def _digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentFeatureConfig:
    hash_dimension: int = 32_768
    ngram_min: int = 2
    ngram_max: int = 5
    candidate_character_limit: int = 20_000
    context_character_limit: int = 8_000
    boundary_character_limit: int = 96
    hash_seed: str = "semantic-asr-document-ranker-v1"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "hash_dimension",
            "ngram_min",
            "ngram_max",
            "candidate_character_limit",
            "context_character_limit",
            "boundary_character_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.ngram_min > self.ngram_max:
            raise ValueError("ngram_min must not exceed ngram_max")
        if not self.hash_seed:
            raise ValueError("hash_seed is required")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentRankInput:
    text: str
    left_context: str = ""
    right_context: str = ""
    topic_summary: str = ""
    entity_ids: tuple[str, ...] = ()
    local_score: float = 0.0
    overlap_score: float = 0.0
    mean_audio_support: float = 0.0
    changed_window_count: int = 0
    generated_window_count: int = 0
    ambiguous_overlap_count: int = 0
    window_count: int = 1
    retained_path: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("document rank input text must not be empty")
        for name in ("local_score", "overlap_score", "mean_audio_support"):
            object.__setattr__(self, name, _strict_float(getattr(self, name), name=name))
        for name in (
            "changed_window_count",
            "generated_window_count",
            "ambiguous_overlap_count",
            "window_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.window_count < 1:
            raise ValueError("window_count must be positive")
        for name in (
            "changed_window_count",
            "generated_window_count",
            "ambiguous_overlap_count",
        ):
            if getattr(self, name) > self.window_count:
                raise ValueError(f"{name} cannot exceed window_count")
        object.__setattr__(self, "entity_ids", tuple(dict.fromkeys(self.entity_ids)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dense_values(self) -> tuple[float, ...]:
        denominator = max(1, self.window_count)
        return (
            self.local_score,
            self.overlap_score,
            self.mean_audio_support,
            self.changed_window_count / denominator,
            self.generated_window_count / denominator,
            self.ambiguous_overlap_count / denominator,
            math.log1p(len(self.text)),
            _overlap(self.text, self.left_context),
            _overlap(self.text, self.right_context),
            _overlap(self.text, self.topic_summary),
            float(self.retained_path),
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "text": self.text,
                "leftContext": self.left_context,
                "rightContext": self.right_context,
                "topicSummary": self.topic_summary,
                "entityIds": self.entity_ids,
                "denseValues": self.dense_values,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class SparseDocumentFeatures:
    indices: tuple[tuple[int, float], ...]
    dense: tuple[float, ...]
    input_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        if len(self.dense) != len(DENSE_FEATURE_NAMES):
            raise ValueError("dense document feature width mismatch")
        if len({index for index, _ in self.indices}) != len(self.indices):
            raise ValueError("sparse document feature indexes must be unique")
        if any(index < 0 for index, _ in self.indices):
            raise ValueError("sparse document feature indexes must be non-negative")
        for _, value in self.indices:
            _strict_float(value, name="sparse feature value")
        for value in self.dense:
            _strict_float(value, name="dense feature value")
        if not _is_sha256(self.input_digest) or not _is_sha256(self.config_digest):
            raise ValueError("document feature digests must be SHA-256 values")


class HashedDocumentFeatureExtractor:
    def __init__(self, config: DocumentFeatureConfig | None = None) -> None:
        self.config = config or DocumentFeatureConfig()

    def _index_sign(self, namespace: str, feature: str) -> tuple[int, float]:
        payload = f"{self.config.hash_seed}\0{namespace}\0{feature}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        index = int.from_bytes(digest[:8], "little") % self.config.hash_dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        return index, sign

    def _ngrams(self, text: str) -> Iterable[str]:
        value = _surface(text)
        for width in range(self.config.ngram_min, self.config.ngram_max + 1):
            for index in range(max(0, len(value) - width + 1)):
                yield value[index : index + width]

    def _add_text(self, output: dict[int, float], namespace: str, text: str) -> None:
        counts: dict[str, int] = defaultdict(int)
        for value in self._ngrams(text):
            counts[value] += 1
        for value, count in counts.items():
            index, sign = self._index_sign(namespace, value)
            output[index] += sign * math.log1p(count)

    def extract(self, value: DocumentRankInput) -> SparseDocumentFeatures:
        candidate = value.text[: self.config.candidate_character_limit]
        left = value.left_context[-self.config.context_character_limit :]
        right = value.right_context[: self.config.context_character_limit]
        boundary = self.config.boundary_character_limit
        sparse: dict[int, float] = defaultdict(float)
        self._add_text(sparse, "candidate", candidate)
        self._add_text(sparse, "left", left)
        self._add_text(sparse, "right", right)
        self._add_text(sparse, "topic", value.topic_summary)
        self._add_text(sparse, "left-candidate-boundary", left[-boundary:] + candidate[:boundary])
        self._add_text(
            sparse,
            "candidate-right-boundary",
            candidate[-boundary:] + right[:boundary],
        )
        for entity_id in value.entity_ids:
            index, sign = self._index_sign("entity-id", entity_id)
            sparse[index] += sign
        return SparseDocumentFeatures(
            indices=tuple(
                sorted(
                    (index, feature_value)
                    for index, feature_value in sparse.items()
                    if feature_value
                )
            ),
            dense=value.dense_values,
            input_digest=value.digest,
            config_digest=self.config.digest,
        )


@dataclass(frozen=True, slots=True)
class DocumentRankExample:
    group_id: str
    candidate_id: str
    rank_input: DocumentRankInput
    character_error_rate: float
    critical_error_count: int = 0
    first_pass_exact: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id or not self.candidate_id:
            raise ValueError("document rank example requires group_id and candidate_id")
        cer = _strict_float(self.character_error_rate, name="character_error_rate")
        if cer < 0.0:
            raise ValueError("character_error_rate must be non-negative")
        if isinstance(self.critical_error_count, bool) or self.critical_error_count < 0:
            raise ValueError("critical_error_count must be non-negative")
        object.__setattr__(self, "character_error_rate", cer)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "groupId": self.group_id,
                "candidateId": self.candidate_id,
                "rankInputDigest": self.rank_input.digest,
                "characterErrorRate": self.character_error_rate,
                "criticalErrorCount": self.critical_error_count,
                "firstPassExact": self.first_pass_exact,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentRankTrainingConfig:
    epochs: int = 12
    learning_rate: float = 0.05
    l2: float = 1e-5
    minimum_objective_gap: float = 1e-8
    critical_error_weight: float = 2.0
    false_correction_weight: float = 4.0
    maximum_pairs_per_group: int = 256
    random_seed: int = 0
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("epochs", "maximum_pairs_per_group"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer")
        for name in (
            "learning_rate",
            "l2",
            "minimum_objective_gap",
            "critical_error_weight",
            "false_correction_weight",
        ):
            value = _strict_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentRankerCalibration:
    center: float
    scale: float
    calibration_manifest_sha256: str
    revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        center = _strict_float(self.center, name="ranker calibration center")
        scale = _strict_float(self.scale, name="ranker calibration scale")
        if scale <= 0.0:
            raise ValueError("ranker calibration scale must be positive")
        if not _is_sha256(self.calibration_manifest_sha256):
            raise ValueError("calibration_manifest_sha256 must be a SHA-256 value")
        if not self.revision:
            raise ValueError("ranker calibration revision is required")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def transform(self, raw_score: float) -> float:
        return math.tanh((_strict_float(raw_score, name="raw ranker score") - self.center) / self.scale)


@dataclass(frozen=True, slots=True)
class DocumentLinearRanker:
    feature_config: DocumentFeatureConfig
    sparse_weights: tuple[tuple[int, float], ...]
    dense_means: tuple[float, ...]
    dense_scales: tuple[float, ...]
    dense_weights: tuple[float, ...]
    bias: float
    training_config_digest: str
    training_manifest_sha256: str
    training_example_digest: str
    epoch_losses: tuple[float, ...]
    pairwise_accuracy: float
    revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if len(self.dense_means) != len(DENSE_FEATURE_NAMES):
            raise ValueError("dense mean width mismatch")
        if len(self.dense_scales) != len(DENSE_FEATURE_NAMES):
            raise ValueError("dense scale width mismatch")
        if len(self.dense_weights) != len(DENSE_FEATURE_NAMES):
            raise ValueError("dense weight width mismatch")
        if len({index for index, _ in self.sparse_weights}) != len(self.sparse_weights):
            raise ValueError("sparse model indexes must be unique")
        if any(index < 0 or index >= self.feature_config.hash_dimension for index, _ in self.sparse_weights):
            raise ValueError("sparse model index is outside the feature dimension")
        if any(scale <= 0.0 for scale in self.dense_scales):
            raise ValueError("dense scales must be positive")
        for value in (
            *self.dense_means,
            *self.dense_scales,
            *self.dense_weights,
            self.bias,
            *self.epoch_losses,
            self.pairwise_accuracy,
        ):
            _strict_float(value, name="document ranker parameter")
        if not 0.0 <= self.pairwise_accuracy <= 1.0:
            raise ValueError("pairwise_accuracy must be in [0, 1]")
        for digest in (
            self.training_config_digest,
            self.training_manifest_sha256,
            self.training_example_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("document ranker digests must be SHA-256 values")
        if not self.revision:
            raise ValueError("document ranker revision is required")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "featureConfigDigest": self.feature_config.digest,
                "sparseWeights": self.sparse_weights,
                "denseMeans": self.dense_means,
                "denseScales": self.dense_scales,
                "denseWeights": self.dense_weights,
                "bias": self.bias,
                "trainingConfigDigest": self.training_config_digest,
                "trainingManifestSha256": self.training_manifest_sha256,
                "trainingExampleDigest": self.training_example_digest,
                "epochLosses": self.epoch_losses,
                "pairwiseAccuracy": self.pairwise_accuracy,
                "revision": self.revision,
            }
        )

    @property
    def sparse_weight_map(self) -> dict[int, float]:
        return dict(self.sparse_weights)

    def raw_score(self, features: SparseDocumentFeatures) -> float:
        if features.config_digest != self.feature_config.digest:
            raise ValueError("document features were created with a different feature config")
        sparse = self.sparse_weight_map
        score = self.bias + sum(sparse.get(index, 0.0) * value for index, value in features.indices)
        for value, mean, scale, weight in zip(
            features.dense,
            self.dense_means,
            self.dense_scales,
            self.dense_weights,
            strict=True,
        ):
            score += weight * ((value - mean) / scale)
        return score

    def score_input(self, value: DocumentRankInput) -> float:
        return self.raw_score(HashedDocumentFeatureExtractor(self.feature_config).extract(value))


@dataclass(frozen=True, slots=True)
class DocumentRankerArtifact:
    model: DocumentLinearRanker
    calibration: DocumentRankerCalibration
    test_manifest_sha256: str
    test_pairwise_accuracy: float
    test_group_top1_accuracy: float
    serialization: str = "canonical-json"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.serialization != "canonical-json":
            raise ValueError("document ranker artifact must use canonical JSON")
        if not _is_sha256(self.test_manifest_sha256):
            raise ValueError("test_manifest_sha256 must be a SHA-256 value")
        for name in ("test_pairwise_accuracy", "test_group_top1_accuracy"):
            value = _strict_float(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "modelDigest": self.model.digest,
                "calibrationDigest": self.calibration.digest,
                "testManifestSha256": self.test_manifest_sha256,
                "testPairwiseAccuracy": self.test_pairwise_accuracy,
                "testGroupTop1Accuracy": self.test_group_top1_accuracy,
                "serialization": self.serialization,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "model": {
                "featureConfig": asdict(self.model.feature_config),
                "sparseWeights": self.model.sparse_weights,
                "denseMeans": self.model.dense_means,
                "denseScales": self.model.dense_scales,
                "denseWeights": self.model.dense_weights,
                "bias": self.model.bias,
                "trainingConfigDigest": self.model.training_config_digest,
                "trainingManifestSha256": self.model.training_manifest_sha256,
                "trainingExampleDigest": self.model.training_example_digest,
                "epochLosses": self.model.epoch_losses,
                "pairwiseAccuracy": self.model.pairwise_accuracy,
                "revision": self.model.revision,
                "schemaVersion": self.model.schema_version,
            },
            "calibration": asdict(self.calibration),
            "testManifestSha256": self.test_manifest_sha256,
            "testPairwiseAccuracy": self.test_pairwise_accuracy,
            "testGroupTop1Accuracy": self.test_group_top1_accuracy,
            "serialization": self.serialization,
            "artifactDigest": self.digest,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> DocumentRankerArtifact:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(payload) != {
            "schemaVersion",
            "model",
            "calibration",
            "testManifestSha256",
            "testPairwiseAccuracy",
            "testGroupTop1Accuracy",
            "serialization",
            "artifactDigest",
        }:
            raise ValueError("document ranker artifact schema is not exact")
        model_payload = dict(payload["model"])
        feature = DocumentFeatureConfig(**model_payload.pop("featureConfig"))
        model = DocumentLinearRanker(
            feature_config=feature,
            sparse_weights=tuple(tuple(row) for row in model_payload.pop("sparseWeights")),
            dense_means=tuple(model_payload.pop("denseMeans")),
            dense_scales=tuple(model_payload.pop("denseScales")),
            dense_weights=tuple(model_payload.pop("denseWeights")),
            epoch_losses=tuple(model_payload.pop("epochLosses")),
            **model_payload,
        )
        calibration = DocumentRankerCalibration(**payload["calibration"])
        artifact = cls(
            model=model,
            calibration=calibration,
            test_manifest_sha256=payload["testManifestSha256"],
            test_pairwise_accuracy=payload["testPairwiseAccuracy"],
            test_group_top1_accuracy=payload["testGroupTop1Accuracy"],
            serialization=payload["serialization"],
            schema_version=payload["schemaVersion"],
        )
        if artifact.digest != payload["artifactDigest"]:
            raise ValueError("document ranker artifact digest mismatch")
        return artifact


@dataclass(frozen=True, slots=True)
class _TrainingRow:
    example: DocumentRankExample
    features: SparseDocumentFeatures
    objective: float


def _objective(example: DocumentRankExample, config: DocumentRankTrainingConfig) -> float:
    false_correction = example.first_pass_exact and not example.rank_input.retained_path
    return (
        example.character_error_rate
        + config.critical_error_weight * example.critical_error_count
        + config.false_correction_weight * float(false_correction)
    )


def _dense_statistics(rows: Sequence[_TrainingRow]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    columns = list(zip(*(row.features.dense for row in rows), strict=True))
    means = tuple(statistics.fmean(column) for column in columns)
    scales = tuple(
        max(1e-6, math.sqrt(statistics.fmean((value - mean) ** 2 for value in column)))
        for column, mean in zip(columns, means, strict=True)
    )
    return means, scales


def _difference(
    chosen: _TrainingRow,
    rejected: _TrainingRow,
    means: Sequence[float],
    scales: Sequence[float],
) -> tuple[dict[int, float], tuple[float, ...]]:
    sparse: dict[int, float] = defaultdict(float)
    for index, value in chosen.features.indices:
        sparse[index] += value
    for index, value in rejected.features.indices:
        sparse[index] -= value
    dense = tuple(
        ((left - mean) / scale) - ((right - mean) / scale)
        for left, right, mean, scale in zip(
            chosen.features.dense,
            rejected.features.dense,
            means,
            scales,
            strict=True,
        )
    )
    return {index: value for index, value in sparse.items() if value}, dense


def _pairs(
    rows: Sequence[_TrainingRow],
    config: DocumentRankTrainingConfig,
) -> list[tuple[_TrainingRow, _TrainingRow, float]]:
    by_group: dict[str, list[_TrainingRow]] = defaultdict(list)
    for row in rows:
        by_group[row.example.group_id].append(row)
    output = []
    randomizer = random.Random(config.random_seed)
    for group_id in sorted(by_group):
        group = by_group[group_id]
        pairs = []
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                gap = abs(left.objective - right.objective)
                if gap <= config.minimum_objective_gap:
                    continue
                chosen, rejected = (left, right) if left.objective < right.objective else (right, left)
                weight = 1.0 + gap
                if chosen.example.critical_error_count != rejected.example.critical_error_count:
                    weight += config.critical_error_weight
                if chosen.example.rank_input.retained_path and chosen.example.first_pass_exact:
                    weight += config.false_correction_weight
                pairs.append((chosen, rejected, weight))
        randomizer.shuffle(pairs)
        output.extend(pairs[: config.maximum_pairs_per_group])
    if not output:
        raise ValueError("training data produced no preference pairs")
    return output


def train_document_ranker(
    examples: Sequence[DocumentRankExample],
    *,
    training_manifest_sha256: str,
    revision: str,
    feature_config: DocumentFeatureConfig | None = None,
    training_config: DocumentRankTrainingConfig | None = None,
) -> DocumentLinearRanker:
    if not examples:
        raise ValueError("document ranker training requires examples")
    if len({(row.group_id, row.candidate_id) for row in examples}) != len(examples):
        raise ValueError("document rank example IDs must be unique within groups")
    if not _is_sha256(training_manifest_sha256):
        raise ValueError("training_manifest_sha256 must be a SHA-256 value")
    if not revision:
        raise ValueError("document ranker revision is required")
    feature_config = feature_config or DocumentFeatureConfig()
    training_config = training_config or DocumentRankTrainingConfig()
    extractor = HashedDocumentFeatureExtractor(feature_config)
    rows = tuple(
        _TrainingRow(
            example=example,
            features=extractor.extract(example.rank_input),
            objective=_objective(example, training_config),
        )
        for example in examples
    )
    means, scales = _dense_statistics(rows)
    pairs = _pairs(rows, training_config)
    sparse_weights: dict[int, float] = defaultdict(float)
    dense_weights = [0.0] * len(DENSE_FEATURE_NAMES)
    epoch_losses = []
    randomizer = random.Random(training_config.random_seed)
    for epoch in range(training_config.epochs):
        randomizer.shuffle(pairs)
        learning_rate = training_config.learning_rate / math.sqrt(epoch + 1)
        loss_sum = 0.0
        weight_sum = 0.0
        for chosen, rejected, pair_weight in pairs:
            sparse_difference, dense_difference = _difference(
                chosen,
                rejected,
                means,
                scales,
            )
            margin = sum(
                sparse_weights[index] * value for index, value in sparse_difference.items()
            ) + sum(
                weight * value
                for weight, value in zip(dense_weights, dense_difference, strict=True)
            )
            probability = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, -margin))))
            gradient = pair_weight * (1.0 - probability)
            loss_sum += pair_weight * math.log1p(math.exp(max(-60.0, min(60.0, -margin))))
            weight_sum += pair_weight
            for index, value in sparse_difference.items():
                sparse_weights[index] += learning_rate * (
                    gradient * value - training_config.l2 * sparse_weights[index]
                )
            for index, value in enumerate(dense_difference):
                dense_weights[index] += learning_rate * (
                    gradient * value - training_config.l2 * dense_weights[index]
                )
        epoch_losses.append(loss_sum / max(weight_sum, 1e-12))
    correct = 0
    for chosen, rejected, _weight in pairs:
        sparse_difference, dense_difference = _difference(chosen, rejected, means, scales)
        margin = sum(
            sparse_weights[index] * value for index, value in sparse_difference.items()
        ) + sum(
            weight * value
            for weight, value in zip(dense_weights, dense_difference, strict=True)
        )
        correct += margin > 0.0
    return DocumentLinearRanker(
        feature_config=feature_config,
        sparse_weights=tuple(sorted(sparse_weights.items())),
        dense_means=means,
        dense_scales=scales,
        dense_weights=tuple(dense_weights),
        bias=0.0,
        training_config_digest=training_config.digest,
        training_manifest_sha256=training_manifest_sha256,
        training_example_digest=sha256_json([example.digest for example in examples]),
        epoch_losses=tuple(epoch_losses),
        pairwise_accuracy=correct / len(pairs),
        revision=revision,
    )


def fit_document_ranker_calibration(
    model: DocumentLinearRanker,
    examples: Sequence[DocumentRankExample],
    *,
    calibration_manifest_sha256: str,
    revision: str,
) -> DocumentRankerCalibration:
    if not examples:
        raise ValueError("document ranker calibration requires examples")
    scores = [model.score_input(example.rank_input) for example in examples]
    center = statistics.median(scores)
    absolute = [abs(value - center) for value in scores]
    scale = max(1e-3, 1.4826 * statistics.median(absolute))
    return DocumentRankerCalibration(
        center=center,
        scale=scale,
        calibration_manifest_sha256=calibration_manifest_sha256,
        revision=revision,
    )


def pairwise_accuracy(
    model: DocumentLinearRanker,
    examples: Sequence[DocumentRankExample],
    config: DocumentRankTrainingConfig | None = None,
) -> float:
    config = config or DocumentRankTrainingConfig()
    extractor = HashedDocumentFeatureExtractor(model.feature_config)
    rows = tuple(
        _TrainingRow(
            example=example,
            features=extractor.extract(example.rank_input),
            objective=_objective(example, config),
        )
        for example in examples
    )
    pairs = _pairs(rows, config)
    correct = 0
    for chosen, rejected, _ in pairs:
        correct += model.raw_score(chosen.features) > model.raw_score(rejected.features)
    return correct / len(pairs)


def group_top1_accuracy(
    model: DocumentLinearRanker,
    examples: Sequence[DocumentRankExample],
    config: DocumentRankTrainingConfig | None = None,
) -> float:
    config = config or DocumentRankTrainingConfig()
    by_group: dict[str, list[DocumentRankExample]] = defaultdict(list)
    for example in examples:
        by_group[example.group_id].append(example)
    correct = 0
    counted = 0
    for group in by_group.values():
        if len(group) < 2:
            continue
        predicted = max(group, key=lambda row: (model.score_input(row.rank_input), row.candidate_id))
        best_objective = min(_objective(row, config) for row in group)
        correct += _objective(predicted, config) <= best_objective + config.minimum_objective_gap
        counted += 1
    if not counted:
        raise ValueError("test examples contain no multi-candidate groups")
    return correct / counted


class DocumentRankerGlobalScorer:
    """Use a frozen document ranker as a complete-path ``GlobalSequenceScorer``."""

    def __init__(self, artifact: DocumentRankerArtifact) -> None:
        self.artifact = artifact
        self.source = f"document-linear-ranker:{artifact.model.revision}"
        self.profile_digest = artifact.digest

    @staticmethod
    def _input(path: Sequence[LatticeArc], context: DocumentContext) -> DocumentRankInput:
        text = "".join(arc.text for arc in path)
        metadata: Mapping[str, object] = path[0].metadata if len(path) == 1 else {}
        window_count = int(metadata.get("windowCount", 1))
        return DocumentRankInput(
            text=text,
            left_context=context.left_context,
            right_context=context.right_context,
            topic_summary=context.topic_summary,
            entity_ids=context.entity_ids,
            local_score=float(metadata.get("localScore", 0.0)),
            overlap_score=float(metadata.get("overlapScore", 0.0)),
            mean_audio_support=float(metadata.get("meanAudioSupport", 0.0)),
            changed_window_count=int(metadata.get("changedWindowCount", 0)),
            generated_window_count=int(metadata.get("generatedWindowCount", 0)),
            ambiguous_overlap_count=int(metadata.get("ambiguousOverlapCount", 0)),
            window_count=window_count,
            retained_path=bool(metadata.get("retainedPath", False)),
            metadata={"pathMetadataDigest": sha256_json(dict(metadata))},
        )

    def score(self, path: Sequence[LatticeArc], *, context: DocumentContext) -> GlobalPathScore:
        return self.score_many((path,), context=context)[0]

    def score_many(
        self,
        paths: Sequence[Sequence[LatticeArc]],
        *,
        context: DocumentContext,
    ) -> tuple[GlobalPathScore, ...]:
        return tuple(
            GlobalPathScore(
                value=self.artifact.calibration.transform(
                    self.artifact.model.score_input(self._input(path, context))
                ),
                source=self.source,
                profile_digest=self.profile_digest,
                path_digest=path_digest(path),
                context_digest=context.digest,
            )
            for path in paths
        )


def manifest_sha256(path: str | Path) -> str:
    return _digest_file(path)
