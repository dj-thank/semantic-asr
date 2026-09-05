"""Dependency-free, immutable language baseline for document-context experiments.

This scorer is deliberately a statistical character model rather than an instruction-following
model. It can measure whether ordered or right-context language evidence separates frozen document
candidates without granting text-generation authority or interpreting context as instructions.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from ..document_deliberation import DocumentPathHypothesis
from .protocol import DocumentExperimentArm

_BOS = "\u0002"
_EOS = "\u0003"
_UNK = "\ufffd"


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _strict_order(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n-gram order must be an integer")
    if not 2 <= value <= 12:
        raise ValueError("n-gram order must be in [2, 12]")
    return value


def _deterministic_order(size: int, *, seed: str, case_id: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(size),
            key=lambda index: sha256_json({"seed": seed, "caseId": case_id, "windowIndex": index}),
        )
    )


@dataclass(frozen=True, slots=True)
class CharacterNgramScore:
    cumulative_log_likelihood: float
    token_count: int
    scored_characters: int

    def __post_init__(self) -> None:
        value = _strict_float(
            self.cumulative_log_likelihood,
            name="cumulative_log_likelihood",
        )
        if isinstance(self.token_count, bool) or self.token_count < 1:
            raise ValueError("token_count must be positive")
        if isinstance(self.scored_characters, bool) or self.scored_characters < 0:
            raise ValueError("scored_characters must be non-negative")
        object.__setattr__(self, "cumulative_log_likelihood", value)

    @property
    def average_log_likelihood(self) -> float:
        return self.cumulative_log_likelihood / self.token_count


@dataclass(frozen=True, slots=True)
class FrozenCharacterNgramModel:
    order: int
    alpha: float
    vocabulary: tuple[str, ...]
    rows: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    training_manifest_sha256: str
    revision: str
    reversed_text: bool = False
    schema_version: str = "1"

    def __post_init__(self) -> None:
        _strict_order(self.order)
        alpha = _strict_float(self.alpha, name="alpha")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if not _is_sha256(self.training_manifest_sha256):
            raise ValueError("training_manifest_sha256 must be a SHA-256 value")
        if not self.revision:
            raise ValueError("model revision is required")
        vocabulary = tuple(dict.fromkeys(self.vocabulary))
        if vocabulary != self.vocabulary or any(not token for token in vocabulary):
            raise ValueError("vocabulary must contain unique non-empty symbols")
        if _UNK not in vocabulary or _EOS not in vocabulary:
            raise ValueError("vocabulary must contain UNK and EOS symbols")
        seen_contexts: set[str] = set()
        allowed = set(vocabulary)
        for context, counts in self.rows:
            if len(context) != self.order - 1:
                raise ValueError("n-gram context has the wrong width")
            if context in seen_contexts:
                raise ValueError("duplicate n-gram context row")
            seen_contexts.add(context)
            if not counts:
                raise ValueError("n-gram context rows must not be empty")
            symbols = [symbol for symbol, _ in counts]
            if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
                raise ValueError("n-gram counts must be unique and sorted")
            for symbol, count in counts:
                if symbol not in allowed:
                    raise ValueError("n-gram row contains a symbol outside the vocabulary")
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("n-gram counts must be positive integers")
        object.__setattr__(self, "alpha", alpha)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    @property
    def count_map(self) -> dict[str, dict[str, int]]:
        return {context: dict(counts) for context, counts in self.rows}

    def _symbol(self, value: str) -> str:
        return value if value in self.vocabulary else _UNK

    def score(
        self,
        text: str,
        *,
        prefix: str = "",
        maximum_characters: int | None = None,
    ) -> CharacterNgramScore:
        if maximum_characters is not None:
            if isinstance(maximum_characters, bool) or not isinstance(maximum_characters, int):
                raise TypeError("maximum_characters must be an integer")
            if maximum_characters < 1:
                raise ValueError("maximum_characters must be positive")
        candidate = _normalize(text)
        if self.reversed_text:
            candidate = candidate[::-1]
            prefix = _normalize(prefix)[::-1]
        else:
            prefix = _normalize(prefix)
        if maximum_characters is not None:
            candidate = candidate[:maximum_characters]
        mapped_prefix = "".join(self._symbol(character) for character in prefix)
        context = (_BOS * (self.order - 1) + mapped_prefix)[-(self.order - 1) :]
        count_map = self.count_map
        vocabulary_size = len(self.vocabulary)
        cumulative = 0.0
        scored = 0
        for character in (*candidate, _EOS):
            symbol = self._symbol(character)
            counts = count_map.get(context, {})
            total = sum(counts.values())
            probability = (counts.get(symbol, 0) + self.alpha) / (
                total + self.alpha * vocabulary_size
            )
            cumulative += math.log(probability)
            if character != _EOS:
                scored += 1
            context = (context + symbol)[-(self.order - 1) :]
        return CharacterNgramScore(
            cumulative_log_likelihood=cumulative,
            token_count=scored + 1,
            scored_characters=scored,
        )


@dataclass(frozen=True, slots=True)
class NgramCalibrationSequence:
    text: str
    left_context: str = ""
    right_context: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("calibration sequence text must not be empty")


@dataclass(frozen=True, slots=True)
class NgramScoreNormalization:
    center: float
    scale: float
    calibration_manifest_sha256: str
    sample_count: int
    revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        center = _strict_float(self.center, name="normalization center")
        scale = _strict_float(self.scale, name="normalization scale")
        if scale <= 0.0:
            raise ValueError("normalization scale must be positive")
        if not _is_sha256(self.calibration_manifest_sha256):
            raise ValueError("calibration_manifest_sha256 must be a SHA-256 value")
        if isinstance(self.sample_count, bool) or self.sample_count < 2:
            raise ValueError("normalization requires at least two calibration samples")
        if not self.revision:
            raise ValueError("normalization revision is required")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def transform(self, value: float) -> float:
        numeric = _strict_float(value, name="raw document language score")
        return math.tanh((numeric - self.center) / self.scale)


@dataclass(frozen=True, slots=True)
class DocumentLanguageScore:
    value: float
    raw_average_log_likelihood: float
    forward_average_log_likelihood: float | None
    backward_average_log_likelihood: float | None
    source: str
    profile_digest: str
    path_digest: str
    arm_digest: str
    scored_characters: int
    scorer_calls: int

    def __post_init__(self) -> None:
        value = _strict_float(self.value, name="document language score")
        raw = _strict_float(
            self.raw_average_log_likelihood,
            name="raw_average_log_likelihood",
        )
        if not -1.0 <= value <= 1.0:
            raise ValueError("document language score must be in [-1, 1]")
        for name in (
            "forward_average_log_likelihood",
            "backward_average_log_likelihood",
        ):
            row = getattr(self, name)
            if row is not None:
                object.__setattr__(self, name, _strict_float(row, name=name))
        if not self.source:
            raise ValueError("document language score source is required")
        for digest in (self.profile_digest, self.path_digest, self.arm_digest):
            if not _is_sha256(digest):
                raise ValueError("document language score contains an invalid SHA-256 value")
        if isinstance(self.scored_characters, bool) or self.scored_characters < 0:
            raise ValueError("scored_characters must be non-negative")
        if isinstance(self.scorer_calls, bool) or self.scorer_calls < 0:
            raise ValueError("scorer_calls must be non-negative")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "raw_average_log_likelihood", raw)

    @classmethod
    def neutral(
        cls,
        *,
        path_digest: str,
        arm_digest: str,
    ) -> DocumentLanguageScore:
        return cls(
            value=0.0,
            raw_average_log_likelihood=0.0,
            forward_average_log_likelihood=None,
            backward_average_log_likelihood=None,
            source="acoustic-only",
            profile_digest=sha256_json({"source": "acoustic-only", "revision": "1"}),
            path_digest=path_digest,
            arm_digest=arm_digest,
            scored_characters=0,
            scorer_calls=0,
        )


@dataclass(frozen=True, slots=True)
class BidirectionalCharacterNgramScorer:
    forward: FrozenCharacterNgramModel
    backward: FrozenCharacterNgramModel
    normalization: NgramScoreNormalization
    source: str = "frozen-bidirectional-character-ngram"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.forward.reversed_text:
            raise ValueError("forward model must not be trained on reversed text")
        if not self.backward.reversed_text:
            raise ValueError("backward model must be trained on reversed text")
        if self.forward.order != self.backward.order:
            raise ValueError("forward and backward n-gram orders must match")
        if not self.source:
            raise ValueError("scorer source is required")

    @property
    def profile_digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "source": self.source,
                "forwardDigest": self.forward.digest,
                "backwardDigest": self.backward.digest,
                "normalizationDigest": self.normalization.digest,
            }
        )

    def _render(
        self,
        path: DocumentPathHypothesis,
        arm: DocumentExperimentArm,
        *,
        case_id: str,
    ) -> str:
        if arm.candidate_view == "ordered-document":
            return path.text
        if arm.candidate_view == "shuffled-document":
            order = _deterministic_order(
                len(path.options),
                seed=arm.shuffled_seed,
                case_id=case_id,
            )
            return "\n".join(path.options[index].text for index in order)
        raise ValueError("linguistic scorer cannot render an acoustic-only arm")

    def score_path(
        self,
        path: DocumentPathHypothesis,
        arm: DocumentExperimentArm,
        *,
        case_id: str,
        left_context: str = "",
        right_context: str = "",
        maximum_characters: int,
    ) -> DocumentLanguageScore:
        if arm.direction == "none":
            return DocumentLanguageScore.neutral(
                path_digest=path.digest,
                arm_digest=arm.digest,
            )
        rendered = self._render(path, arm, case_id=case_id)
        forward = self.forward.score(
            rendered,
            prefix=left_context,
            maximum_characters=maximum_characters,
        )
        backward: CharacterNgramScore | None = None
        if arm.direction == "bidirectional":
            backward = self.backward.score(
                rendered,
                prefix=right_context,
                maximum_characters=maximum_characters,
            )
        raw = (
            forward.average_log_likelihood
            if backward is None
            else (forward.average_log_likelihood + backward.average_log_likelihood) / 2.0
        )
        return DocumentLanguageScore(
            value=self.normalization.transform(raw),
            raw_average_log_likelihood=raw,
            forward_average_log_likelihood=forward.average_log_likelihood,
            backward_average_log_likelihood=(
                None if backward is None else backward.average_log_likelihood
            ),
            source=self.source,
            profile_digest=self.profile_digest,
            path_digest=path.digest,
            arm_digest=arm.digest,
            scored_characters=forward.scored_characters
            + (0 if backward is None else backward.scored_characters),
            scorer_calls=1 if backward is None else 2,
        )


def fit_character_ngram_model(
    texts: Iterable[str],
    *,
    order: int,
    alpha: float,
    training_manifest_sha256: str,
    revision: str,
    reversed_text: bool = False,
) -> FrozenCharacterNgramModel:
    order = _strict_order(order)
    alpha = _strict_float(alpha, name="alpha")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if not _is_sha256(training_manifest_sha256):
        raise ValueError("training_manifest_sha256 must be a SHA-256 value")
    rows = tuple(_normalize(text) for text in texts)
    if not rows or any(not text for text in rows):
        raise ValueError("training text must contain non-empty sequences")
    if reversed_text:
        rows = tuple(text[::-1] for text in rows)
    vocabulary = tuple(sorted({_UNK, _EOS, *(character for text in rows for character in text)}))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for text in rows:
        context = _BOS * (order - 1)
        for character in (*text, _EOS):
            symbol = character if character in vocabulary else _UNK
            counts[context][symbol] += 1
            context = (context + symbol)[-(order - 1) :]
    serialized = tuple(
        (context, tuple(sorted(symbol_counts.items())))
        for context, symbol_counts in sorted(counts.items())
    )
    return FrozenCharacterNgramModel(
        order=order,
        alpha=alpha,
        vocabulary=vocabulary,
        rows=serialized,
        training_manifest_sha256=training_manifest_sha256,
        revision=revision,
        reversed_text=reversed_text,
    )


def _raw_bidirectional_score(
    forward: FrozenCharacterNgramModel,
    backward: FrozenCharacterNgramModel,
    row: NgramCalibrationSequence,
) -> float:
    left = forward.score(row.text, prefix=row.left_context)
    right = backward.score(row.text, prefix=row.right_context)
    return (left.average_log_likelihood + right.average_log_likelihood) / 2.0


def fit_ngram_normalization(
    forward: FrozenCharacterNgramModel,
    backward: FrozenCharacterNgramModel,
    rows: Sequence[NgramCalibrationSequence],
    *,
    calibration_manifest_sha256: str,
    revision: str,
) -> NgramScoreNormalization:
    if not _is_sha256(calibration_manifest_sha256):
        raise ValueError("calibration_manifest_sha256 must be a SHA-256 value")
    if len(rows) < 2:
        raise ValueError("normalization requires at least two calibration sequences")
    values = [_raw_bidirectional_score(forward, backward, row) for row in rows]
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / len(values)
    scale = max(math.sqrt(variance), 1e-6)
    return NgramScoreNormalization(
        center=center,
        scale=scale,
        calibration_manifest_sha256=calibration_manifest_sha256,
        sample_count=len(values),
        revision=revision,
    )
