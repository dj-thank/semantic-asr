from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

from .contracts import CandidateEvidence, canonical_json
from .japanese import mora_sequence, optional_reading
from .score_types import EvidenceScore, ScoreSemantics
from .sequence_scorers import TextCandidate

TokenizationMode = Literal["character", "mora", "subword", "whitespace"]
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[+._/#-][A-Za-z0-9]+)*|\d+(?:[.,:/-]\d+)*|"
    r"[\u3040-\u30ffー]+|[\u3400-\u9fff\uf900-\ufaff]+|[^\s]"
)


def tokenize_ngram_text(text: str, mode: TokenizationMode) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not normalized:
        return []
    if mode == "character":
        return [character for character in normalized if not character.isspace()]
    if mode == "whitespace":
        return normalized.split()
    if mode == "mora":
        reading = optional_reading(normalized) or normalized
        mora = mora_sequence(reading)
        if mora:
            return mora
        return [f"surface:{character}" for character in normalized if not character.isspace()]
    if mode == "subword":
        units = _TOKEN_PATTERN.findall(normalized)
        output: list[str] = []
        for unit in units:
            if re.fullmatch(r"[\u3040-\u30ffー\u3400-\u9fff\uf900-\ufaff]+", unit):
                characters = list(unit)
                if len(characters) == 1:
                    output.append(characters[0])
                else:
                    output.extend(
                        characters[index] + characters[index + 1]
                        for index in range(len(characters) - 1)
                    )
            else:
                output.append(unit.lower())
        return output
    raise ValueError(f"unknown n-gram tokenization mode: {mode}")


@dataclass(frozen=True, slots=True)
class NGramScore:
    total_log_probability: float
    average_log_probability: float
    token_count: int
    unknown_token_count: int
    order: int
    mode: TokenizationMode


@dataclass(slots=True)
class NGramLanguageModel:
    order: int = 5
    mode: TokenizationMode = "character"
    alpha: float = 0.1
    lowercase_ascii: bool = True
    counts: dict[int, Counter[tuple[str, ...]]] = field(default_factory=dict)
    context_counts: dict[int, Counter[tuple[str, ...]]] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)
    document_count: int = 0
    token_count: int = 0
    source_sha256: str | None = None
    source_revision: str | None = None
    schema_version: str = "ngram-v2"

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("n-gram order must be positive")
        if self.alpha <= 0 or not math.isfinite(self.alpha):
            raise ValueError("n-gram alpha must be finite and positive")
        if self.mode not in {"character", "mora", "subword", "whitespace"}:
            raise ValueError("unknown n-gram tokenization mode")
        if self.schema_version not in {"ngram-v1", "ngram-v2"}:
            raise ValueError("unsupported n-gram model schema")
        if self.source_sha256 is not None and (
            len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256.lower())
        ):
            raise ValueError("n-gram source_sha256 must be a 64-character hexadecimal digest")
        if self.source_revision is not None and not self.source_revision.strip():
            raise ValueError("n-gram source_revision must not be empty")
        if self.source_revision is not None and self.source_sha256 is None:
            raise ValueError("n-gram source_revision requires source_sha256")
        if self.schema_version == "ngram-v1" and (
            self.source_sha256 is not None or self.source_revision is not None
        ):
            raise ValueError("ngram-v1 cannot carry source provenance")
        for n in range(1, self.order + 1):
            self.counts.setdefault(n, Counter())
            self.context_counts.setdefault(n, Counter())

    def _tokens(self, text: str) -> list[str]:
        tokens = tokenize_ngram_text(text, self.mode)
        if self.lowercase_ascii:
            tokens = [token.lower() if token.isascii() else token for token in tokens]
        return tokens

    def fit(self, texts: Iterable[str]) -> NGramLanguageModel:
        seen_documents = 0
        for text in texts:
            tokens = self._tokens(text)
            if not tokens:
                continue
            seen_documents += 1
            self.token_count += len(tokens)
            self.vocabulary.update(tokens)
            padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
            self.vocabulary.add("</s>")
            for index in range(self.order - 1, len(padded)):
                for n in range(1, self.order + 1):
                    start = max(0, index - n + 1)
                    ngram = tuple(padded[start : index + 1])
                    if len(ngram) != n:
                        continue
                    context = ngram[:-1]
                    self.counts[n][ngram] += 1
                    self.context_counts[n][context] += 1
        self.document_count += seen_documents
        if seen_documents == 0:
            raise ValueError("n-gram corpus contains no tokens")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict()).encode("utf-8")).hexdigest()

    def _conditional_log_probability(
        self, context: Sequence[str], target: str
    ) -> tuple[float, bool]:
        maximum_order = min(self.order, len(context) + 1)
        vocabulary_size = max(1, len(self.vocabulary) + 1)
        unknown = target not in self.vocabulary
        for n in range(maximum_order, 0, -1):
            suffix = tuple(context[-(n - 1) :]) if n > 1 else ()
            ngram = suffix + (target,)
            context_count = self.context_counts[n].get(suffix, 0)
            ngram_count = self.counts[n].get(ngram, 0)
            if context_count > 0 or n == 1:
                probability = (ngram_count + self.alpha) / (
                    context_count + self.alpha * vocabulary_size
                )
                backoff_steps = maximum_order - n
                return math.log(probability) - 0.12 * backoff_steps, unknown
        raise AssertionError("unigram probability path is unreachable")

    def score(self, text: str) -> NGramScore:
        tokens = self._tokens(text)
        if not tokens:
            raise ValueError("cannot score empty n-gram text")
        context: list[str] = ["<s>"] * (self.order - 1)
        total = 0.0
        unknown = 0
        for target in [*tokens, "</s>"]:
            value, is_unknown = self._conditional_log_probability(context, target)
            total += value
            unknown += int(is_unknown)
            context.append(target)
        count = len(tokens) + 1
        return NGramScore(
            total_log_probability=total,
            average_log_probability=total / count,
            token_count=len(tokens),
            unknown_token_count=unknown,
            order=self.order,
            mode=self.mode,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schemaVersion": self.schema_version,
            "order": self.order,
            "mode": self.mode,
            "alpha": self.alpha,
            "lowercaseAscii": self.lowercase_ascii,
            "documentCount": self.document_count,
            "tokenCount": self.token_count,
            "vocabulary": sorted(self.vocabulary),
            "counts": {
                str(n): [[list(key), value] for key, value in sorted(counter.items())]
                for n, counter in self.counts.items()
            },
            "contextCounts": {
                str(n): [[list(key), value] for key, value in sorted(counter.items())]
                for n, counter in self.context_counts.items()
            },
        }
        if self.schema_version == "ngram-v2":
            payload["sourceSha256"] = self.source_sha256
            payload["sourceRevision"] = self.source_revision
        return payload

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> NGramLanguageModel:
        if row.get("schemaVersion") not in {"ngram-v1", "ngram-v2"}:
            raise ValueError("unsupported n-gram model schema")
        model = cls(
            order=int(row["order"]),
            mode=str(row["mode"]),
            alpha=float(row["alpha"]),
            lowercase_ascii=bool(row.get("lowercaseAscii", True)),
            source_sha256=(
                str(row["sourceSha256"])
                if "sourceSha256" in row and row["sourceSha256"] is not None
                else None
            ),
            source_revision=(
                str(row["sourceRevision"])
                if "sourceRevision" in row and row["sourceRevision"] is not None
                else None
            ),
            schema_version=str(row["schemaVersion"]),
        )
        model.document_count = int(row.get("documentCount", 0))
        model.token_count = int(row.get("tokenCount", 0))
        model.vocabulary = {str(value) for value in row.get("vocabulary", [])}
        model.counts = {
            int(n): Counter({tuple(str(token) for token in key): int(value) for key, value in rows})
            for n, rows in dict(row.get("counts", {})).items()
        }
        model.context_counts = {
            int(n): Counter({tuple(str(token) for token in key): int(value) for key, value in rows})
            for n, rows in dict(row.get("contextCounts", {})).items()
        }
        for n in range(1, model.order + 1):
            model.counts.setdefault(n, Counter())
            model.context_counts.setdefault(n, Counter())
        return model

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NGramLanguageModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class WeightedNGramModel:
    name: str
    model: NGramLanguageModel
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("weighted n-gram model name is required")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError("n-gram weight must be finite and non-negative")


class NGramCandidateRanker:
    def __init__(self, models: Sequence[WeightedNGramModel]) -> None:
        self.models = tuple(models)
        if not self.models or sum(item.weight for item in self.models) <= 0:
            raise ValueError("at least one positive-weight n-gram model is required")
        self.name = (
            "ngram:"
            + hashlib.sha256(
                canonical_json(
                    [
                        {
                            "name": item.name,
                            "weight": item.weight,
                            "digest": item.model.digest,
                        }
                        for item in self.models
                    ]
                ).encode("utf-8")
            ).hexdigest()[:16]
        )

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        del context, consensus, contradiction
        total_weight = sum(item.weight for item in self.models)
        return {
            candidate.candidate_id: sum(
                item.weight
                * item.model.score(
                    candidate.reading
                    if item.model.mode == "mora" and candidate.reading
                    else candidate.text
                ).average_log_probability
                for item in self.models
            )
            / total_weight
            for candidate in candidates
        }


class KenLMCandidateRanker:
    """Optional KenLM backend with explicit tokenization and raw log scores."""

    def __init__(
        self,
        model_path: str,
        *,
        mode: TokenizationMode = "character",
        bos: bool = True,
        eos: bool = True,
    ) -> None:
        try:
            import kenlm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install semantic-asr with the 'kenlm' extra") from exc
        self.model_path = str(model_path)
        self.mode = mode
        self.bos = bool(bos)
        self.eos = bool(eos)
        self.model = kenlm.Model(self.model_path)
        self.name = f"kenlm:{Path(model_path).name}:{mode}"

    def score(
        self,
        candidates: Sequence[CandidateEvidence],
        *,
        context: str = "",
        consensus: str = "",
        contradiction: str = "",
    ) -> Mapping[str, float]:
        del context, consensus, contradiction
        output: dict[str, float] = {}
        for candidate in candidates:
            source = (
                candidate.reading if self.mode == "mora" and candidate.reading else candidate.text
            )
            tokens = tokenize_ngram_text(source, self.mode)
            if not tokens:
                output[candidate.candidate_id] = -math.inf
                continue
            score = float(self.model.score(" ".join(tokens), bos=self.bos, eos=self.eos))
            output[candidate.candidate_id] = score / max(1, len(tokens))
        return output


# Typed dependency-free baseline and explicit score-semantics adapters.

Tokenizer = Callable[[str], Sequence[str]]


def character_tokenize(text: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", str(text or ""))
    return tuple(character for character in value if not character.isspace())


def whitespace_tokenize(text: str) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFKC", str(text or "")).split())


def _logsum(values: Iterable[float]) -> float:
    return sum(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class NGramScoreResult:
    candidate_id: str
    cumulative: EvidenceScore
    average: EvidenceScore
    token_count: int


@dataclass(frozen=True, slots=True)
class CountNGramLanguageModel:
    """Small dependency-free backoff n-gram baseline.

    This model is intentionally simple and reproducible. Large-corpus experiments
    should use KenLM through ``KenLMScorer`` while retaining this implementation for
    unit tests and tiny CPU baselines.
    """

    order: int
    counts: tuple[dict[tuple[str, ...], int], ...]
    context_totals: tuple[dict[tuple[str, ...], int], ...]
    vocabulary: tuple[str, ...]
    add_k: float
    tokenizer_name: str
    corpus_digest: str

    def __post_init__(self) -> None:
        if (
            self.order < 1
            or len(self.counts) != self.order
            or len(self.context_totals) != self.order
        ):
            raise ValueError("count tables must match n-gram order")
        if self.add_k <= 0 or not math.isfinite(self.add_k):
            raise ValueError("add_k must be finite and positive")
        if not self.vocabulary:
            raise ValueError("vocabulary must not be empty")

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        *,
        order: int = 5,
        tokenizer: Tokenizer = character_tokenize,
        tokenizer_name: str = "character-nfkc",
        add_k: float = 0.1,
    ) -> Self:
        if order < 1:
            raise ValueError("order must be positive")
        rows = [str(text) for text in texts if str(text)]
        if not rows:
            raise ValueError("training texts are required")
        count_tables = [Counter() for _ in range(order)]
        context_tables = [Counter() for _ in range(order)]
        vocabulary = {"<s>", "</s>", "<unk>"}
        tokenized_rows: list[tuple[str, ...]] = []
        for text in rows:
            tokens = tuple(str(token) for token in tokenizer(text) if str(token))
            if not tokens:
                continue
            tokenized_rows.append(tokens)
            vocabulary.update(tokens)
            padded = ("<s>",) * (order - 1) + tokens + ("</s>",)
            for position in range(order - 1, len(padded)):
                for current_order in range(1, order + 1):
                    start = position - current_order + 1
                    if start < 0:
                        continue
                    gram = padded[start : position + 1]
                    context = gram[:-1]
                    count_tables[current_order - 1][gram] += 1
                    context_tables[current_order - 1][context] += 1
        if not tokenized_rows:
            raise ValueError("all training rows tokenized to empty sequences")
        corpus_digest = hashlib.sha256(
            json.dumps(
                {
                    "rows": tokenized_rows,
                    "order": order,
                    "tokenizer": tokenizer_name,
                    "addK": add_k,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            order=order,
            counts=tuple(dict(table) for table in count_tables),
            context_totals=tuple(dict(table) for table in context_tables),
            vocabulary=tuple(sorted(vocabulary)),
            add_k=float(add_k),
            tokenizer_name=tokenizer_name,
            corpus_digest=corpus_digest,
        )

    def _probability(self, history: tuple[str, ...], token: str) -> float:
        vocabulary_size = len(self.vocabulary)
        resolved = token if token in self.vocabulary else "<unk>"
        for current_order in range(min(self.order, len(history) + 1), 0, -1):
            context = history[-(current_order - 1) :] if current_order > 1 else ()
            gram = (*context, resolved)
            count = self.counts[current_order - 1].get(gram, 0)
            total = self.context_totals[current_order - 1].get(context, 0)
            if total > 0 or current_order == 1:
                return (count + self.add_k) / (total + self.add_k * vocabulary_size)
        raise AssertionError("unreachable unigram backoff")

    def score_tokens(self, tokens: Sequence[str]) -> tuple[float, int]:
        resolved = tuple(token if token in self.vocabulary else "<unk>" for token in tokens)
        padded_history: list[str] = ["<s>"] * (self.order - 1)
        log_probability = 0.0
        count = 0
        for token in (*resolved, "</s>"):
            probability = self._probability(tuple(padded_history), token)
            log_probability += math.log(max(1e-12, probability))
            padded_history.append(token)
            count += 1
        return log_probability, count


class CountNGramScorer:
    name = "count-ngram-loglikelihood"

    def __init__(
        self, model: CountNGramLanguageModel, tokenizer: Tokenizer = character_tokenize
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def score(self, candidates: list[TextCandidate]) -> list[NGramScoreResult]:
        output: list[NGramScoreResult] = []
        for candidate in candidates:
            tokens = tuple(self.tokenizer(candidate.text))
            cumulative, token_count = self.model.score_tokens(tokens)
            common = {
                "scorer": self.name,
                "model": f"count-{self.model.order}gram",
                "revision": self.model.corpus_digest,
                "runtime": "python-stdlib",
                "metadata": {
                    "tokenizer": self.model.tokenizer_name,
                    "order": self.model.order,
                    "corpusDigest": self.model.corpus_digest,
                    "tokenCount": token_count,
                },
            }
            output.append(
                NGramScoreResult(
                    candidate_id=candidate.candidate_id,
                    cumulative=EvidenceScore.raw(
                        cumulative,
                        semantics=ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
                        **common,
                    ),
                    average=EvidenceScore.raw(
                        cumulative / max(1, token_count),
                        semantics=ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
                        **common,
                    ),
                    token_count=token_count,
                )
            )
        return output


class KenLMScorer:
    """Thin typed wrapper around a KenLM binary.

    Tokenization is explicit because Japanese character, mora, subword and word
    language models are separate experiments. KenLM scores are converted from
    log10 to natural log before entering shared features.
    """

    name = "kenlm-loglikelihood"

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer: Tokenizer = character_tokenize,
        tokenizer_name: str = "character-nfkc",
        bos: bool = True,
        eos: bool = True,
    ) -> None:
        try:
            import kenlm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("KenLM scoring requires the kenlm package") from exc
        self.model_path = model_path
        self.model: Any = kenlm.Model(model_path)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.bos = bool(bos)
        self.eos = bool(eos)

    def score(self, candidates: list[TextCandidate]) -> list[NGramScoreResult]:
        output: list[NGramScoreResult] = []
        for candidate in candidates:
            tokens = tuple(str(token) for token in self.tokenizer(candidate.text) if str(token))
            if not tokens:
                raise ValueError(f"candidate {candidate.candidate_id} has no LM tokens")
            lm_text = " ".join(tokens)
            log10_value = float(self.model.score(lm_text, bos=self.bos, eos=self.eos))
            cumulative = log10_value * math.log(10.0)
            token_count = len(tokens) + int(self.eos)
            common = {
                "scorer": self.name,
                "model": self.model_path,
                "runtime": "kenlm",
                "metadata": {
                    "tokenizer": self.tokenizer_name,
                    "bos": self.bos,
                    "eos": self.eos,
                    "tokenCount": token_count,
                    "sourceLogBase": 10,
                },
            }
            output.append(
                NGramScoreResult(
                    candidate_id=candidate.candidate_id,
                    cumulative=EvidenceScore.raw(
                        cumulative,
                        semantics=ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
                        **common,
                    ),
                    average=EvidenceScore.raw(
                        cumulative / max(1, token_count),
                        semantics=ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
                        **common,
                    ),
                    token_count=token_count,
                )
            )
        return output
