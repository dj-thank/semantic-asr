from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

from .score_types import EvidenceScore, ScoreSemantics
from .sequence_scorers import TextCandidate

Tokenization = Literal["character", "whitespace", "custom"]
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
        if self.order < 1 or len(self.counts) != self.order or len(self.context_totals) != self.order:
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
                return (count + self.add_k) / (
                    total + self.add_k * vocabulary_size
                )
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

    def __init__(self, model: CountNGramLanguageModel, tokenizer: Tokenizer = character_tokenize) -> None:
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
