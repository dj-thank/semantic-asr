from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .contracts import CandidateEvidence, canonical_json
from .japanese import mora_sequence, optional_reading

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

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("n-gram order must be positive")
        if self.alpha <= 0 or not math.isfinite(self.alpha):
            raise ValueError("n-gram alpha must be finite and positive")
        if self.mode not in {"character", "mora", "subword", "whitespace"}:
            raise ValueError("unknown n-gram tokenization mode")
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
        return {
            "schemaVersion": "ngram-v1",
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

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> NGramLanguageModel:
        if row.get("schemaVersion") != "ngram-v1":
            raise ValueError("unsupported n-gram model schema")
        model = cls(
            order=int(row["order"]),
            mode=str(row["mode"]),
            alpha=float(row["alpha"]),
            lowercase_ascii=bool(row.get("lowercaseAscii", True)),
        )
        model.document_count = int(row.get("documentCount", 0))
        model.token_count = int(row.get("tokenCount", 0))
        model.vocabulary = {str(value) for value in row.get("vocabulary", [])}
        model.counts = {
            int(n): Counter(
                {tuple(str(token) for token in key): int(value) for key, value in rows}
            )
            for n, rows in dict(row.get("counts", {})).items()
        }
        model.context_counts = {
            int(n): Counter(
                {tuple(str(token) for token in key): int(value) for key, value in rows}
            )
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
        self.name = "ngram:" + hashlib.sha256(
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
                item.weight * item.model.score(
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
            source = candidate.reading if self.mode == "mora" and candidate.reading else candidate.text
            tokens = tokenize_ngram_text(source, self.mode)
            if not tokens:
                output[candidate.candidate_id] = -math.inf
                continue
            score = float(self.model.score(" ".join(tokens), bos=self.bos, eos=self.eos))
            output[candidate.candidate_id] = score / max(1, len(tokens))
        return output
