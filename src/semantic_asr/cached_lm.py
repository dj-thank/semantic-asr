from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CachedProbability:
    context_order: int
    context_digest: str
    target_token_id: int
    log_probability: float
    teacher: str
    teacher_revision: str | None = None
    sample_count: int = 1

    def __post_init__(self) -> None:
        if self.context_order < 0:
            raise ValueError("context_order must be non-negative")
        if len(self.context_digest) != 64:
            raise ValueError("context_digest must be SHA-256 hex")
        if self.target_token_id < 0:
            raise ValueError("target_token_id must be non-negative")
        if not math.isfinite(self.log_probability) or self.log_probability > 0:
            raise ValueError("log_probability must be finite and <= 0")
        if not self.teacher:
            raise ValueError("teacher name is required")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")


@dataclass(frozen=True, slots=True)
class CacheLookup:
    log_probability: float
    matched_order: int
    exact: bool
    teacher: str | None
    backoff_steps: int


@dataclass(frozen=True, slots=True)
class SequenceCacheScore:
    total_log_probability: float
    average_log_probability: float
    matched_tokens: int
    missing_tokens: int
    mean_matched_order: float
    exact_match_rate: float


class HashedLMProbabilityCache:
    """Privacy-conscious cache for offline-teacher next-token probabilities.

    Contexts are stored only as keyed SHA-256 digests. The cache supports
    longest-suffix retrieval with an explicit backoff penalty. It is a
    training-free inference component; cached teacher values are never labelled
    as acoustic evidence.
    """

    schema_version = "1.0.0"

    def __init__(
        self,
        *,
        key: bytes,
        maximum_context: int = 8,
        backoff_penalty: float = 0.35,
        missing_log_probability: float = -12.0,
    ) -> None:
        if len(key) < 16:
            raise ValueError("cache key must contain at least 16 bytes")
        if maximum_context < 0:
            raise ValueError("maximum_context must be non-negative")
        if backoff_penalty < 0:
            raise ValueError("backoff_penalty must be non-negative")
        if not math.isfinite(missing_log_probability) or missing_log_probability > 0:
            raise ValueError("missing_log_probability must be finite and <= 0")
        self._key = bytes(key)
        self.maximum_context = int(maximum_context)
        self.backoff_penalty = float(backoff_penalty)
        self.missing_log_probability = float(missing_log_probability)
        self._entries: dict[tuple[int, str, int], CachedProbability] = {}

    def _digest(self, context: Sequence[int]) -> str:
        payload = ",".join(str(int(token)) for token in context).encode("ascii")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def put(
        self,
        context: Sequence[int],
        target_token_id: int,
        probability: float,
        *,
        teacher: str,
        teacher_revision: str | None = None,
        sample_count: int = 1,
    ) -> CachedProbability:
        if not 0 < probability <= 1 or not math.isfinite(probability):
            raise ValueError("probability must be finite in (0, 1]")
        suffix = tuple(int(token) for token in context[-self.maximum_context :])
        entry = CachedProbability(
            context_order=len(suffix),
            context_digest=self._digest(suffix),
            target_token_id=int(target_token_id),
            log_probability=math.log(float(probability)),
            teacher=teacher,
            teacher_revision=teacher_revision,
            sample_count=int(sample_count),
        )
        self._entries[(entry.context_order, entry.context_digest, entry.target_token_id)] = entry
        return entry

    def put_log_probability(
        self,
        context: Sequence[int],
        target_token_id: int,
        log_probability: float,
        *,
        teacher: str,
        teacher_revision: str | None = None,
        sample_count: int = 1,
    ) -> CachedProbability:
        if not math.isfinite(log_probability) or log_probability > 0:
            raise ValueError("log_probability must be finite and <= 0")
        suffix = tuple(int(token) for token in context[-self.maximum_context :])
        entry = CachedProbability(
            context_order=len(suffix),
            context_digest=self._digest(suffix),
            target_token_id=int(target_token_id),
            log_probability=float(log_probability),
            teacher=teacher,
            teacher_revision=teacher_revision,
            sample_count=int(sample_count),
        )
        self._entries[(entry.context_order, entry.context_digest, entry.target_token_id)] = entry
        return entry

    def lookup(
        self,
        context: Sequence[int],
        target_token_id: int,
    ) -> CacheLookup:
        maximum = min(self.maximum_context, len(context))
        for order in range(maximum, -1, -1):
            suffix = tuple(int(token) for token in context[-order:]) if order else ()
            digest = self._digest(suffix)
            entry = self._entries.get((order, digest, int(target_token_id)))
            if entry is None:
                continue
            backoff_steps = maximum - order
            return CacheLookup(
                log_probability=entry.log_probability - self.backoff_penalty * backoff_steps,
                matched_order=order,
                exact=backoff_steps == 0,
                teacher=entry.teacher,
                backoff_steps=backoff_steps,
            )
        return CacheLookup(
            log_probability=self.missing_log_probability,
            matched_order=0,
            exact=False,
            teacher=None,
            backoff_steps=maximum + 1,
        )

    def score(self, token_ids: Sequence[int]) -> SequenceCacheScore:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        total = 0.0
        matched = 0
        exact = 0
        orders: list[int] = []
        for index, target in enumerate(token_ids):
            lookup = self.lookup(token_ids[:index], int(target))
            total += lookup.log_probability
            if lookup.teacher is not None:
                matched += 1
                exact += lookup.exact
                orders.append(lookup.matched_order)
        return SequenceCacheScore(
            total_log_probability=total,
            average_log_probability=total / len(token_ids),
            matched_tokens=matched,
            missing_tokens=len(token_ids) - matched,
            mean_matched_order=sum(orders) / len(orders) if orders else 0.0,
            exact_match_rate=exact / len(token_ids),
        )

    def export(self, path: str | Path) -> None:
        payload = {
            "schemaVersion": self.schema_version,
            "maximumContext": self.maximum_context,
            "backoffPenalty": self.backoff_penalty,
            "missingLogProbability": self.missing_log_probability,
            "entries": [
                asdict(entry)
                for entry in sorted(
                    self._entries.values(),
                    key=lambda row: (
                        row.context_order,
                        row.context_digest,
                        row.target_token_id,
                    ),
                )
            ],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, key: bytes) -> HashedLMProbabilityCache:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schemaVersion") != cls.schema_version:
            raise ValueError("unsupported probability-cache schema")
        cache = cls(
            key=key,
            maximum_context=int(payload["maximumContext"]),
            backoff_penalty=float(payload["backoffPenalty"]),
            missing_log_probability=float(payload["missingLogProbability"]),
        )
        for raw in payload.get("entries", []):
            entry = CachedProbability(**raw)
            cache._entries[(entry.context_order, entry.context_digest, entry.target_token_id)] = (
                entry
            )
        return cache

    def __len__(self) -> int:
        return len(self._entries)


def import_teacher_rows(
    cache: HashedLMProbabilityCache,
    rows: Iterable[dict[str, object]],
    *,
    teacher: str,
    teacher_revision: str | None = None,
) -> int:
    count = 0
    for row in rows:
        context = row.get("contextTokenIds")
        target = row.get("targetTokenId")
        probability = row.get("probability")
        if not isinstance(context, list) or target is None or probability is None:
            raise ValueError("teacher row requires contextTokenIds, targetTokenId, probability")
        cache.put(
            [int(token) for token in context],
            int(target),
            float(probability),
            teacher=teacher,
            teacher_revision=teacher_revision,
            sample_count=int(row.get("sampleCount", 1)),
        )
        count += 1
    return count
