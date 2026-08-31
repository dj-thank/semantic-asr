from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import Any, Literal

from .contracts import CandidateEvidence

EquivalencePolicy = Literal["exact", "nfkc", "nfkc-whitespace"]


def _finite(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(_finite(value, name="log likelihood") for value in values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def equivalence_key(text: str, policy: EquivalencePolicy = "exact") -> str:
    if not text:
        raise ValueError("candidate text must not be empty")
    if policy == "exact":
        return text
    value = unicodedata.normalize("NFKC", text)
    if policy == "nfkc":
        return value
    if policy == "nfkc-whitespace":
        return " ".join(value.split())
    raise ValueError(f"unsupported equivalence policy: {policy}")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidatePath:
    """One decoder path before surface-text deduplication."""

    path_id: str
    text: str
    cumulative_log_likelihood: float
    token_ids: tuple[int, ...] = ()
    source: str = "unknown"
    model: str | None = None
    model_revision: str | None = None
    normalized_score: float | None = None
    rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path_id:
            raise ValueError("path_id is required")
        if not self.text:
            raise ValueError("path text is required")
        _finite(self.cumulative_log_likelihood, name="cumulative_log_likelihood")
        if self.normalized_score is not None:
            _finite(self.normalized_score, name="normalized_score")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank is one-based")

    @property
    def provenance_digest(self) -> str:
        return _digest(
            {
                "pathId": self.path_id,
                "text": self.text,
                "cumulativeLogLikelihood": self.cumulative_log_likelihood,
                "tokenIds": self.token_ids,
                "source": self.source,
                "model": self.model,
                "revision": self.model_revision,
                "normalizedScore": self.normalized_score,
                "rank": self.rank,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class SurfaceCandidate:
    candidate_id: str
    text: str
    equivalence_key: str
    paths: tuple[CandidatePath, ...]
    aggregate_log_likelihood: float
    best_path_log_likelihood: float
    source_support: tuple[str, ...]
    surface_forms: tuple[str, ...]
    tokenizations: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text or not self.paths:
            raise ValueError("surface candidate requires ID, text and paths")
        if any(path.text not in self.surface_forms for path in self.paths):
            raise ValueError("all path texts must be retained as surface forms")
        expected = logsumexp([path.cumulative_log_likelihood for path in self.paths])
        if not math.isclose(expected, self.aggregate_log_likelihood, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("aggregate_log_likelihood does not match path mass")

    @property
    def path_count(self) -> int:
        return len(self.paths)

    @property
    def path_mass_bonus(self) -> float:
        """Log-mass beyond the best path; zero for a single path."""

        return self.aggregate_log_likelihood - self.best_path_log_likelihood

    @classmethod
    def from_paths(
        cls,
        paths: Sequence[CandidatePath],
        *,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> SurfaceCandidate:
        if not paths:
            raise ValueError("at least one path is required")
        ordered = tuple(
            sorted(
                paths,
                key=lambda path: (
                    -path.cumulative_log_likelihood,
                    path.rank if path.rank is not None else 2**31,
                    path.path_id,
                ),
            )
        )
        best = ordered[0]
        candidate_id = "surface-" + _digest(
            {
                "key": key,
                "paths": [path.provenance_digest for path in ordered],
            }
        )[:16]
        surface_forms = tuple(dict.fromkeys(path.text for path in ordered))
        tokenizations = tuple(dict.fromkeys(path.token_ids for path in ordered))
        sources = tuple(sorted({path.source for path in ordered}))
        return cls(
            candidate_id=candidate_id,
            text=best.text,
            equivalence_key=key,
            paths=ordered,
            aggregate_log_likelihood=logsumexp(
                [path.cumulative_log_likelihood for path in ordered]
            ),
            best_path_log_likelihood=best.cumulative_log_likelihood,
            source_support=sources,
            surface_forms=surface_forms,
            tokenizations=tokenizations,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class CandidateSetDiagnostics:
    path_count: int
    surface_count: int
    unique_surface_ratio: float
    normalized_path_entropy: float
    mean_pairwise_character_distance: float
    source_count: int


@dataclass(frozen=True, slots=True)
class CandidatePool:
    candidates: tuple[SurfaceCandidate, ...] = ()
    equivalence_policy: EquivalencePolicy = "exact"

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[CandidatePath],
        *,
        policy: EquivalencePolicy = "exact",
        key_fn: Callable[[str], str] | None = None,
    ) -> CandidatePool:
        grouped: dict[str, list[CandidatePath]] = {}
        seen_path_ids: set[str] = set()
        for path in paths:
            if path.path_id in seen_path_ids:
                raise ValueError(f"duplicate path ID: {path.path_id}")
            seen_path_ids.add(path.path_id)
            key = key_fn(path.text) if key_fn is not None else equivalence_key(path.text, policy)
            grouped.setdefault(key, []).append(path)
        if not grouped:
            raise ValueError("candidate pool requires at least one path")
        candidates = tuple(
            sorted(
                (
                    SurfaceCandidate.from_paths(grouped[key], key=key)
                    for key in sorted(grouped)
                ),
                key=lambda candidate: (
                    -candidate.aggregate_log_likelihood,
                    candidate.candidate_id,
                ),
            )
        )
        return cls(candidates=candidates, equivalence_policy=policy)

    @property
    def paths(self) -> tuple[CandidatePath, ...]:
        return tuple(path for candidate in self.candidates for path in candidate.paths)

    def with_paths(self, paths: Iterable[CandidatePath]) -> CandidatePool:
        return CandidatePool.from_paths(
            (*self.paths, *tuple(paths)),
            policy=self.equivalence_policy,
        )

    def posterior(self, *, temperature: float = 1.0) -> dict[str, float]:
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be finite and positive")
        scaled = [candidate.aggregate_log_likelihood / temperature for candidate in self.candidates]
        normalizer = logsumexp(scaled)
        return {
            candidate.candidate_id: math.exp(score - normalizer)
            for candidate, score in zip(self.candidates, scaled, strict=True)
        }

    def top_k(self, count: int) -> CandidatePool:
        if count < 1:
            raise ValueError("count must be positive")
        return replace(self, candidates=self.candidates[:count])

    def diagnostics(self) -> CandidateSetDiagnostics:
        path_count = len(self.paths)
        surface_count = len(self.candidates)
        path_scores = [path.cumulative_log_likelihood for path in self.paths]
        normalizer = logsumexp(path_scores)
        probabilities = [math.exp(score - normalizer) for score in path_scores]
        entropy = -sum(probability * math.log(probability + 1e-12) for probability in probabilities)
        normalized_entropy = (
            entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        )
        distances = [
            _normalized_edit_distance(left.text, right.text)
            for index, left in enumerate(self.candidates)
            for right in self.candidates[index + 1 :]
        ]
        return CandidateSetDiagnostics(
            path_count=path_count,
            surface_count=surface_count,
            unique_surface_ratio=surface_count / path_count,
            normalized_path_entropy=normalized_entropy,
            mean_pairwise_character_distance=fmean(distances) if distances else 0.0,
            source_count=len({path.source for path in self.paths}),
        )


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def _normalized_edit_distance(left: str, right: str) -> float:
    denominator = max(1, len(left), len(right))
    return _edit_distance(tuple(left), tuple(right)) / denominator


def path_from_candidate_evidence(
    candidate: CandidateEvidence,
    *,
    cumulative_log_likelihood: float | None = None,
    path_id: str | None = None,
) -> CandidatePath:
    """Convert legacy evidence without pretending a normalized score is path mass.

    The caller should pass the decoder's cumulative log likelihood. As a migration
    fallback, metadata key ``cumulativeLogLikelihood`` is accepted. No value is
    reconstructed from a rank or arbitrary confidence.
    """

    value = cumulative_log_likelihood
    if value is None:
        raw = candidate.metadata.get("cumulativeLogLikelihood")
        if raw is not None:
            value = float(raw)
    if value is None:
        raise ValueError(
            "cumulative log likelihood is required; normalized scores cannot be "
            "silently converted into path mass"
        )
    return CandidatePath(
        path_id=path_id or candidate.candidate_id,
        text=candidate.text,
        cumulative_log_likelihood=float(value),
        token_ids=candidate.token_ids,
        source=candidate.evidence_source,
        model=str(candidate.metadata.get("model") or "") or None,
        model_revision=str(candidate.metadata.get("modelRevision") or "") or None,
        normalized_score=candidate.sequence_score,
        rank=candidate.rank,
        metadata={
            **candidate.metadata,
            "legacyCandidateId": candidate.candidate_id,
            "legacyAverageLogProbability": candidate.avg_logprob,
        },
    )
