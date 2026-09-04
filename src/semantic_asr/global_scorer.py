"""Complete-path context-scorer contracts for ASR deliberation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .contracts import canonical_json
from .deliberation_evidence import _is_sha256, _strict_float
from .deliberation_lattice import DocumentContext, LatticeArc, path_digest


@dataclass(frozen=True, slots=True)
class GlobalPathScore:
    """Bounded rank preference from a model that reads a complete path and context."""

    value: float
    source: str
    profile_digest: str
    path_digest: str
    context_digest: str

    def __post_init__(self) -> None:
        value = _strict_float(self.value, name="global path preference")
        if not -1.0 <= value <= 1.0:
            raise ValueError("global path preference must be in [-1, 1]")
        if not self.source:
            raise ValueError("global path score source is required")
        for digest in (self.profile_digest, self.path_digest, self.context_digest):
            if not _is_sha256(digest):
                raise ValueError("global path score digests must be SHA-256 values")
        object.__setattr__(self, "value", value)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "value": self.value,
                    "source": self.source,
                    "profileDigest": self.profile_digest,
                    "pathDigest": self.path_digest,
                    "contextDigest": self.context_digest,
                }
            ).encode("utf-8")
        ).hexdigest()


class GlobalSequenceScorer(Protocol):
    """Rank-only seam for a bidirectional Transformer/LLM deliberation encoder."""

    def score(
        self,
        path: Sequence[LatticeArc],
        *,
        context: DocumentContext,
    ) -> GlobalPathScore: ...


class GlobalBatchSequenceScorer(GlobalSequenceScorer, Protocol):
    """Optional batched form used to avoid one model invocation per complete path."""

    def score_many(
        self,
        paths: Sequence[Sequence[LatticeArc]],
        *,
        context: DocumentContext,
    ) -> tuple[GlobalPathScore, ...]: ...


class CallableGlobalSequenceScorer:
    """Small adapter useful for deterministic baselines and external model integrations."""

    def __init__(
        self,
        function: Callable[[Sequence[LatticeArc], DocumentContext], float],
        *,
        source: str,
        profile_digest: str,
    ) -> None:
        if not source or not _is_sha256(profile_digest):
            raise ValueError("global scorer requires source and a frozen profile digest")
        self.function = function
        self.source = source
        self.profile_digest = profile_digest

    def score(
        self,
        path: Sequence[LatticeArc],
        *,
        context: DocumentContext,
    ) -> GlobalPathScore:
        value = _strict_float(self.function(path, context), name="global scorer output")
        return GlobalPathScore(
            value=value,
            source=self.source,
            profile_digest=self.profile_digest,
            path_digest=path_digest(path),
            context_digest=context.digest,
        )

    def score_many(
        self,
        paths: Sequence[Sequence[LatticeArc]],
        *,
        context: DocumentContext,
    ) -> tuple[GlobalPathScore, ...]:
        return tuple(self.score(path, context=context) for path in paths)


def frozen_profile_digest(name: str, revision: str, payload: Mapping[str, object]) -> str:
    """Create a stable digest for an external context scorer or normalization profile."""

    if not name or not revision:
        raise ValueError("profile name and revision are required")
    canonical = canonical_json({"name": name, "revision": revision, "payload": dict(payload)})
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
