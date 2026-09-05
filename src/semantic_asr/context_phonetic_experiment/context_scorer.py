"""Candidate-bound context scoring for frozen factorial candidate pools."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..deliberation_lattice import DocumentContext, LatticeArc
from ..global_scorer import GlobalPathScore, GlobalSequenceScorer
from .protocol import FrozenContextSnapshot


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    candidate_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text:
            raise ValueError("context candidate requires candidate_id and text")

    @property
    def text_sha256(self) -> str:
        return sha256_json({"text": self.text})

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "candidateId": self.candidate_id,
                "textSha256": self.text_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextCandidateScore:
    candidate_id: str
    candidate_text_sha256: str
    context_digest: str
    value: float
    source: str
    scorer_profile_digest: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source:
            raise ValueError("context score requires candidate_id and source")
        for digest in (
            self.candidate_text_sha256,
            self.context_digest,
            self.scorer_profile_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("context score digests must be SHA-256 values")
        if isinstance(self.value, bool):
            raise TypeError("context score value must be a real number")
        value = float(self.value)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("context score value must be finite and in [-1, 1]")
        object.__setattr__(self, "value", value)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "candidateId": self.candidate_id,
                "candidateTextSha256": self.candidate_text_sha256,
                "contextDigest": self.context_digest,
                "value": self.value,
                "source": self.source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )


class CandidateContextScorer(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def profile_digest(self) -> str: ...

    def score_many(
        self,
        candidates: Sequence[ContextCandidate],
        *,
        context: FrozenContextSnapshot,
    ) -> tuple[ContextCandidateScore, ...]: ...


class CallableCandidateContextScorer:
    """Deterministic dependency-free scorer for tests and registered baselines."""

    def __init__(
        self,
        function: Callable[[ContextCandidate, FrozenContextSnapshot], float],
        *,
        source: str,
        profile_digest: str,
    ) -> None:
        if not source or not _is_sha256(profile_digest):
            raise ValueError("callable context scorer requires source and profile digest")
        self.function = function
        self.source = source
        self.profile_digest = profile_digest

    def score_many(
        self,
        candidates: Sequence[ContextCandidate],
        *,
        context: FrozenContextSnapshot,
    ) -> tuple[ContextCandidateScore, ...]:
        if not candidates:
            raise ValueError("context scorer requires candidates")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("context candidate IDs must be unique")
        return tuple(
            ContextCandidateScore(
                candidate_id=candidate.candidate_id,
                candidate_text_sha256=candidate.text_sha256,
                context_digest=context.digest,
                value=self.function(candidate, context),
                source=self.source,
                scorer_profile_digest=self.profile_digest,
            )
            for candidate in candidates
        )


class GlobalSequenceCandidateContextAdapter:
    """Adapt the complete-path scorer to score each frozen surface under one context."""

    def __init__(
        self,
        scorer: GlobalSequenceScorer,
        *,
        source: str | None = None,
        profile_digest: str | None = None,
    ) -> None:
        resolved_source = source or getattr(scorer, "source", None)
        resolved_profile = profile_digest or getattr(scorer, "profile_digest", None)
        if not isinstance(resolved_source, str) or not resolved_source:
            raise ValueError("global context adapter requires a frozen scorer source")
        if not isinstance(resolved_profile, str) or not _is_sha256(resolved_profile):
            raise ValueError("global context adapter requires a frozen scorer profile digest")
        self.scorer = scorer
        self.source = resolved_source
        self.profile_digest = resolved_profile

    @staticmethod
    def _document_context(context: FrozenContextSnapshot) -> DocumentContext:
        return DocumentContext(
            left_context=context.left_context,
            right_context=context.right_context,
            topic_summary=context.topic_summary,
            entity_ids=context.entity_ids,
            metadata={
                "contextSnapshotDigest": context.digest,
                "contextId": context.context_id,
                "sourceCaseId": context.source_case_id,
                "contextRevision": context.revision,
            },
        )

    @staticmethod
    def _path(candidate: ContextCandidate) -> tuple[LatticeArc, ...]:
        return (
            LatticeArc(
                arc_id=f"context-score:{candidate.candidate_id}",
                span_id=f"context-score:{candidate.candidate_id}",
                text=candidate.text,
                origin="human",
                utilities=(),
                observed_eligible=False,
                metadata={
                    "candidateDigest": candidate.digest,
                    "role": "rank-only-context-candidate",
                },
            ),
        )

    def score_many(
        self,
        candidates: Sequence[ContextCandidate],
        *,
        context: FrozenContextSnapshot,
    ) -> tuple[ContextCandidateScore, ...]:
        if not candidates:
            raise ValueError("global context adapter requires candidates")
        paths = tuple(self._path(candidate) for candidate in candidates)
        document_context = self._document_context(context)
        batched = getattr(self.scorer, "score_many", None)
        if callable(batched):
            rows = tuple(batched(paths, context=document_context))
        else:
            rows = tuple(
                self.scorer.score(path, context=document_context) for path in paths
            )
        if len(rows) != len(candidates):
            raise ValueError("global scorer returned the wrong number of candidate scores")
        output: list[ContextCandidateScore] = []
        seen: set[str] = set()
        for candidate, path, row in zip(candidates, paths, rows, strict=True):
            if not isinstance(row, GlobalPathScore):
                raise TypeError("global scorer must return GlobalPathScore values")
            if row.path_digest != sha256_json(
                [
                    {
                        "arcId": path[0].arc_id,
                        "spanId": path[0].span_id,
                        "text": path[0].text,
                        "arcDigest": path[0].digest,
                    }
                ]
            ):
                raise ValueError("global score is bound to a different candidate path")
            if row.context_digest != document_context.digest:
                raise ValueError("global score is bound to a different document context")
            if row.source != self.source or row.profile_digest != self.profile_digest:
                raise ValueError("global scorer identity changed within the factorial experiment")
            if candidate.candidate_id in seen:
                raise ValueError("global scorer returned duplicate candidate identity")
            seen.add(candidate.candidate_id)
            output.append(
                ContextCandidateScore(
                    candidate_id=candidate.candidate_id,
                    candidate_text_sha256=candidate.text_sha256,
                    context_digest=context.digest,
                    value=row.value,
                    source=self.source,
                    scorer_profile_digest=self.profile_digest,
                )
            )
        return tuple(output)
