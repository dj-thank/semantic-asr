"""Adapt existing sequence scorers to complete-path, bidirectional-context deliberation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import canonical_json, sha256_json
from .deliberation_lattice import DocumentContext, LatticeArc, path_digest
from .global_scorer import GlobalPathScore
from .sequence_scorers import SequenceScorer, TextCandidate


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class GlobalScoreNormalization:
    """Held-out affine+tanh normalization for full-path sequence scores.

    The output is a bounded ranking preference. It is not a correctness probability.
    """

    center: float
    scale: float
    fitted_manifest_sha256: str
    revision: str
    length_normalized: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        center = _strict_float(self.center, name="normalization center")
        scale = _strict_float(self.scale, name="normalization scale")
        if scale <= 0.0:
            raise ValueError("normalization scale must be positive")
        if len(self.fitted_manifest_sha256) != 64:
            raise ValueError("fitted_manifest_sha256 must be a SHA-256 value")
        try:
            int(self.fitted_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("fitted_manifest_sha256 must be hexadecimal") from exc
        if not self.revision:
            raise ValueError("normalization revision is required")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "center": self.center,
                "scale": self.scale,
                "fittedManifestSha256": self.fitted_manifest_sha256,
                "revision": self.revision,
                "lengthNormalized": self.length_normalized,
                "transform": "tanh-affine-v1",
            }
        )

    def transform(self, *, total_log_probability: float, token_count: int) -> float:
        total = _strict_float(total_log_probability, name="total_log_probability")
        if token_count < 1:
            raise ValueError("token_count must be positive")
        source = total / token_count if self.length_normalized else total
        return math.tanh((source - self.center) / self.scale)


@dataclass(frozen=True, slots=True)
class DocumentPromptFormat:
    include_left_context: bool = True
    include_right_context: bool = True
    include_topic_summary: bool = True
    include_entity_ids: bool = True
    maximum_context_characters: int = 12_000
    revision: str = "document-context-v1"

    def __post_init__(self) -> None:
        if self.maximum_context_characters < 0:
            raise ValueError("maximum_context_characters must be non-negative")
        if not self.revision:
            raise ValueError("prompt format revision is required")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "includeLeftContext": self.include_left_context,
                "includeRightContext": self.include_right_context,
                "includeTopicSummary": self.include_topic_summary,
                "includeEntityIds": self.include_entity_ids,
                "maximumContextCharacters": self.maximum_context_characters,
                "revision": self.revision,
            }
        )

    def render(self, context: DocumentContext) -> str:
        sections: list[tuple[str, object]] = []
        if self.include_left_context and context.left_context:
            sections.append(("leftContext", context.left_context))
        if self.include_right_context and context.right_context:
            sections.append(("rightContext", context.right_context))
        if self.include_topic_summary and context.topic_summary:
            sections.append(("topicSummary", context.topic_summary))
        if self.include_entity_ids and context.entity_ids:
            sections.append(("entityIds", context.entity_ids))
        if context.metadata:
            sections.append(("declaredMetadata", context.metadata))
        rendered = canonical_json(
            {
                "format": self.revision,
                "sections": sections,
            }
        )
        if self.maximum_context_characters == 0:
            return ""
        if len(rendered) <= self.maximum_context_characters:
            return rendered
        digest = sha256_json({"fullContext": rendered})
        budget = max(0, self.maximum_context_characters - len(digest) - 40)
        return canonical_json(
            {
                "format": self.revision,
                "truncatedContextPrefix": rendered[:budget],
                "fullContextSha256": digest,
            }
        )


class SequenceScorerGlobalAdapter:
    """One-batch adapter from the existing sequence scorer to complete path preferences."""

    def __init__(
        self,
        scorer: SequenceScorer,
        normalization: GlobalScoreNormalization,
        *,
        prompt_format: DocumentPromptFormat | None = None,
        source: str | None = None,
    ) -> None:
        self.scorer = scorer
        self.normalization = normalization
        self.prompt_format = prompt_format or DocumentPromptFormat()
        self.source = source or f"global-sequence:{scorer.name}"
        self.profile_digest = sha256_json(
            {
                "source": self.source,
                "scorerName": scorer.name,
                "normalizationDigest": normalization.digest,
                "promptFormatDigest": self.prompt_format.digest,
                "candidateFormat": "complete-deliberation-path-v1",
            }
        )

    def score(
        self,
        path: Sequence[LatticeArc],
        *,
        context: DocumentContext,
    ) -> GlobalPathScore:
        return self.score_many((path,), context=context)[0]

    def score_many(
        self,
        paths: Sequence[Sequence[LatticeArc]],
        *,
        context: DocumentContext,
    ) -> tuple[GlobalPathScore, ...]:
        if not paths:
            return ()
        rendered_context = self.prompt_format.render(context)
        candidates: list[TextCandidate] = []
        identifiers: list[str] = []
        for index, path in enumerate(paths):
            digest = path_digest(path)
            candidate_id = f"path-{index:06d}-{digest[:16]}"
            identifiers.append(candidate_id)
            candidates.append(
                TextCandidate(
                    candidate_id=candidate_id,
                    text="".join(arc.text for arc in path),
                    context=rendered_context,
                )
            )
        rows = tuple(self.scorer.score(tuple(candidates), context=rendered_context))
        if len(rows) != len(candidates):
            raise ValueError("sequence scorer returned the wrong number of path scores")
        by_id = {}
        for row in rows:
            if row.candidate_id in by_id:
                raise ValueError("sequence scorer returned a duplicate path score")
            by_id[row.candidate_id] = row
        if set(by_id) != set(identifiers):
            raise ValueError("sequence scorer returned unknown or missing path IDs")
        output: list[GlobalPathScore] = []
        for identifier, path in zip(identifiers, paths, strict=True):
            row = by_id[identifier]
            output.append(
                GlobalPathScore(
                    value=self.normalization.transform(
                        total_log_probability=row.total_log_probability,
                        token_count=row.token_count,
                    ),
                    source=self.source,
                    profile_digest=self.profile_digest,
                    path_digest=path_digest(path),
                    context_digest=context.digest,
                )
            )
        return tuple(output)