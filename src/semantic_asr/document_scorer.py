"""Adapt existing sequence scorers to complete-path, bidirectional-context deliberation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import canonical_json, sha256_json
from .deliberation_evidence import _is_sha256
from .deliberation_lattice import DocumentContext, LatticeArc, path_digest
from .global_scorer import GlobalPathScore
from .score_types import ScoreSemantics
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


def _scorer_identity(
    scorer: SequenceScorer,
    explicit_digest: str | None,
) -> tuple[str, dict[str, object]]:
    if explicit_digest is not None:
        if not _is_sha256(explicit_digest):
            raise ValueError("scorer_identity_digest must be a SHA-256 value")
        return explicit_digest, {"explicitIdentityDigest": explicit_digest}

    artifact = getattr(scorer, "model_artifact_sha256", None)
    config_digest = getattr(scorer, "config_digest", None)
    config = getattr(scorer, "config", None)
    model_revision = getattr(scorer, "model_revision", None)
    if model_revision is None and config is not None:
        model_revision = getattr(config, "model_revision", None)
    immutable = next(
        (
            value
            for value in (artifact, config_digest)
            if isinstance(value, str) and _is_sha256(value)
        ),
        None,
    )
    if (
        immutable is None
        and model_revision is None
        and not bool(getattr(scorer, "allow_legacy_deliberation_identity", False))
    ):
        raise ValueError(
            "global sequence scorer identity is not immutable; provide an exact model revision, "
            "artifact/config digest, or scorer_identity_digest"
        )
    payload: dict[str, object] = {
        "type": f"{type(scorer).__module__}.{type(scorer).__qualname__}",
        "name": scorer.name,
        "modelRevision": model_revision,
        "modelArtifactSha256": artifact,
        "configDigest": config_digest,
        "config": config,
        "legacyFixture": bool(getattr(scorer, "allow_legacy_deliberation_identity", False)),
    }
    return sha256_json(payload), payload


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
        if not _is_sha256(self.fitted_manifest_sha256):
            raise ValueError("fitted_manifest_sha256 must be a SHA-256 value")
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

    def transform(self, *, cumulative_log_likelihood: float, token_count: int) -> float:
        total = _strict_float(cumulative_log_likelihood, name="cumulative_log_likelihood")
        if isinstance(token_count, bool) or token_count < 1:
            raise ValueError("token_count must be positive")
        source = total / token_count if self.length_normalized else total
        return math.tanh((source - self.center) / self.scale)


@dataclass(frozen=True, slots=True)
class DocumentPromptFormat:
    include_left_context: bool = True
    include_right_context: bool = True
    include_topic_summary: bool = True
    include_entity_ids: bool = True
    include_metadata: bool = True
    maximum_context_characters: int = 12_000
    revision: str = "document-context-v2"

    def __post_init__(self) -> None:
        if isinstance(self.maximum_context_characters, bool):
            raise TypeError("maximum_context_characters must be an integer")
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
                "includeMetadata": self.include_metadata,
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
        if self.include_metadata and context.metadata:
            sections.append(("declaredMetadata", context.metadata))
        rendered = canonical_json({"format": self.revision, "sections": sections})
        limit = self.maximum_context_characters
        if limit == 0:
            return ""
        if len(rendered) <= limit:
            return rendered
        digest = sha256_json({"fullContext": rendered})
        digest_only = canonical_json(
            {"format": self.revision, "fullContextSha256": digest, "truncated": True}
        )
        if len(digest_only) >= limit:
            return digest_only[:limit]

        low = 0
        high = len(rendered)
        best = digest_only
        while low <= high:
            middle = (low + high) // 2
            candidate = canonical_json(
                {
                    "format": self.revision,
                    "contextPrefix": rendered[:middle],
                    "fullContextSha256": digest,
                    "truncated": True,
                }
            )
            if len(candidate) <= limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best


class SequenceScorerGlobalAdapter:
    """Batch adapter from the existing causal scorer to complete-path preferences."""

    def __init__(
        self,
        scorer: SequenceScorer,
        normalization: GlobalScoreNormalization,
        *,
        prompt_format: DocumentPromptFormat | None = None,
        source: str | None = None,
        scorer_identity_digest: str | None = None,
    ) -> None:
        self.scorer = scorer
        self.normalization = normalization
        self.prompt_format = prompt_format or DocumentPromptFormat()
        self.source = source or f"global-sequence:{scorer.name}"
        identity_digest, identity = _scorer_identity(scorer, scorer_identity_digest)
        self.scorer_identity_digest = identity_digest
        self.profile_digest = sha256_json(
            {
                "source": self.source,
                "scorerIdentityDigest": identity_digest,
                "scorerIdentity": identity,
                "normalizationDigest": normalization.digest,
                "promptFormatDigest": self.prompt_format.digest,
                "candidateFormat": "complete-deliberation-path-v2",
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
            text = "".join(arc.text for arc in path)
            if not text:
                raise ValueError("global sequence scorer cannot score an empty transcript path")
            candidate_id = f"path-{index:06d}-{digest[:16]}"
            identifiers.append(candidate_id)
            candidates.append(TextCandidate(candidate_id=candidate_id, text=text))
        rows = tuple(self.scorer.score(candidates, context=rendered_context))
        if len(rows) != len(candidates):
            raise ValueError("sequence scorer returned the wrong number of path scores")
        by_id = {}
        for row in rows:
            if row.candidate_id in by_id:
                raise ValueError("sequence scorer returned a duplicate path score")
            cumulative = getattr(row, "cumulative", None)
            token_count = getattr(row, "token_count", None)
            if cumulative is None or token_count is None:
                raise TypeError("global adapter requires sequence log-likelihood results")
            if cumulative.semantics != ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD:
                raise ValueError("global adapter requires cumulative log-likelihood semantics")
            by_id[row.candidate_id] = row
        if set(by_id) != set(identifiers):
            raise ValueError("sequence scorer returned unknown or missing path IDs")
        output: list[GlobalPathScore] = []
        for identifier, path in zip(identifiers, paths, strict=True):
            row = by_id[identifier]
            output.append(
                GlobalPathScore(
                    value=self.normalization.transform(
                        cumulative_log_likelihood=row.cumulative.value,
                        token_count=row.token_count,
                    ),
                    source=self.source,
                    profile_digest=self.profile_digest,
                    path_digest=path_digest(path),
                    context_digest=context.digest,
                )
            )
        return tuple(output)
