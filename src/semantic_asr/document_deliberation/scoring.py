"""Complete-document rank-only rescoring."""

from __future__ import annotations

from collections.abc import Sequence

from ..deliberation_lattice import DocumentContext, path_digest
from ..global_scorer import GlobalPathScore, GlobalSequenceScorer
from ..longform import LongformResult
from .config import DocumentBeamConfig
from .path_types import DocumentPathHypothesis


def _score_document_paths(
    paths: Sequence[DocumentPathHypothesis],
    retained_digest: str,
    *,
    first_pass: LongformResult,
    config: DocumentBeamConfig,
    sequence_scorer: GlobalSequenceScorer | None,
    declared_context: DocumentContext,
) -> tuple[tuple[DocumentPathHypothesis, ...], str | None, str | None, DocumentContext]:
    candidates = list(paths[: config.global_rescore_paths])
    retained = next(path for path in paths if path.digest == retained_digest)
    if retained.digest not in {path.digest for path in candidates}:
        candidates.append(retained)
    candidates = list({path.digest: path for path in candidates}.values())
    context = DocumentContext(
        left_context=declared_context.left_context,
        right_context=declared_context.right_context,
        topic_summary=declared_context.topic_summary,
        entity_ids=declared_context.entity_ids,
        metadata={
            **declared_context.metadata,
            "mode": "whole-document-bidirectional-offline",
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "windowCount": len(first_pass.segments),
            "candidateDocumentContainsAllWindows": True,
            "streaming": False,
        },
    )
    if sequence_scorer is None:
        return tuple(candidates), None, None, context

    scorer_paths = tuple(
        (
            path.scorer_arc(
                source_audio_sha256=first_pass.source_audio_sha256,
                config_digest=config.digest,
            ),
        )
        for path in candidates
    )
    score_many = getattr(sequence_scorer, "score_many", None)
    rows = (
        tuple(score_many(scorer_paths, context=context))
        if callable(score_many)
        else tuple(sequence_scorer.score(path, context=context) for path in scorer_paths)
    )
    if len(rows) != len(candidates):
        raise ValueError("document scorer returned the wrong number of path scores")
    expected = {
        path_digest(scorer_path): candidate
        for candidate, scorer_path in zip(candidates, scorer_paths, strict=True)
    }
    by_document: dict[str, GlobalPathScore] = {}
    for row in rows:
        candidate = expected.get(row.path_digest)
        if candidate is None:
            raise ValueError("document scorer returned an unknown path digest")
        if row.context_digest != context.digest:
            raise ValueError("document scorer returned a score for different context")
        if candidate.digest in by_document:
            raise ValueError("document scorer returned a duplicate path score")
        by_document[candidate.digest] = row
    if set(by_document) != {path.digest for path in candidates}:
        raise ValueError("document scorer omitted a candidate path")
    sources = {row.source for row in rows}
    profiles = {row.profile_digest for row in rows}
    if len(sources) != 1 or len(profiles) != 1:
        raise ValueError("one document decision cannot mix scorer identities")
    source = next(iter(sources))
    profile = next(iter(profiles))
    rescored = tuple(
        DocumentPathHypothesis(
            options=path.options,
            overlap_receipts=path.overlap_receipts,
            base_score=path.base_score,
            mean_audio_support=path.mean_audio_support,
            context_score=by_document[path.digest].value,
            final_score=path.base_score
            + config.global_context_weight * by_document[path.digest].value,
            scorer_source=source,
            scorer_profile_digest=profile,
        )
        for path in candidates
    )
    return rescored, source, profile, context
