"""Local-window option construction for document deliberation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..deliberation_evidence import GENERATED_ORIGINS
from ..deliberation_lattice import DocumentContext
from ..global_deliberation import DeliberationPolicy, PathHypothesis, decode_global_lattice
from ..longform import LongformResult, LongformSegment
from ..longform_deliberation import SpanProposalProvider, _pivot_timeline, _posterior
from ..semantic_deliberation import (
    SemanticDeliberationBuild,
    SemanticDeliberationConfig,
    build_semantic_deliberation_lattice,
    path_is_recombined,
    path_source_candidate_ids,
)
from .config import DocumentBeamConfig
from .context_types import FrozenWindowContext
from .window_types import WindowPathOption, WindowPathSet


def _coverage_attribution(segments: Sequence[LongformSegment]) -> tuple[float, ...]:
    boundaries = sorted(
        {
            value
            for segment in segments
            for value in (segment.window.start_ms, segment.window.end_ms)
        }
    )
    attributed = [0.0] * len(segments)
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        active = [
            index
            for index, segment in enumerate(segments)
            if segment.window.start_ms < end and segment.window.end_ms > start
        ]
        if not active:
            continue
        share = (end - start) / len(active)
        for index in active:
            attributed[index] += share
    if any(value <= 0.0 for value in attributed):
        raise ValueError("every long-form window must receive positive audio coverage")
    return tuple(attributed)


def _option(
    segment: LongformSegment,
    build: SemanticDeliberationBuild,
    path: PathHypothesis,
    retained: PathHypothesis,
    coverage_ms: float,
) -> WindowPathOption:
    exact = path_source_candidate_ids(path.arcs)
    generated = any(arc.origin in GENERATED_ORIGINS for arc in path.arcs)
    return WindowPathOption(
        window_index=segment.window.index,
        start_ms=segment.window.start_ms,
        end_ms=segment.window.end_ms,
        build=build,
        path=path,
        retained_path_digest=retained.digest,
        coverage_ms=coverage_ms,
        local_score_delta=path.base_score - retained.base_score,
        audio_regression=retained.mean_audio_support - path.mean_audio_support,
        changed=path.digest != retained.digest,
        generated=generated,
        recombined=path_is_recombined(path.arcs),
        exact_source_candidate_ids=exact,
    )


def _build_window_set(
    first_pass: LongformResult,
    segment: LongformSegment,
    context: FrozenWindowContext,
    *,
    coverage_ms: float,
    config: DocumentBeamConfig,
    build_config: SemanticDeliberationConfig,
    local_policy: DeliberationPolicy,
    proposal_provider: SpanProposalProvider | None,
    audio_path: str | Path | None,
) -> WindowPathSet:
    build = build_semantic_deliberation_lattice(
        segment.observed.candidates,
        posterior=_posterior(segment),
        pivot_candidate_id=segment.observed.selected_candidate_id,
        document_id=f"{first_pass.source_audio_sha256[:16]}:window:{segment.window.index:04d}",
        source_audio_sha256=first_pass.source_audio_sha256,
        segment_start_ms=segment.window.start_ms,
        segment_end_ms=segment.window.end_ms,
        pivot_timeline=_pivot_timeline(segment),
        config=build_config,
    )
    active_contradictions = tuple(
        span for span in build.lattice.spans if bool(span.metadata.get("isContradiction"))
    )
    if proposal_provider is not None and active_contradictions:
        proposals = proposal_provider(
            audio_path=audio_path,
            segment_index=segment.window.index,
            segment=segment,
            build=build,
            context=context.context,
            source_audio_sha256=first_pass.source_audio_sha256,
        )
        if proposals:
            build = build_semantic_deliberation_lattice(
                segment.observed.candidates,
                posterior=_posterior(segment),
                pivot_candidate_id=segment.observed.selected_candidate_id,
                document_id=f"{first_pass.source_audio_sha256[:16]}:window:{segment.window.index:04d}",
                source_audio_sha256=first_pass.source_audio_sha256,
                segment_start_ms=segment.window.start_ms,
                segment_end_ms=segment.window.end_ms,
                pivot_timeline=_pivot_timeline(segment),
                proposals=proposals,
                config=build_config,
            )
    local = decode_global_lattice(
        build.lattice,
        policy=local_policy,
        context=DocumentContext(
            metadata={
                "mode": "document-beam-local-acoustic-filter",
                "firstPassEvidenceSha256": first_pass.evidence_sha256,
            }
        ),
        sequence_scorer=None,
    )
    retained_path = local.retained
    paths = list(local.alternatives[: config.local_paths_per_window])
    if retained_path.digest not in {path.digest for path in paths}:
        paths.append(retained_path)
    paths = sorted(
        {path.digest: path for path in paths}.values(),
        key=lambda path: (-path.base_score, path.digest),
    )
    options = tuple(_option(segment, build, path, retained_path, coverage_ms) for path in paths)
    retained_option = next(option for option in options if not option.changed)
    return WindowPathSet(
        window_index=segment.window.index,
        segment=segment,
        build=build,
        retained=retained_option,
        options=options,
        proposal_context=context,
    )
