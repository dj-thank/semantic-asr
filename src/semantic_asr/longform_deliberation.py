"""Opt-in long-form second pass over exact local semantic lattices.

The measured first-pass transcriber remains unchanged. This module wraps it, freezes every first-
pass window, then performs a bidirectional document-context deliberation pass. A changed observed
transcript is represented by a new immutable receipt bound to the first-pass evidence, lattice,
context, policy, complete selected path and source audio.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    CandidateEvidence,
    NormalizedTranscript,
    ObservedTranscript,
    sha256_json,
)
from .deliberation_evidence import _is_sha256
from .deliberation_lattice import DocumentContext, LatticeArc
from .global_deliberation import (
    DeliberationPolicy,
    GlobalDeliberationDecision,
    decode_global_lattice,
)
from .global_scorer import GlobalSequenceScorer
from .japanese import deterministic_normalize, join_japanese_fragments
from .longform import LongformResult, LongformSegment, SemanticASRTranscriber, Window
from .semantic_deliberation import (
    SemanticDeliberationBuild,
    SemanticDeliberationConfig,
    VerifiedSpanProposal,
    build_semantic_deliberation_lattice,
    path_is_recombined,
    path_source_candidate_ids,
)


class SpanProposalProvider(Protocol):
    """Acquire independently verified local proposals after the base lattice is known."""

    def __call__(
        self,
        *,
        audio_path: str | Path | None,
        segment_index: int,
        segment: LongformSegment,
        build: SemanticDeliberationBuild,
        context: DocumentContext,
        source_audio_sha256: str,
    ) -> Mapping[str, Sequence[VerifiedSpanProposal]]: ...


@dataclass(frozen=True, slots=True)
class LongformDeliberationConfig:
    enabled: bool = True
    maximum_left_windows: int = 4
    maximum_right_windows: int = 4
    maximum_context_characters: int = 12_000
    minimum_distinct_surfaces: int = 2
    require_sequence_scorer: bool = True
    apply_provisional: bool = False
    fail_closed_to_first_pass: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "maximum_left_windows",
            "maximum_right_windows",
            "maximum_context_characters",
            "minimum_distinct_surfaces",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.minimum_distinct_surfaces < 2:
            raise ValueError("minimum_distinct_surfaces must be at least two")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "enabled": self.enabled,
                "maximumLeftWindows": self.maximum_left_windows,
                "maximumRightWindows": self.maximum_right_windows,
                "maximumContextCharacters": self.maximum_context_characters,
                "minimumDistinctSurfaces": self.minimum_distinct_surfaces,
                "requireSequenceScorer": self.require_sequence_scorer,
                "applyProvisional": self.apply_provisional,
                "failClosedToFirstPass": self.fail_closed_to_first_pass,
                "contextSource": "frozen-first-pass-windows-v1",
            }
        )


@dataclass(frozen=True, slots=True)
class PathArcReceipt:
    arc_id: str
    span_id: str
    text: str
    arc_digest: str
    origin: str
    source_candidate_ids: tuple[str, ...]
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.arc_id or not self.span_id or not _is_sha256(self.arc_digest):
            raise ValueError("path arc receipt requires IDs and an arc SHA-256")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("path arc receipt has an invalid time range")

    @classmethod
    def from_arc(cls, arc: LatticeArc, *, start_ms: int, end_ms: int) -> PathArcReceipt:
        return cls(
            arc_id=arc.arc_id,
            span_id=arc.span_id,
            text=arc.text,
            arc_digest=arc.digest,
            origin=arc.origin,
            source_candidate_ids=arc.source_candidate_ids,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class SegmentDeliberationTrace:
    attempted: bool
    applied: bool
    reason: str
    first_pass_evidence_sha256: str
    context_digest: str
    config_digest: str
    policy_digest: str
    build_digest: str | None = None
    lattice_digest: str | None = None
    decision_digest: str | None = None
    selected_path_digest: str | None = None
    retained_path_digest: str | None = None
    scorer_source: str | None = None
    scorer_profile_digest: str | None = None
    decision_status: str | None = None
    margin: float | None = None
    selected_text_sha256: str | None = None
    changed_span_ids: tuple[str, ...] = ()
    proposal_digests: tuple[str, ...] = ()
    exact_source_candidate_ids: tuple[str, ...] = ()
    recombined: bool = False

    def __post_init__(self) -> None:
        for digest in (
            self.first_pass_evidence_sha256,
            self.context_digest,
            self.config_digest,
            self.policy_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("deliberation trace requires SHA-256 base digests")
        for digest in (
            self.build_digest,
            self.lattice_digest,
            self.decision_digest,
            self.selected_path_digest,
            self.retained_path_digest,
            self.scorer_profile_digest,
            self.selected_text_sha256,
            *self.proposal_digests,
        ):
            if digest is not None and not _is_sha256(digest):
                raise ValueError("deliberation trace contains an invalid SHA-256 value")
        if self.margin is not None:
            margin = float(self.margin)
            if not math.isfinite(margin) or margin < 0.0:
                raise ValueError("deliberation trace margin must be finite and non-negative")
            object.__setattr__(self, "margin", margin)
        if self.applied and not self.attempted:
            raise ValueError("an unapplied deliberation attempt cannot be marked applied")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DeliberatedObservedTranscript:
    """Immutable observed receipt for a selected multi-arc path."""

    text: str
    selected_candidate_id: str
    source_audio_sha256: str
    evidence_sha256: str
    first_pass_evidence_sha256: str
    build_digest: str
    decision_digest: str
    selected_path_digest: str
    decision: str
    path_arcs: tuple[PathArcReceipt, ...]
    exact_source_candidate_ids: tuple[str, ...] = ()
    selected_posterior: float = 0.0
    candidates: tuple[CandidateEvidence, ...] = ()
    ranked: tuple[object, ...] = ()
    uncertainty_spans: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not self.text or not self.selected_candidate_id or not self.path_arcs:
            raise ValueError("deliberated observed transcript requires text, ID and path arcs")
        for digest in (
            self.source_audio_sha256,
            self.evidence_sha256,
            self.first_pass_evidence_sha256,
            self.build_digest,
            self.decision_digest,
            self.selected_path_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("deliberated observed transcript contains an invalid digest")
        if self.decision not in {"accepted", "provisional"}:
            raise ValueError("deliberated observed decision must be accepted or provisional")
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        text: str,
        source_audio_sha256: str,
        first_pass_evidence_sha256: str,
        build_digest: str,
        decision: GlobalDeliberationDecision,
        path_arcs: tuple[PathArcReceipt, ...],
        exact_source_candidate_ids: tuple[str, ...],
        uncertainty_spans: tuple[dict[str, object], ...],
    ) -> DeliberatedObservedTranscript:
        selected_candidate_id = f"deliberation-path-{decision.selected.digest[:20]}"
        payload = {
            "text": text,
            "selectedCandidateId": selected_candidate_id,
            "sourceAudioSha256": source_audio_sha256,
            "firstPassEvidenceSha256": first_pass_evidence_sha256,
            "buildDigest": build_digest,
            "decisionDigest": decision.digest,
            "selectedPathDigest": decision.selected.digest,
            "decision": decision.status,
            "pathArcDigests": [arc.digest for arc in path_arcs],
            "exactSourceCandidateIds": exact_source_candidate_ids,
            "uncertaintySpans": uncertainty_spans,
        }
        return cls(
            text=text,
            selected_candidate_id=selected_candidate_id,
            source_audio_sha256=source_audio_sha256,
            evidence_sha256=sha256_json(payload),
            first_pass_evidence_sha256=first_pass_evidence_sha256,
            build_digest=build_digest,
            decision_digest=decision.digest,
            selected_path_digest=decision.selected.digest,
            decision=decision.status,
            path_arcs=path_arcs,
            exact_source_candidate_ids=exact_source_candidate_ids,
            uncertainty_spans=uncertainty_spans,
        )

    def verify(self) -> None:
        if "".join(arc.text for arc in self.path_arcs) != self.text:
            raise ValueError("deliberated path receipts do not reconstruct observed text")
        payload = {
            "text": self.text,
            "selectedCandidateId": self.selected_candidate_id,
            "sourceAudioSha256": self.source_audio_sha256,
            "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
            "buildDigest": self.build_digest,
            "decisionDigest": self.decision_digest,
            "selectedPathDigest": self.selected_path_digest,
            "decision": self.decision,
            "pathArcDigests": [arc.digest for arc in self.path_arcs],
            "exactSourceCandidateIds": self.exact_source_candidate_ids,
            "uncertaintySpans": self.uncertainty_spans,
        }
        if sha256_json(payload) != self.evidence_sha256:
            raise ValueError("deliberated observed evidence hash mismatch")


@dataclass(frozen=True, slots=True)
class DeliberatedLongformSegment:
    window: Window
    observed: ObservedTranscript | DeliberatedObservedTranscript
    normalized: NormalizedTranscript
    diagnostics: dict[str, object]
    trace: SegmentDeliberationTrace
    first_pass_evidence_sha256: str
    actions: tuple[object, ...] = ()
    cache_hits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.observed.evidence_sha256 != self.normalized.observed_evidence_sha256:
            raise ValueError("normalized text is not linked to final observed evidence")
        if self.first_pass_evidence_sha256 != self.trace.first_pass_evidence_sha256:
            raise ValueError("segment trace is not linked to first-pass evidence")
        self.observed.verify()
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def changed(self) -> bool:
        return isinstance(self.observed, DeliberatedObservedTranscript)

    def as_dict(self) -> dict[str, object]:
        return {
            "window": asdict(self.window),
            "observed": asdict(self.observed),
            "normalized": asdict(self.normalized),
            "diagnostics": dict(self.diagnostics),
            "actions": [
                asdict(action) if hasattr(action, "__dataclass_fields__") else action
                for action in self.actions
            ],
            "cache_hits": list(self.cache_hits),
            "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
            "deliberation": asdict(self.trace),
            "deliberationDigest": self.trace.digest,
        }


@dataclass(frozen=True, slots=True)
class DeliberatedLongformResult:
    source_name: str
    source_audio_sha256: str
    duration_ms: int
    observed_text: str
    normalized_text: str
    segments: tuple[DeliberatedLongformSegment, ...]
    evidence_sha256: str
    diagnostics: dict[str, object]
    first_pass_evidence_sha256: str
    deliberation_evidence_sha256: str
    config_digest: str
    policy_digest: str
    first_pass: LongformResult = field(repr=False)

    def __post_init__(self) -> None:
        self.verify()
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def as_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_audio_sha256": self.source_audio_sha256,
            "duration_ms": self.duration_ms,
            "observed_text": self.observed_text,
            "normalized_text": self.normalized_text,
            "segments": [segment.as_dict() for segment in self.segments],
            "evidence_sha256": self.evidence_sha256,
            "first_pass_evidence_sha256": self.first_pass_evidence_sha256,
            "deliberation_evidence_sha256": self.deliberation_evidence_sha256,
            "config_digest": self.config_digest,
            "policy_digest": self.policy_digest,
            "diagnostics": dict(self.diagnostics),
        }

    def verify(self) -> None:
        if self.first_pass.evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("long-form deliberation is linked to different first-pass evidence")
        expected_first_pass = sha256_json(
            {
                "sourceAudioSha256": self.first_pass.source_audio_sha256,
                "durationMs": self.first_pass.duration_ms,
                "observedText": self.first_pass.observed_text,
                "normalizedText": self.first_pass.normalized_text,
                "segmentEvidence": [
                    segment.observed.evidence_sha256 for segment in self.first_pass.segments
                ],
            }
        )
        if expected_first_pass != self.first_pass_evidence_sha256:
            raise ValueError("first-pass long-form evidence hash mismatch")
        for segment in self.segments:
            segment.observed.verify()
            if segment.observed.evidence_sha256 != segment.normalized.observed_evidence_sha256:
                raise ValueError("segment normalization is linked to different observed evidence")
        observed = join_japanese_fragments(segment.observed.text for segment in self.segments)
        normalized = join_japanese_fragments(segment.normalized.text for segment in self.segments)
        if observed != self.observed_text or normalized != self.normalized_text:
            raise ValueError("deliberated long-form text does not match its segment sequence")
        deliberation_digest = sha256_json(
            {
                "traceDigests": [segment.trace.digest for segment in self.segments],
                "configDigest": self.config_digest,
                "policyDigest": self.policy_digest,
            }
        )
        if deliberation_digest != self.deliberation_evidence_sha256:
            raise ValueError("long-form deliberation trace hash mismatch")
        expected = sha256_json(
            {
                "sourceAudioSha256": self.source_audio_sha256,
                "durationMs": self.duration_ms,
                "observedText": self.observed_text,
                "normalizedText": self.normalized_text,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "deliberationEvidenceSha256": self.deliberation_evidence_sha256,
                "segmentEvidence": [segment.observed.evidence_sha256 for segment in self.segments],
            }
        )
        if expected != self.evidence_sha256:
            raise ValueError("deliberated long-form evidence hash mismatch")


def _bounded_context(value: str, limit: int, *, keep_suffix: bool) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[-limit:] if keep_suffix else value[:limit]


def _document_context(
    first_pass: LongformResult,
    segment_index: int,
    *,
    config: LongformDeliberationConfig,
    declared: DocumentContext,
) -> DocumentContext:
    texts = [segment.observed.text for segment in first_pass.segments]
    left_start = max(0, segment_index - config.maximum_left_windows)
    right_end = min(len(texts), segment_index + 1 + config.maximum_right_windows)
    left_rows = texts[left_start:segment_index]
    right_rows = texts[segment_index + 1 : right_end]
    left = "\n".join((*((declared.left_context,) if declared.left_context else ()), *left_rows))
    right = "\n".join((*right_rows, *((declared.right_context,) if declared.right_context else ())))
    half = config.maximum_context_characters // 2
    left = _bounded_context(left, half, keep_suffix=True)
    right = _bounded_context(
        right,
        config.maximum_context_characters - len(left),
        keep_suffix=False,
    )
    return DocumentContext(
        left_context=left,
        right_context=right,
        topic_summary=declared.topic_summary,
        entity_ids=declared.entity_ids,
        metadata={
            **declared.metadata,
            "targetWindowIndex": segment_index,
            "leftWindowIndexes": tuple(range(left_start, segment_index)),
            "rightWindowIndexes": tuple(range(segment_index + 1, right_end)),
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "contextSource": "frozen-first-pass-windows-v1",
        },
    )


def _pivot_timeline(segment: LongformSegment) -> tuple[tuple[int, int, int, int], ...]:
    candidate = next(
        (
            row
            for row in segment.observed.candidates
            if row.candidate_id == segment.observed.selected_candidate_id
        ),
        None,
    )
    if candidate is None or not candidate.mora_units:
        return ()
    duration = segment.window.end_ms - segment.window.start_ms
    output: list[tuple[int, int, int, int]] = []
    for unit in candidate.mora_units:
        if (
            unit.char_start is None
            or unit.char_end is None
            or unit.start_ms is None
            or unit.end_ms is None
        ):
            return ()
        start = round(float(unit.start_ms))
        end = round(float(unit.end_ms))
        if 0 <= start < end <= duration:
            start += segment.window.start_ms
            end += segment.window.start_ms
        output.append((unit.char_start, unit.char_end, start, end))
    return tuple(output)


def _posterior(segment: LongformSegment) -> dict[str, float] | None:
    if not segment.observed.ranked:
        return None
    values = dict(segment.observed.ranked[0].gate.posterior)
    expected = {candidate.candidate_id for candidate in segment.observed.candidates}
    return values if set(values) == expected else None


def _skip_trace(
    segment: LongformSegment,
    context: DocumentContext,
    config: LongformDeliberationConfig,
    policy: DeliberationPolicy,
    reason: str,
) -> SegmentDeliberationTrace:
    return SegmentDeliberationTrace(
        attempted=False,
        applied=False,
        reason=reason,
        first_pass_evidence_sha256=segment.observed.evidence_sha256,
        context_digest=context.digest,
        config_digest=config.digest,
        policy_digest=policy.digest,
    )


def _decision_trace(
    segment: LongformSegment,
    build: SemanticDeliberationBuild,
    decision: GlobalDeliberationDecision,
    context: DocumentContext,
    config: LongformDeliberationConfig,
    policy: DeliberationPolicy,
    *,
    applied: bool,
    reason: str,
) -> SegmentDeliberationTrace:
    changed_span_ids = tuple(
        resolution.span_id
        for resolution in decision.resolutions
        if resolution.selected_arc_id != resolution.retained_arc_id
    )
    exact = path_source_candidate_ids(decision.selected.arcs)
    return SegmentDeliberationTrace(
        attempted=True,
        applied=applied,
        reason=reason,
        first_pass_evidence_sha256=segment.observed.evidence_sha256,
        context_digest=context.digest,
        config_digest=config.digest,
        policy_digest=policy.digest,
        build_digest=build.digest,
        lattice_digest=build.lattice.digest,
        decision_digest=decision.digest,
        selected_path_digest=decision.selected.digest,
        retained_path_digest=decision.retained.digest,
        scorer_source=decision.scorer_source,
        scorer_profile_digest=decision.scorer_profile_digest,
        decision_status=decision.status,
        margin=decision.margin,
        selected_text_sha256=sha256_json({"text": decision.selected.text}),
        changed_span_ids=changed_span_ids,
        proposal_digests=build.proposal_digests,
        exact_source_candidate_ids=exact,
        recombined=path_is_recombined(decision.selected.arcs),
    )


def _unchanged_segment(
    segment: LongformSegment,
    trace: SegmentDeliberationTrace,
) -> DeliberatedLongformSegment:
    diagnostics = {**dict(segment.diagnostics), "globalDeliberation": asdict(trace)}
    return DeliberatedLongformSegment(
        window=segment.window,
        observed=segment.observed,
        normalized=segment.normalized,
        diagnostics=diagnostics,
        trace=trace,
        first_pass_evidence_sha256=segment.observed.evidence_sha256,
        actions=segment.actions,
        cache_hits=segment.cache_hits,
    )


def _applied_segment(
    segment: LongformSegment,
    build: SemanticDeliberationBuild,
    decision: GlobalDeliberationDecision,
    trace: SegmentDeliberationTrace,
) -> DeliberatedLongformSegment:
    receipts = tuple(
        PathArcReceipt.from_arc(arc, start_ms=span.start_ms, end_ms=span.end_ms)
        for span, arc in zip(build.lattice.spans, decision.selected.arcs, strict=True)
    )
    uncertainty = tuple(
        {
            "spanId": span.span_id,
            "startMs": span.start_ms,
            "endMs": span.end_ms,
            "retainedArcId": resolution.retained_arc_id,
            "selectedArcId": resolution.selected_arc_id,
            "resolutionMode": resolution.mode,
        }
        for span, resolution in zip(build.lattice.spans, decision.resolutions, strict=True)
        if resolution.selected_arc_id != resolution.retained_arc_id
    )
    exact = path_source_candidate_ids(decision.selected.arcs)
    observed = DeliberatedObservedTranscript.create(
        text=decision.selected.text,
        source_audio_sha256=build.lattice.source_audio_sha256,
        first_pass_evidence_sha256=segment.observed.evidence_sha256,
        build_digest=build.digest,
        decision=decision,
        path_arcs=receipts,
        exact_source_candidate_ids=exact,
        uncertainty_spans=uncertainty,
    )
    normalized = NormalizedTranscript.attach(
        observed,
        text=deterministic_normalize(observed.text),
        mode="deterministic",
    )
    diagnostics = {
        **dict(segment.diagnostics),
        "firstPassTopPosterior": dict(segment.diagnostics).get("topPosterior"),
        "topPosterior": None,
        "confidenceInvalidatedByGlobalDeliberation": True,
        "globalDeliberation": asdict(trace),
        "observedStatus": decision.status,
    }
    return DeliberatedLongformSegment(
        window=segment.window,
        observed=observed,
        normalized=normalized,
        diagnostics=diagnostics,
        trace=trace,
        first_pass_evidence_sha256=segment.observed.evidence_sha256,
        actions=segment.actions,
        cache_hits=segment.cache_hits,
    )


def apply_longform_deliberation(
    first_pass: LongformResult,
    *,
    config: LongformDeliberationConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    policy: DeliberationPolicy | None = None,
    sequence_scorer: GlobalSequenceScorer | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
    audio_path: str | Path | None = None,
) -> LongformResult | DeliberatedLongformResult:
    """Apply a deterministic second pass after all first-pass windows have completed."""

    config = config or LongformDeliberationConfig()
    if not config.enabled:
        return first_pass
    if config.require_sequence_scorer and sequence_scorer is None:
        raise ValueError("long-form deliberation requires an explicit global sequence scorer")
    build_config = build_config or SemanticDeliberationConfig()
    policy = policy or DeliberationPolicy.conservative_default()
    declared_context = declared_context or DocumentContext()

    final_segments: list[DeliberatedLongformSegment] = []
    failure_count = 0
    proposed_not_applied = 0
    for index, segment in enumerate(first_pass.segments):
        context = _document_context(
            first_pass,
            index,
            config=config,
            declared=declared_context,
        )
        distinct_surfaces = {candidate.text for candidate in segment.observed.candidates}
        if len(distinct_surfaces) < config.minimum_distinct_surfaces:
            final_segments.append(
                _unchanged_segment(
                    segment,
                    _skip_trace(segment, context, config, policy, "insufficient-distinct-surfaces"),
                )
            )
            continue
        try:
            build = build_semantic_deliberation_lattice(
                segment.observed.candidates,
                posterior=_posterior(segment),
                pivot_candidate_id=segment.observed.selected_candidate_id,
                document_id=f"{first_pass.source_audio_sha256[:16]}:window:{index:04d}",
                source_audio_sha256=first_pass.source_audio_sha256,
                segment_start_ms=segment.window.start_ms,
                segment_end_ms=segment.window.end_ms,
                pivot_timeline=_pivot_timeline(segment),
                config=build_config,
            )
            if proposal_provider is not None:
                proposals = proposal_provider(
                    audio_path=audio_path,
                    segment_index=index,
                    segment=segment,
                    build=build,
                    context=context,
                    source_audio_sha256=first_pass.source_audio_sha256,
                )
                if proposals:
                    build = build_semantic_deliberation_lattice(
                        segment.observed.candidates,
                        posterior=_posterior(segment),
                        pivot_candidate_id=segment.observed.selected_candidate_id,
                        document_id=f"{first_pass.source_audio_sha256[:16]}:window:{index:04d}",
                        source_audio_sha256=first_pass.source_audio_sha256,
                        segment_start_ms=segment.window.start_ms,
                        segment_end_ms=segment.window.end_ms,
                        pivot_timeline=_pivot_timeline(segment),
                        proposals=proposals,
                        config=build_config,
                    )
            decision = decode_global_lattice(
                build.lattice,
                policy=policy,
                context=context,
                sequence_scorer=sequence_scorer,
            )
            changed = decision.selected.text != segment.observed.text
            applied = changed and (decision.status == "accepted" or config.apply_provisional)
            if not changed:
                reason = "retained-first-pass"
            elif not applied:
                reason = "provisional-not-applied"
                proposed_not_applied += 1
            else:
                reason = "applied-global-deliberation"
            trace = _decision_trace(
                segment,
                build,
                decision,
                context,
                config,
                policy,
                applied=applied,
                reason=reason,
            )
            final_segments.append(
                _applied_segment(segment, build, decision, trace)
                if applied
                else _unchanged_segment(segment, trace)
            )
        except Exception as exc:
            if not config.fail_closed_to_first_pass:
                raise
            failure_count += 1
            trace = _skip_trace(
                segment,
                context,
                config,
                policy,
                f"failed-closed:{type(exc).__name__}:{exc}",
            )
            final_segments.append(_unchanged_segment(segment, trace))

    observed_text = join_japanese_fragments(segment.observed.text for segment in final_segments)
    normalized_text = join_japanese_fragments(segment.normalized.text for segment in final_segments)
    deliberation_evidence_sha256 = sha256_json(
        {
            "traceDigests": [segment.trace.digest for segment in final_segments],
            "configDigest": config.digest,
            "policyDigest": policy.digest,
        }
    )
    evidence_sha256 = sha256_json(
        {
            "sourceAudioSha256": first_pass.source_audio_sha256,
            "durationMs": first_pass.duration_ms,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "deliberationEvidenceSha256": deliberation_evidence_sha256,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in final_segments],
        }
    )
    changed_count = sum(segment.changed for segment in final_segments)
    provisional_count = sum(segment.observed.decision != "accepted" for segment in final_segments)
    diagnostics = {
        **dict(first_pass.diagnostics),
        "provisionalWindowCount": provisional_count,
        "globalDeliberation": {
            "enabled": True,
            "configDigest": config.digest,
            "buildConfigDigest": build_config.digest,
            "policyDigest": policy.digest,
            "scorerSource": getattr(sequence_scorer, "source", None),
            "scorerProfileDigest": getattr(sequence_scorer, "profile_digest", None),
            "changedWindowCount": changed_count,
            "proposedButNotAppliedCount": proposed_not_applied,
            "failedWindowCount": failure_count,
            "evidenceSha256": deliberation_evidence_sha256,
        },
    }
    return DeliberatedLongformResult(
        source_name=first_pass.source_name,
        source_audio_sha256=first_pass.source_audio_sha256,
        duration_ms=first_pass.duration_ms,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=tuple(final_segments),
        evidence_sha256=evidence_sha256,
        diagnostics=diagnostics,
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        deliberation_evidence_sha256=deliberation_evidence_sha256,
        config_digest=config.digest,
        policy_digest=policy.digest,
        first_pass=first_pass,
    )


class DeliberatingSemanticASRTranscriber:
    """Composition wrapper that keeps the measured first-pass transcriber untouched."""

    def __init__(
        self,
        first_pass: SemanticASRTranscriber,
        *,
        config: LongformDeliberationConfig | None = None,
        build_config: SemanticDeliberationConfig | None = None,
        policy: DeliberationPolicy | None = None,
        sequence_scorer: GlobalSequenceScorer | None = None,
        proposal_provider: SpanProposalProvider | None = None,
        declared_context: DocumentContext | None = None,
    ) -> None:
        self.first_pass = first_pass
        self.deliberation_config = config or LongformDeliberationConfig()
        self.deliberation_build_config = build_config or SemanticDeliberationConfig()
        self.deliberation_policy = policy or DeliberationPolicy.conservative_default()
        self.deliberation_sequence_scorer = sequence_scorer
        self.deliberation_proposal_provider = proposal_provider
        self.declared_context = declared_context or DocumentContext()
        if self.deliberation_config.require_sequence_scorer and sequence_scorer is None:
            raise ValueError("deliberating transcriber requires an explicit global sequence scorer")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.first_pass, name)

    def transcribe(
        self,
        audio_path: str | Path,
        **kwargs: Any,
    ) -> LongformResult | DeliberatedLongformResult:
        first_pass = self.first_pass.transcribe(audio_path, **kwargs)
        return apply_longform_deliberation(
            first_pass,
            config=self.deliberation_config,
            build_config=self.deliberation_build_config,
            policy=self.deliberation_policy,
            sequence_scorer=self.deliberation_sequence_scorer,
            proposal_provider=self.deliberation_proposal_provider,
            declared_context=self.declared_context,
            audio_path=audio_path,
        )


def with_global_deliberation(
    transcriber: SemanticASRTranscriber,
    *,
    sequence_scorer: GlobalSequenceScorer,
    config: LongformDeliberationConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    policy: DeliberationPolicy | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
) -> DeliberatingSemanticASRTranscriber:
    """Wrap a warm, profile-bound transcriber with the opt-in full-document second pass."""

    return DeliberatingSemanticASRTranscriber(
        transcriber,
        config=config,
        build_config=build_config,
        policy=policy,
        sequence_scorer=sequence_scorer,
        proposal_provider=proposal_provider,
        declared_context=declared_context,
    )
