"""Application receipts for a jointly selected document path."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

from ..contracts import sha256_json
from ..deliberation_evidence import GENERATED_ORIGINS
from ..deliberation_lattice import LatticeArc
from ..global_deliberation import GlobalDeliberationDecision, SpanResolution
from ..longform import LongformResult, join_segment_text
from ..longform_deliberation import (
    DeliberatedLongformResult,
    DeliberatedLongformSegment,
    SegmentDeliberationTrace,
)
from ..semantic_deliberation import path_source_candidate_ids
from .decision_types import DocumentDeliberationDecision
from .window_types import WindowPathOption, WindowPathSet


def _resolution_mode(selected: LatticeArc, retained: LatticeArc) -> str:
    if selected.arc_id == retained.arc_id:
        return "retained-first-pass"
    if (
        selected.pronunciation_key is not None
        and selected.pronunciation_key == retained.pronunciation_key
    ):
        return "context-resolved-orthography"
    if selected.origin in GENERATED_ORIGINS:
        return "acoustically-verified-proposal"
    return "acoustic-context-consensus"


def _local_decision(
    window: WindowPathSet,
    option: WindowPathOption,
    document: DocumentDeliberationDecision,
    *,
    policy_digest: str,
) -> GlobalDeliberationDecision:
    resolutions = tuple(
        SpanResolution(
            span_id=span.span_id,
            retained_arc_id=span.retained_arc_id,
            selected_arc_id=arc.arc_id,
            mode=_resolution_mode(arc, span.retained_arc),  # type: ignore[arg-type]
            retained_audio_support=None,
            selected_audio_support=None,
        )
        for span, arc in zip(window.build.lattice.spans, option.path.arcs, strict=True)
    )
    return GlobalDeliberationDecision(
        selected=option.path,
        retained=window.retained.path,
        alternatives=tuple(row.path for row in window.options),
        status=document.status,
        margin=document.margin,
        resolutions=resolutions,
        reasons=(
            "selected-by-document-joint-beam",
            f"document-decision:{document.digest}",
        ),
        lattice_digest=window.build.lattice.digest,
        policy_digest=policy_digest,
        context_digest=document.context_digest,
        scorer_source=document.scorer_source,
        scorer_profile_digest=document.scorer_profile_digest,
    )


def _trace(
    window: WindowPathSet,
    option: WindowPathOption,
    local: GlobalDeliberationDecision,
    document: DocumentDeliberationDecision,
    *,
    applied: bool,
    config_digest: str,
    policy_digest: str,
    reason: str,
) -> SegmentDeliberationTrace:
    changed_spans = tuple(
        resolution.span_id
        for resolution in local.resolutions
        if resolution.selected_arc_id != resolution.retained_arc_id
    )
    exact_source_candidate_ids = path_source_candidate_ids(option.path.arcs)
    if exact_source_candidate_ids != option.exact_source_candidate_ids:
        raise ValueError("window option exact source support changed before application")
    return SegmentDeliberationTrace(
        attempted=True,
        applied=applied,
        reason=f"{reason};document={document.digest}",
        first_pass_evidence_sha256=window.segment.observed.evidence_sha256,
        context_digest=document.context_digest,
        config_digest=config_digest,
        policy_digest=policy_digest,
        build_digest=window.build.digest,
        lattice_digest=window.build.lattice.digest,
        decision_digest=local.digest,
        selected_path_digest=option.path.digest,
        retained_path_digest=window.retained.path.digest,
        scorer_source=document.scorer_source,
        scorer_profile_digest=document.scorer_profile_digest,
        decision_status=document.status,
        margin=document.margin,
        selected_text_sha256=sha256_json({"text": option.text}),
        changed_span_ids=changed_spans,
        proposal_digests=window.build.proposal_digests,
        exact_source_candidate_ids=exact_source_candidate_ids,
        recombined=option.recombined,
    )


def _result(
    first_pass: LongformResult,
    segments: Sequence[DeliberatedLongformSegment],
    *,
    config_digest: str,
    policy_digest: str,
    decision: DocumentDeliberationDecision,
    plan_digest: str,
) -> DeliberatedLongformResult:
    rows = tuple(segments)
    observed_text = join_segment_text(rows)
    normalized_text = join_segment_text(rows, normalized=True)
    deliberation_evidence_sha256 = sha256_json(
        {
            "traceDigests": [segment.trace.digest for segment in rows],
            "configDigest": config_digest,
            "policyDigest": policy_digest,
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
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in rows],
            "segmentWindows": [asdict(segment.window) for segment in rows],
            "normalizations": [asdict(segment.normalized) for segment in rows],
        }
    )
    diagnostics = {
        **dict(first_pass.diagnostics),
        "provisionalWindowCount": sum(segment.observed.decision != "accepted" for segment in rows),
        "globalDeliberation": {
            "enabled": True,
            "mode": "document-joint-beam-v1",
            "configDigest": config_digest,
            "policyDigest": policy_digest,
            "documentDecisionDigest": decision.digest,
            "documentPlanDigest": plan_digest,
            "documentStatus": decision.status,
            "documentMargin": decision.margin,
            "scorerSource": decision.scorer_source,
            "scorerProfileDigest": decision.scorer_profile_digest,
            "changedWindowCount": sum(segment.changed for segment in rows),
            "selectedChangedWindowCount": decision.selected.changed_window_count,
            "selectedGeneratedWindowCount": decision.selected.generated_window_count,
            "candidateDocumentCount": len(decision.alternatives),
            "overlapReceiptDigests": tuple(
                receipt.digest for receipt in decision.selected.overlap_receipts
            ),
            "evidenceSha256": deliberation_evidence_sha256,
        },
    }
    return DeliberatedLongformResult(
        source_name=first_pass.source_name,
        source_audio_sha256=first_pass.source_audio_sha256,
        duration_ms=first_pass.duration_ms,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=rows,
        evidence_sha256=evidence_sha256,
        diagnostics=diagnostics,
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        deliberation_evidence_sha256=deliberation_evidence_sha256,
        config_digest=config_digest,
        policy_digest=policy_digest,
        first_pass=first_pass,
    )
