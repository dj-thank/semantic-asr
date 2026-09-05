"""Opt-in application and transcriber wrapper for document deliberation."""

from __future__ import annotations

from pathlib import Path

from ..contracts import sha256_json
from ..deliberation_lattice import DocumentContext
from ..global_deliberation import DeliberationPolicy
from ..global_scorer import GlobalSequenceScorer
from ..longform import LongformResult, SemanticASRTranscriber
from ..longform_deliberation import (
    DeliberatedLongformResult,
    DeliberatedLongformSegment,
    SpanProposalProvider,
    _applied_segment,
    _unchanged_segment,
)
from ..semantic_deliberation import SemanticDeliberationConfig
from .application import _local_decision, _result, _trace
from .config import DocumentBeamConfig
from .planning import plan_document_deliberation


def apply_document_deliberation(
    first_pass: LongformResult,
    *,
    config: DocumentBeamConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    sequence_scorer: GlobalSequenceScorer | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
    audio_path: str | Path | None = None,
    shuffle_seed: str = "semantic-asr-shuffled-context-v1",
) -> LongformResult | DeliberatedLongformResult:
    """Jointly choose and optionally apply one window-path sequence for a recording."""

    config = config or DocumentBeamConfig()
    if not config.enabled:
        return first_pass
    local_policy = local_policy or DeliberationPolicy.conservative_default()
    try:
        plan = plan_document_deliberation(
            first_pass,
            config=config,
            build_config=build_config,
            local_policy=local_policy,
            sequence_scorer=sequence_scorer,
            proposal_provider=proposal_provider,
            declared_context=declared_context,
            audio_path=audio_path,
            shuffle_seed=shuffle_seed,
        )
    except Exception:
        if config.fail_closed_to_first_pass:
            return first_pass
        raise

    document = plan.decision
    apply_changes = document.changed and (document.status == "accepted" or config.apply_provisional)
    bound_policy_digest = sha256_json(
        {
            "localPolicyDigest": local_policy.digest,
            "documentDecisionDigest": document.digest,
            "application": "document-joint-beam-v1",
        }
    )
    segments: list[DeliberatedLongformSegment] = []
    for window, option in zip(plan.window_sets, document.selected.options, strict=True):
        local = _local_decision(
            window,
            option,
            document,
            policy_digest=bound_policy_digest,
        )
        applied = apply_changes and option.changed
        if applied:
            reason = "applied-document-joint-beam"
        elif option.changed:
            reason = "document-provisional-not-applied"
        else:
            reason = "document-retained-first-pass"
        trace = _trace(
            window,
            option,
            local,
            document,
            applied=applied,
            config_digest=config.digest,
            policy_digest=bound_policy_digest,
            reason=reason,
        )
        segments.append(
            _applied_segment(window.segment, window.build, local, trace)
            if applied
            else _unchanged_segment(window.segment, trace)
        )
    return _result(
        first_pass,
        segments,
        config_digest=config.digest,
        policy_digest=bound_policy_digest,
        decision=document,
        plan_digest=plan.digest,
    )


class DocumentDeliberatingTranscriber:
    """Composition wrapper; the measured first pass remains untouched."""

    def __init__(
        self,
        first_pass: SemanticASRTranscriber,
        *,
        config: DocumentBeamConfig | None = None,
        build_config: SemanticDeliberationConfig | None = None,
        local_policy: DeliberationPolicy | None = None,
        sequence_scorer: GlobalSequenceScorer | None = None,
        proposal_provider: SpanProposalProvider | None = None,
        declared_context: DocumentContext | None = None,
    ) -> None:
        self.first_pass = first_pass
        self.config = config or DocumentBeamConfig()
        self.build_config = build_config or SemanticDeliberationConfig()
        self.local_policy = local_policy or DeliberationPolicy.conservative_default()
        self.sequence_scorer = sequence_scorer
        self.proposal_provider = proposal_provider
        self.declared_context = declared_context or DocumentContext()
        if self.config.require_sequence_scorer and sequence_scorer is None:
            raise ValueError("document-deliberating transcriber requires a scorer")

    def __getattr__(self, name: str):
        return getattr(self.first_pass, name)

    def transcribe(self, audio_path: str | Path, **kwargs):
        first_pass = self.first_pass.transcribe(audio_path, **kwargs)
        return apply_document_deliberation(
            first_pass,
            config=self.config,
            build_config=self.build_config,
            local_policy=self.local_policy,
            sequence_scorer=self.sequence_scorer,
            proposal_provider=self.proposal_provider,
            declared_context=self.declared_context,
            audio_path=audio_path,
        )


def with_document_deliberation(
    transcriber: SemanticASRTranscriber,
    *,
    sequence_scorer: GlobalSequenceScorer,
    config: DocumentBeamConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
) -> DocumentDeliberatingTranscriber:
    return DocumentDeliberatingTranscriber(
        transcriber,
        config=config,
        build_config=build_config,
        local_policy=local_policy,
        sequence_scorer=sequence_scorer,
        proposal_provider=proposal_provider,
        declared_context=declared_context,
    )
