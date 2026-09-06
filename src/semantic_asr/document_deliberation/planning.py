"""Public planning entry point for joint document deliberation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..contracts import sha256_json
from ..deliberation_lattice import DocumentContext
from ..global_deliberation import DeliberationPolicy
from ..global_scorer import GlobalSequenceScorer
from ..longform import LongformResult, sha256_file
from ..longform_deliberation import SpanProposalProvider
from ..semantic_deliberation import SemanticDeliberationConfig
from .config import DocumentBeamConfig
from .context import build_frozen_window_contexts
from .decision_types import DocumentDeliberationDecision, DocumentDeliberationPlan
from .local import _build_window_set, _coverage_attribution
from .scoring import _score_document_paths
from .search import _base_document_paths, _retained_hypothesis_with_config


def plan_document_deliberation(
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
) -> DocumentDeliberationPlan:
    """Plan one jointly selected sequence of window paths for the complete recording."""

    config = config or DocumentBeamConfig()
    first_pass.verify()
    if audio_path is not None and sha256_file(audio_path) != first_pass.source_audio_sha256:
        raise ValueError("document deliberation audio_path belongs to a different recording")
    if config.require_sequence_scorer and sequence_scorer is None:
        raise ValueError("document deliberation requires an explicit complete-document scorer")
    build_config = build_config or SemanticDeliberationConfig()
    local_policy = local_policy or DeliberationPolicy.conservative_default()
    declared = declared_context or DocumentContext()
    contexts = build_frozen_window_contexts(
        first_pass,
        arm=config.proposal_context_arm,
        declared_context=declared,
        maximum_left_windows=config.maximum_left_windows,
        maximum_right_windows=config.maximum_right_windows,
        maximum_context_characters=config.maximum_context_characters,
        shuffle_seed=shuffle_seed,
    )
    coverages = _coverage_attribution(first_pass.segments)
    window_sets = tuple(
        _build_window_set(
            first_pass,
            segment,
            contexts[index],
            coverage_ms=coverages[index],
            config=config,
            build_config=build_config,
            local_policy=local_policy,
            proposal_provider=proposal_provider,
            audio_path=audio_path,
        )
        for index, segment in enumerate(first_pass.segments)
    )
    base_paths = _base_document_paths(window_sets, config=config)
    retained = _retained_hypothesis_with_config(window_sets, config=config)
    scored, scorer_source, scorer_profile, scorer_context = _score_document_paths(
        base_paths,
        retained.digest,
        first_pass=first_pass,
        config=config,
        sequence_scorer=sequence_scorer,
        declared_context=declared,
    )
    alternatives = tuple(
        sorted(
            scored,
            key=lambda path: (
                -path.final_score,
                path.changed_window_count,
                path.generated_window_count,
                path.digest,
            ),
        )
    )
    selected = alternatives[0]
    retained_scored = next(
        (path for path in alternatives if path.digest == retained.digest),
        retained,
    )
    has_runner_up = len(alternatives) > 1
    margin = selected.final_score - alternatives[1].final_score if has_runner_up else 0.0
    reasons: list[str] = ["document-joint-beam", "unique-audio-coverage-weighting"]
    if any(receipt.similarity is not None for receipt in selected.overlap_receipts):
        reasons.append("overlap-aware-compatibility")
    status: Literal["accepted", "provisional"] = "accepted"
    if has_runner_up and margin < config.minimum_final_margin:
        status = "provisional"
        reasons.append("low-document-margin")
    if selected.generated_window_count:
        status = "provisional"
        reasons.append("selected-generated-window")
    if sequence_scorer is not None:
        reasons.append("whole-document-context-applied")
    else:
        reasons.append("no-document-scorer")
    decision = DocumentDeliberationDecision(
        selected=selected,
        retained=retained_scored,
        alternatives=alternatives,
        status=status,
        margin=margin,
        reasons=tuple(reasons),
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        config_digest=config.digest,
        local_policy_digest=local_policy.digest,
        context_digest=scorer_context.digest,
        scorer_source=scorer_source,
        scorer_profile_digest=scorer_profile,
    )
    return DocumentDeliberationPlan(
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        window_sets=window_sets,
        decision=decision,
        config_digest=config.digest,
        local_policy_digest=local_policy.digest,
        context_plan_digest=sha256_json([context.digest for context in contexts]),
    )
