"""Opt-in ASR adapter and existing-lattice bridges for unverified span drafts.

This module neither implements nor impersonates an acoustic verifier. Lattice
insertion requires separately produced, pool-bound whole-window evidence for
both the retained text and each proposed text. No transcript is selected here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from .span_candidates import (
    SpanAnchor,
    SpanBudget,
    SpanDecode,
    SpanExpansion,
    SpanWindow,
    generate_span_drafts,
)

if TYPE_CHECKING:
    from .adapters import ASRAdapter, DecodeRequest
    from .contracts import CandidateEvidence
    from .semantic_deliberation import SemanticDeliberationBuild, VerifiedSpanProposal


def run_span_redecode(
    window: SpanWindow,
    anchors: Sequence[SpanAnchor],
    *,
    request: DecodeRequest,
    adapter: ASRAdapter,
    decoder_fingerprint: str,
    budget: SpanBudget | None = None,
    enabled: bool = False,
) -> SpanExpansion:
    """Re-decode exact crops through the canonical ASR adapter, default OFF.

    The supplied fingerprint must cover model/tokenizer/runtime/config identity.
    File hashes identify the source recording bytes, NOT extracted PCM bytes.
    Prompt/hotword hints are cleared; the parent text is never a decoding hint.
    Original crop scores remain only in the decoder-evidence digest. They are
    never copied onto a new full-window hypothesis or used to rank these drafts.
    """
    if type(enabled) is not bool:
        raise TypeError("enabled must be a boolean")
    if not enabled:
        return SpanExpansion(window)

    from .adapters import DecodeRequest
    from .contracts import CandidateEvidence, sha256_json
    from .longform import sha256_file

    if not isinstance(request, DecodeRequest):
        raise TypeError("request must be the canonical DecodeRequest")
    if (request.start_ms, request.end_ms) != (window.start_ms, window.end_ms):
        raise ValueError("decode request must explicitly match the parent audio window")
    anchors = tuple(anchors)
    for anchor in anchors:
        anchor.validate(window)
    budget = budget or SpanBudget()
    if sha256_file(request.audio_path) != window.source_audio_sha256:
        raise ValueError("source recording does not match the parent audio identity")

    def decode(anchor: SpanAnchor):
        crop_request = replace(
            request,
            start_ms=anchor.start_ms,
            end_ms=anchor.end_ms,
            hypotheses=min(request.hypotheses, budget.max_hypotheses_per_span),
            initial_prompt=None,
            hotwords=(),
            return_timestamps=False,
        )
        for candidate in adapter.decode(crop_request):
            if not isinstance(candidate, CandidateEvidence):
                raise TypeError("adapter must return canonical CandidateEvidence records")
            yield SpanDecode(
                candidate.text,
                sha256_json({"candidate": asdict(candidate), "request": asdict(crop_request)}),
            )

    expansion = generate_span_drafts(
        window,
        anchors,
        decode,
        decoder_fingerprint=decoder_fingerprint,
        budget=budget,
        enabled=True,
    )
    if sha256_file(request.audio_path) != window.source_audio_sha256:
        raise ValueError("source recording changed during span re-decoding")
    return expansion


def span_verification_request(expansion: SpanExpansion) -> dict[str, object]:
    """An allowlist for a trusted, independent whole-window acoustic verifier.

    References, naturalness scores, source paths and crop likelihoods are absent.
    Candidate text is a pronunciation hypothesis, not an audio observation.
    """
    return {
        "schema": "span-whole-window-verification-v1",
        "sourceAudioSha256": expansion.window.source_audio_sha256,
        "startMs": expansion.window.start_ms,
        "endMs": expansion.window.end_ms,
        "expansionDigest": expansion.digest,
        "candidates": [
            {"candidateId": "retained", "text": expansion.window.text},
            *(
                {"candidateId": f"draft:{draft.digest}", "text": draft.text}
                for draft in expansion.drafts
            ),
        ],
    }


def build_verified_redecode_lattice(
    expansion: SpanExpansion,
    *,
    pivot: CandidateEvidence,
    verified: Sequence[VerifiedSpanProposal],
    document_id: str,
) -> SemanticDeliberationBuild:
    """Build a single-parent auxiliary lattice with canonical verified proposals.

    This is NOT a replacement for an existing multi-candidate/document lattice.
    It addresses a single-parent candidate-generation experiment while retaining
    the original path. Only a verifier-produced allowlisted subset is inserted.
    No evidence/abstain yields the unchanged parent lattice; no utility is made up.

    Each receipt must carry ``spanVerificationRequestSha256`` (the canonical
    SHA-256 of ``span_verification_request``) and ``verificationScope`` equal to
    ``whole-window`` in metadata. This is a binding contract for a trusted verifier,
    not a cryptographic proof that it evaluated audio or used valid calibration.
    """
    from .contracts import sha256_json
    from .deliberation_evidence import INDEPENDENT_AUDIO_CHANNELS
    from .semantic_deliberation import (
        SemanticDeliberationConfig,
        VerifiedSpanProposal,
        build_semantic_deliberation_lattice,
    )

    if (pivot.candidate_id, pivot.text) != (expansion.window.candidate_id, expansion.window.text):
        raise ValueError("pivot does not match the immutable expansion parent")
    verified = tuple(verified)
    if verified and not expansion.enabled:
        raise ValueError("disabled expansion cannot accept verification receipts")
    payload = span_verification_request(expansion)
    request_digest = sha256_json(payload)
    expected = {"retained": expansion.window.text}
    expected.update({f"draft:{draft.digest}": draft.text for draft in expansion.drafts})
    seen: set[str] = set()
    domains: set[tuple[str, str, str]] = set()
    for proposal in verified:
        if not isinstance(proposal, VerifiedSpanProposal):
            raise TypeError("verifier must return canonical VerifiedSpanProposal records")
        if proposal.proposal_id in seen:
            raise ValueError("duplicate verification proposal ID")
        seen.add(proposal.proposal_id)
        if expected.get(proposal.proposal_id) != proposal.text:
            raise ValueError("verification text/ID is outside the candidate allowlist")
        if proposal.source_audio_sha256 != expansion.window.source_audio_sha256:
            raise ValueError("verification belongs to another source recording")
        if proposal.metadata.get("spanVerificationRequestSha256") != request_digest:
            raise ValueError("verification belongs to another candidate pool or window")
        if proposal.metadata.get("verificationScope") != "whole-window":
            raise ValueError("crop-only scores cannot verify a reconstructed whole window")
        if not proposal.observed_eligible or proposal.origin != "phonetic-proposal":
            raise ValueError("only independently verified phonetic proposals are eligible")
        if proposal.source_candidate_ids:
            raise ValueError("verification must not invent first-pass candidate support")
        if len(proposal.utilities) != 1:
            raise ValueError("require one independent channel, not correlated double votes")
        utility = proposal.utilities[0]
        if utility.channel not in INDEPENDENT_AUDIO_CHANNELS:
            raise ValueError("verification requires an independent audio channel")
        domains.add((utility.channel, utility.source, utility.profile_digest))
    if verified and "retained" not in seen:
        raise ValueError("the retained text needs comparable whole-window verification too")
    if len(domains) > 1:
        raise ValueError("verification channels/sources/calibration profiles must match")
    return build_semantic_deliberation_lattice(
        [pivot],
        posterior={pivot.candidate_id: 1.0},
        pivot_candidate_id=pivot.candidate_id,
        document_id=document_id,
        source_audio_sha256=expansion.window.source_audio_sha256,
        segment_start_ms=expansion.window.start_ms,
        segment_end_ms=expansion.window.end_ms,
        proposals={f"{document_id}:span:0000": verified} if verified else None,
        config=SemanticDeliberationConfig(include_consensus_utilities=bool(verified)),
    )
