from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from .candidate_pool import merge_candidate_pools
from .contracts import CandidateEvidence, canonical_json
from .mbr import critical_units, mora_loss, surface_loss

ProposalDecision = Literal["accepted-candidate", "provisional-candidate", "rejected"]


@dataclass(frozen=True, slots=True)
class GERProposal:
    text: str
    generator: str
    generator_revision: str | None
    source_candidate_ids: tuple[str, ...]
    prompt_digest: str
    proposal_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.generator:
            raise ValueError("GER proposal text and generator are required")
        if not self.source_candidate_ids:
            raise ValueError("GER proposal requires source candidate IDs")
        if len(set(self.source_candidate_ids)) != len(self.source_candidate_ids):
            raise ValueError("GER source candidate IDs must be unique")
        if len(self.prompt_digest) != 64:
            raise ValueError("GER prompt digest must be SHA-256 hex")

    @property
    def digest(self) -> str:
        payload = {
            "text": self.text,
            "generator": self.generator,
            "generatorRevision": self.generator_revision,
            "sourceCandidateIds": self.source_candidate_ids,
            "promptDigest": self.prompt_digest,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def candidate_id(self) -> str:
        return self.proposal_id or f"ger:{self.digest[:20]}"


@dataclass(frozen=True, slots=True)
class AcousticVerificationEvidence:
    verifier: str
    acoustic_compatibility: float
    alignment_coverage: float
    mora_compatibility: float
    verifier_revision: str | None = None
    calibrated: bool = False
    calibration_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.verifier:
            raise ValueError("acoustic verifier name is required")
        for name, value in (
            ("acoustic_compatibility", self.acoustic_compatibility),
            ("alignment_coverage", self.alignment_coverage),
            ("mora_compatibility", self.mora_compatibility),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0, 1]")
        if self.calibration_digest and not self.calibrated:
            raise ValueError("calibration digest requires calibrated evidence")
        if self.calibrated and not self.calibration_digest:
            raise ValueError("calibrated verifier evidence requires a digest")


@dataclass(frozen=True, slots=True)
class GuardedGERConfig:
    minimum_acoustic_compatibility: float = 0.72
    minimum_alignment_coverage: float = 0.82
    minimum_mora_compatibility: float = 0.75
    minimum_joint_score: float = 0.76
    critical_change_extra: float = 0.10
    maximum_surface_distance: float = 0.55
    maximum_mora_distance: float = 0.50
    provisional_margin: float = 0.08

    def __post_init__(self) -> None:
        for value in (
            self.minimum_acoustic_compatibility,
            self.minimum_alignment_coverage,
            self.minimum_mora_compatibility,
            self.minimum_joint_score,
            self.critical_change_extra,
            self.maximum_surface_distance,
            self.maximum_mora_distance,
            self.provisional_margin,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("GER thresholds must be finite in [0, 1]")


@dataclass(frozen=True, slots=True)
class GERGateResult:
    proposal: GERProposal
    decision: ProposalDecision
    joint_score: float
    nearest_candidate_id: str
    surface_distance: float
    mora_distance: float
    critical_change: bool
    reasons: tuple[str, ...]
    candidate: CandidateEvidence | None = None


def _joint_score(evidence: AcousticVerificationEvidence) -> float:
    return (
        0.50 * evidence.acoustic_compatibility
        + 0.28 * evidence.mora_compatibility
        + 0.22 * evidence.alignment_coverage
    )


def _nearest_candidate(
    proposal: GERProposal, candidates: Sequence[CandidateEvidence]
) -> tuple[CandidateEvidence, float, float]:
    proposed = CandidateEvidence(proposal.candidate_id, proposal.text)
    rows = [
        (
            candidate,
            surface_loss(proposed, candidate),
            mora_loss(proposed, candidate),
        )
        for candidate in candidates
    ]
    return min(
        rows,
        key=lambda row: (
            0.55 * row[1] + 0.45 * row[2],
            row[0].candidate_id,
        ),
    )


def evaluate_ger_proposal(
    proposal: GERProposal,
    existing_candidates: Sequence[CandidateEvidence],
    evidence: AcousticVerificationEvidence,
    *,
    config: GuardedGERConfig | None = None,
) -> GERGateResult:
    config = config or GuardedGERConfig()
    if not existing_candidates:
        raise ValueError("GER verification requires existing acoustic candidates")
    identifiers = {candidate.candidate_id for candidate in existing_candidates}
    if not set(proposal.source_candidate_ids).issubset(identifiers):
        raise ValueError("GER proposal references candidates outside observed evidence")
    nearest, surface_distance, mora_distance_value = _nearest_candidate(
        proposal, existing_candidates
    )
    critical_change = critical_units(proposal.text) != critical_units(nearest.text)
    joint = _joint_score(evidence)
    threshold = min(
        1.0,
        config.minimum_joint_score + (config.critical_change_extra if critical_change else 0.0),
    )
    reasons: list[str] = []
    if not evidence.calibrated:
        reasons.append("uncalibrated-acoustic-verifier")
    if evidence.acoustic_compatibility < config.minimum_acoustic_compatibility:
        reasons.append("low-acoustic-compatibility")
    if evidence.alignment_coverage < config.minimum_alignment_coverage:
        reasons.append("low-alignment-coverage")
    if evidence.mora_compatibility < config.minimum_mora_compatibility:
        reasons.append("low-mora-compatibility")
    if joint < threshold:
        reasons.append("low-joint-verification")
    if surface_distance > config.maximum_surface_distance:
        reasons.append("excessive-surface-distance")
    if mora_distance_value > config.maximum_mora_distance:
        reasons.append("excessive-mora-distance")
    if critical_change:
        reasons.append("meaning-critical-change")

    hard_failures = {
        "uncalibrated-acoustic-verifier",
        "low-acoustic-compatibility",
        "low-alignment-coverage",
        "low-mora-compatibility",
        "excessive-surface-distance",
        "excessive-mora-distance",
    }
    if hard_failures.intersection(reasons):
        decision: ProposalDecision = "rejected"
    elif joint >= threshold:
        decision = "accepted-candidate"
    elif joint >= max(0.0, threshold - config.provisional_margin):
        decision = "provisional-candidate"
    else:
        decision = "rejected"

    candidate = None
    if decision != "rejected":
        candidate = CandidateEvidence(
            candidate_id=proposal.candidate_id,
            text=proposal.text.strip(),
            acoustic=evidence.acoustic_compatibility,
            mora=evidence.mora_compatibility,
            cross_model=None,
            source="guarded-ger-proposal",
            metadata={
                "adapter": "guarded-ger-proposal",
                "proposalDigest": proposal.digest,
                "generator": proposal.generator,
                "generatorRevision": proposal.generator_revision,
                "sourceCandidateIds": list(proposal.source_candidate_ids),
                "promptDigest": proposal.prompt_digest,
                "acousticVerifier": evidence.verifier,
                "acousticVerifierRevision": evidence.verifier_revision,
                "acousticVerifierCalibrated": evidence.calibrated,
                "acousticVerifierCalibrationDigest": evidence.calibration_digest,
                "alignmentCoverage": evidence.alignment_coverage,
                "jointVerificationScore": joint,
                "gerDecision": decision,
                "observedEligible": decision == "accepted-candidate",
                "criticalChange": critical_change,
            },
        )
    return GERGateResult(
        proposal=proposal,
        decision=decision,
        joint_score=joint,
        nearest_candidate_id=nearest.candidate_id,
        surface_distance=surface_distance,
        mora_distance=mora_distance_value,
        critical_change=critical_change,
        reasons=tuple(dict.fromkeys(reasons)),
        candidate=candidate,
    )


def merge_verified_ger_candidates(
    existing_candidates: Sequence[CandidateEvidence],
    results: Sequence[GERGateResult],
) -> list[CandidateEvidence]:
    accepted = [
        result.candidate
        for result in results
        if result.decision == "accepted-candidate" and result.candidate is not None
    ]
    return merge_candidate_pools(
        existing_candidates,
        [candidate for candidate in accepted if candidate is not None],
        id_prefix="ger-verified",
    )


def mark_provisional_only(candidate: CandidateEvidence) -> CandidateEvidence:
    metadata = dict(candidate.metadata)
    metadata["observedEligible"] = False
    metadata["gerDecision"] = "provisional-candidate"
    return replace(candidate, metadata=metadata)
