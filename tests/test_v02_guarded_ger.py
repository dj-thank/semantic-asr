from __future__ import annotations

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.guarded_ger import (
    AcousticVerificationEvidence,
    GERProposal,
    evaluate_ger_proposal,
    merge_verified_ger_candidates,
)


def _proposal(text: str) -> GERProposal:
    return GERProposal(
        text=text,
        generator="offline-teacher-12b",
        generator_revision="fixture",
        source_candidate_ids=("a", "b"),
        prompt_digest="a" * 64,
    )


def _candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence("a", "料金は3000円です", reading="りょうきんはさんぜんえんです"),
        CandidateEvidence("b", "料金は30000円です", reading="りょうきんはさんまんえんです"),
    ]


def test_uncalibrated_ger_verifier_can_never_admit_observed_candidate() -> None:
    result = evaluate_ger_proposal(
        _proposal("料金は3000円でした"),
        _candidates(),
        AcousticVerificationEvidence(
            verifier="fixture-verifier",
            acoustic_compatibility=0.99,
            alignment_coverage=0.99,
            mora_compatibility=0.99,
            calibrated=False,
        ),
    )
    assert result.decision == "rejected"
    assert "uncalibrated-acoustic-verifier" in result.reasons
    assert result.candidate is None


def test_calibrated_acoustically_supported_proposal_becomes_candidate_only() -> None:
    result = evaluate_ger_proposal(
        _proposal("料金は3000円でした"),
        _candidates(),
        AcousticVerificationEvidence(
            verifier="fixture-verifier",
            acoustic_compatibility=0.94,
            alignment_coverage=0.96,
            mora_compatibility=0.93,
            calibrated=True,
            calibration_digest="b" * 64,
        ),
    )
    assert result.decision == "accepted-candidate"
    assert result.candidate is not None
    assert result.candidate.metadata["observedEligible"] is True
    assert result.candidate.metadata["generator"] == "offline-teacher-12b"
    assert result.candidate.text == "料金は3000円でした"


def test_meaning_critical_change_requires_stronger_joint_evidence() -> None:
    result = evaluate_ger_proposal(
        _proposal("料金は3001円です"),
        _candidates(),
        AcousticVerificationEvidence(
            verifier="fixture-verifier",
            acoustic_compatibility=0.84,
            alignment_coverage=0.90,
            mora_compatibility=0.84,
            calibrated=True,
            calibration_digest="c" * 64,
        ),
    )
    assert result.critical_change
    assert result.decision != "accepted-candidate"
    assert "meaning-critical-change" in result.reasons


def test_only_accepted_ger_candidates_enter_merged_pool() -> None:
    accepted = evaluate_ger_proposal(
        _proposal("料金は3000円でした"),
        _candidates(),
        AcousticVerificationEvidence(
            verifier="fixture-verifier",
            acoustic_compatibility=0.95,
            alignment_coverage=0.95,
            mora_compatibility=0.95,
            calibrated=True,
            calibration_digest="d" * 64,
        ),
    )
    rejected = evaluate_ger_proposal(
        _proposal("明日は北海道で一億円を払います"),
        _candidates(),
        AcousticVerificationEvidence(
            verifier="fixture-verifier",
            acoustic_compatibility=0.40,
            alignment_coverage=0.30,
            mora_compatibility=0.20,
            calibrated=True,
            calibration_digest="e" * 64,
        ),
    )
    merged = merge_verified_ger_candidates(_candidates(), [accepted, rejected])
    texts = {candidate.text for candidate in merged}
    assert "料金は3000円でした" in texts
    assert "明日は北海道で一億円を払います" not in texts
