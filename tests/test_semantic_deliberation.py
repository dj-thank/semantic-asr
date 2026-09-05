from __future__ import annotations

import math

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.deliberation_evidence import BoundedUtility
from semantic_asr.semantic_deliberation import (
    VerifiedSpanProposal,
    build_semantic_deliberation_lattice,
)

AUDIO = "a" * 64
PROFILE = "1" * 64
INPUT = "2" * 64


def candidate(candidate_id: str, text: str, score: float) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        text=text,
        acoustic=score,
        mora=score - 0.05,
        lexical=score - 0.10,
        preservation=score - 0.02,
        cross_model=score - 0.08,
        source="test-asr",
    )


def utility(channel: str, value: float) -> BoundedUtility:
    return BoundedUtility(
        channel=channel,  # type: ignore[arg-type]
        value=value,
        source=f"test-{channel}",
        profile_digest=PROFILE,
        input_digest=INPUT,
    )


def candidates() -> tuple[CandidateEvidence, ...]:
    return (
        candidate("pivot", "私はまた行く", 0.70),
        candidate("mada", "私はまだ行く", 0.67),
        candidate("delete", "私は行く", 0.45),
        candidate("comma", "私は、また行く", 0.40),
    )


def build(**kwargs):
    return build_semantic_deliberation_lattice(
        candidates(),
        posterior={"pivot": 0.40, "mada": 0.35, "delete": 0.15, "comma": 0.10},
        pivot_candidate_id="pivot",
        document_id="doc-window-0",
        source_audio_sha256=AUDIO,
        segment_start_ms=0,
        segment_end_ms=1_000,
        **kwargs,
    )


def test_exact_projection_reconstructs_every_whole_candidate() -> None:
    result = build()

    assert {row.candidate_id: row.text for row in result.projections} == {
        "pivot": "私はまた行く",
        "mada": "私はまだ行く",
        "delete": "私は行く",
        "comma": "私は、また行く",
    }
    assert "".join(span.retained_arc.text for span in result.lattice.spans) == "私はまた行く"
    assert len(result.lattice.source_paths) == 4
    assert math.isclose(sum(row.posterior for row in result.lattice.source_paths), 1.0)
    assert all(
        arc.source_audio_sha256 == AUDIO for span in result.lattice.spans for arc in span.arcs
    )


def test_deletion_is_an_explicit_epsilon_arc() -> None:
    result = build()
    epsilon = [
        arc
        for span in result.lattice.spans
        for arc in span.arcs
        if "delete" in arc.source_candidate_ids and not arc.text
    ]

    assert epsilon
    assert all(arc.is_epsilon for arc in epsilon)


def test_projected_factor_budget_is_finite_and_mora_is_not_independent() -> None:
    result = build()
    active_spans = [
        span for span in result.lattice.spans if float(span.metadata["factorWeight"]) > 0.0
    ]

    assert math.isclose(
        sum(float(span.metadata["factorWeight"]) for span in active_spans),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    channels = {
        utility.channel for span in active_spans for arc in span.arcs for utility in arc.utilities
    }
    assert "mora_shadow" in channels
    assert "mora" not in channels
    for channel in channels:
        if channel == "transition":
            continue
        factors = {
            utility.factor_weight
            for span in active_spans
            for arc in span.arcs
            for utility in arc.utilities
            if utility.channel == channel
        }
        assert factors.issubset({float(span.metadata["factorWeight"]) for span in active_spans})


def test_candidate_derived_mora_cannot_authenticate_a_generated_proposal() -> None:
    with pytest.raises(ValueError, match="phone, mora, or discrete-unit"):
        VerifiedSpanProposal(
            proposal_id="unsafe",
            text="なお",
            utilities=(utility("mora_shadow", 0.9),),
            source_audio_sha256=AUDIO,
        )


def test_verified_proposal_is_audio_bound_and_rebased_to_span_factor() -> None:
    initial = build()
    target = next(span for span in initial.lattice.spans if bool(span.metadata["isContradiction"]))
    proposal = VerifiedSpanProposal(
        proposal_id="phonetic-nao",
        text="なお",
        utilities=(utility("phone", 0.8),),
        source_audio_sha256=AUDIO,
    )

    result = build(proposals={target.span_id: (proposal,)})
    rebuilt = next(span for span in result.lattice.spans if span.span_id == target.span_id)
    arc = next(arc for arc in rebuilt.arcs if arc.text == "なお")

    assert arc.origin == "phonetic-proposal"
    assert arc.source_audio_sha256 == AUDIO
    assert arc.independent_audio_channels == {"phone"}
    assert arc.utilities[0].factor_weight == target.metadata["factorWeight"]


def test_wrong_audio_proposal_fails_closed() -> None:
    initial = build()
    target = next(span for span in initial.lattice.spans if bool(span.metadata["isContradiction"]))
    proposal = VerifiedSpanProposal(
        proposal_id="wrong-audio",
        text="なお",
        utilities=(utility("phone", 0.8),),
        source_audio_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="different source audio"):
        build(proposals={target.span_id: (proposal,)})


def test_transition_factors_have_one_total_boundary_budget() -> None:
    result = build()
    factors = {row.utility.factor_weight for row in result.lattice.transitions}

    assert factors == {1.0 / (len(result.lattice.spans) - 1)}
