from dataclasses import replace

import pytest

from semantic_asr.contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript
from semantic_asr.fusion import FusionConfig, fuse_candidates
from semantic_asr.longform import merge_candidates


def fixture_candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence(
            "spoken",
            "昨日学校を行きました",
            acoustic=0.91,
            mora=0.89,
            lexical=0.38,
            preservation=0.96,
            teacher=0.08,
            source="whisper",
        ),
        CandidateEvidence(
            "clean",
            "昨日学校に行きました",
            acoustic=0.62,
            mora=0.58,
            lexical=0.95,
            preservation=0.42,
            teacher=0.92,
            source="whisper",
        ),
        CandidateEvidence(
            "wrong",
            "昨日会社に行きました",
            acoustic=0.31,
            mora=0.22,
            lexical=0.83,
            preservation=0.21,
            teacher=0.71,
            source="whisper",
        ),
    ]


def test_grammar_honeytrap_cannot_replace_observed_speech() -> None:
    ranked = fuse_candidates(fixture_candidates())
    assert ranked[0].candidate.candidate_id == "spoken"
    clean = next(item for item in ranked if item.candidate.candidate_id == "clean")
    assert clean.grammar_honeytrap_penalty > 0
    assert sum(ranked[0].gate.weights.values()) == pytest.approx(1.0)


def test_observed_evidence_is_tamper_evident() -> None:
    ranked = fuse_candidates(fixture_candidates())
    observed = ObservedTranscript.create(
        selected=ranked[0],
        ranked=ranked,
        uncertainty_spans=[],
        source_audio_sha256="a" * 64,
    )
    observed.verify()
    with pytest.raises(ValueError):
        replace(observed, text="昨日学校に行きました").verify()


def test_rank_only_normalization_must_use_exact_candidate_text() -> None:
    ranked = fuse_candidates(fixture_candidates())
    observed = ObservedTranscript.create(selected=ranked[0], ranked=ranked, uncertainty_spans=[])
    normalized = NormalizedTranscript.attach(
        observed,
        text="昨日学校に行きました",
        mode="rank-only",
        selected_candidate_id="clean",
    )
    assert normalized.observed_evidence_sha256 == observed.evidence_sha256
    with pytest.raises(ValueError):
        NormalizedTranscript.attach(
            observed,
            text="新しく発明した文章",
            mode="rank-only",
            selected_candidate_id="clean",
        )


def test_ambiguous_sparse_evidence_becomes_provisional() -> None:
    candidates = [
        CandidateEvidence("a", "東京です", acoustic=0.51),
        CandidateEvidence("b", "東京でした", acoustic=0.50),
    ]
    config = replace(FusionConfig(), minimum_evidence_coverage=0.80, acceptance_posterior=0.80)
    ranked = fuse_candidates(candidates, config)
    assert ranked[0].gate.needs_relisten
    assert ranked[0].gate.abstain
    observed = ObservedTranscript.create(selected=ranked[0], ranked=ranked, uncertainty_spans=[])
    assert observed.decision == "provisional"


def test_clear_candidate_is_accepted() -> None:
    ranked = fuse_candidates(
        [
            CandidateEvidence(
                "a",
                "東京です",
                acoustic=1.0,
                mora=1.0,
                lexical=1.0,
                preservation=1.0,
                cross_model=1.0,
            ),
            CandidateEvidence(
                "b",
                "東京でした",
                acoustic=0.0,
                mora=0.0,
                lexical=0.0,
                preservation=0.0,
                cross_model=0.0,
            ),
        ]
    )
    assert ranked[0].candidate.candidate_id == "a"
    assert not ranked[0].gate.abstain


def test_cross_model_consensus_is_recorded_on_duplicate_text() -> None:
    merged = merge_candidates(
        [CandidateEvidence("w", "同じ文", acoustic=0.7, source="whisper")],
        [CandidateEvidence("q", "同じ文", source="qwen3-asr")],
    )
    assert len(merged) == 1
    assert merged[0].cross_model is not None and merged[0].cross_model >= 0.62
    assert merged[0].source_support == ("qwen3-asr", "whisper")
