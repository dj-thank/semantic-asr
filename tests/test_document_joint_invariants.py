from __future__ import annotations

from semantic_asr.contracts import (
    CandidateEvidence,
    GateDecision,
    NormalizedTranscript,
    ObservedTranscript,
    RankedCandidate,
    sha256_json,
)
from semantic_asr.document_joint_deliberation import (
    DocumentDeliberatedResult,
    DocumentDeliberationConfig,
    DocumentPassThroughResult,
    apply_joint_document_deliberation,
)
from semantic_asr.global_deliberation import DeliberationPolicy
from semantic_asr.global_scorer import CallableGlobalSequenceScorer
from semantic_asr.longform import LongformResult, LongformSegment, Window

AUDIO = "a" * 64


def candidate(identifier: str, text: str, acoustic: float) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=identifier,
        text=text,
        acoustic=acoustic,
        mora=acoustic,
        lexical=acoustic,
        preservation=acoustic,
        cross_model=acoustic,
        source="test",
    )


def observed(rows, selected_id, posterior):
    gate = GateDecision(
        weights={"acoustic": 1.0},
        posterior=posterior,
        entropy=0.1,
        disagreement=0.0,
        evidence_coverage=1.0,
        selective_risk=0.1,
        needs_relisten=False,
        abstain=False,
    )
    ranked = tuple(
        RankedCandidate(
            candidate=row,
            final_score=posterior[row.candidate_id],
            posterior=posterior[row.candidate_id],
            calibrated_scores={"acoustic": posterior[row.candidate_id]},
            gate=gate,
        )
        for row in rows
    )
    selected = next(row for row in ranked if row.candidate.candidate_id == selected_id)
    return ObservedTranscript.create(
        selected=selected,
        ranked=ranked,
        uncertainty_spans=(),
        source_audio_sha256=AUDIO,
    )


def one_window_result() -> LongformResult:
    rows = (
        candidate("retained", "この変更はまた保留です。", 0.8),
        candidate("alternative", "この変更はまだ保留です。", 0.75),
    )
    obs = observed(rows, "retained", {"retained": 0.6, "alternative": 0.4})
    normalized = NormalizedTranscript.attach(obs, text=obs.text, mode="deterministic")
    segment = LongformSegment(
        window=Window(index=0, start_ms=0, end_ms=1_000),
        observed=obs,
        normalized=normalized,
        diagnostics={"topPosterior": 0.6},
    )
    evidence = sha256_json(
        {
            "sourceAudioSha256": AUDIO,
            "durationMs": 1_000,
            "observedText": obs.text,
            "normalizedText": normalized.text,
            "segmentEvidence": [obs.evidence_sha256],
        }
    )
    return LongformResult(
        source_name="one.wav",
        source_audio_sha256=AUDIO,
        duration_ms=1_000,
        observed_text=obs.text,
        normalized_text=normalized.text,
        segments=(segment,),
        evidence_sha256=evidence,
        diagnostics={},
    )


def policy() -> DeliberationPolicy:
    return DeliberationPolicy(
        channel_weights=(("first_pass", 0.8), ("asr_acoustic", 1.0)),
        global_context_weight=0.0,
        retention_bonus=0.0,
        maximum_span_audio_regression=0.5,
        maximum_mean_audio_regression=0.5,
        minimum_final_margin=0.0,
    )


def config(**updates):
    values = {
        "local_paths_per_window": 4,
        "document_beam_size": 8,
        "global_document_weight": 1.0,
        "maximum_document_audio_regression": 0.5,
        "maximum_changed_windows": 1,
        "maximum_changed_ratio": 1.0,
        "minimum_document_margin": 0.01,
    }
    values.update(updates)
    return DocumentDeliberationConfig(**values)


def scorer(prefer_alternative: bool):
    return CallableGlobalSequenceScorer(
        lambda path, context: (
            1.0
            if prefer_alternative and "まだ" in "".join(arc.text for arc in path)
            else 0.0
        ),
        source="identity-test",
        profile_digest="1" * 64,
    )


def test_scoring_metadata_does_not_make_retained_path_look_changed() -> None:
    raw = one_window_result()

    result = apply_joint_document_deliberation(
        raw,
        config=config(),
        local_policy=policy(),
        document_scorer=scorer(prefer_alternative=False),
    )

    assert isinstance(result, DocumentDeliberatedResult)
    assert not result.document_decision.applied
    assert result.observed_text == raw.observed_text
    assert "retained-first-pass-document" in result.document_decision.reasons


def test_provisional_alternative_preserves_exact_first_pass_output() -> None:
    raw = one_window_result()

    result = apply_joint_document_deliberation(
        raw,
        config=config(minimum_document_margin=10.0, apply_provisional=False),
        local_policy=policy(),
        document_scorer=scorer(prefer_alternative=True),
    )

    assert isinstance(result, DocumentDeliberatedResult)
    assert result.document_decision.status == "provisional"
    assert not result.document_decision.applied
    assert result.observed_text == raw.observed_text
    assert result.segments[0].observed.text == raw.segments[0].observed.text


def test_scorer_failure_returns_auditable_pass_through() -> None:
    class Broken:
        source = "broken"
        profile_digest = "2" * 64

        def score(self, path, *, context):
            raise RuntimeError("broken scorer")

    raw = one_window_result()
    result = apply_joint_document_deliberation(
        raw,
        config=config(),
        local_policy=policy(),
        document_scorer=Broken(),
    )

    assert isinstance(result, DocumentPassThroughResult)
    assert result.observed_text == raw.observed_text
    assert result.failure.error_type == "RuntimeError"
    assert result.evidence_sha256 != raw.evidence_sha256
    result.verify()
