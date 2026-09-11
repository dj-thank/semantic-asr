from __future__ import annotations

from pathlib import Path

import pytest

from semantic_asr.api import TranscriptSegment, utterances_from_segments
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
    OverlapPolicy,
    apply_joint_document_deliberation,
    resolve_window_overlap,
    with_joint_document_deliberation,
)
from semantic_asr.global_deliberation import DeliberationPolicy
from semantic_asr.global_scorer import CallableGlobalSequenceScorer, frozen_profile_digest
from semantic_asr.longform import LongformResult, LongformSegment, Window
from semantic_asr.outputs import write_outputs

AUDIO = "a" * 64


def candidate(candidate_id: str, text: str, acoustic: float) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=candidate_id,
        text=text,
        acoustic=acoustic,
        mora=acoustic - 0.02,
        lexical=acoustic - 0.04,
        preservation=acoustic - 0.01,
        cross_model=acoustic - 0.03,
        source="test-asr",
    )


def observed(
    rows: tuple[CandidateEvidence, ...],
    selected_id: str,
    posterior: dict[str, float],
) -> ObservedTranscript:
    gate = GateDecision(
        weights={"acoustic": 1.0},
        posterior=posterior,
        entropy=0.5,
        disagreement=0.0,
        evidence_coverage=1.0,
        selective_risk=0.1,
        needs_relisten=False,
        abstain=False,
    )
    ranked = [
        RankedCandidate(
            candidate=row,
            final_score=posterior[row.candidate_id],
            posterior=posterior[row.candidate_id],
            calibrated_scores={"acoustic": posterior[row.candidate_id]},
            gate=gate,
        )
        for row in sorted(rows, key=lambda item: item.candidate_id != selected_id)
    ]
    selected = next(row for row in ranked if row.candidate.candidate_id == selected_id)
    return ObservedTranscript.create(
        selected=selected,
        ranked=ranked,
        uncertainty_spans=[],
        source_audio_sha256=AUDIO,
    )


def first_pass(*, alternative_acoustic: float = 0.68) -> LongformResult:
    first_rows = (
        candidate("mata", "計画はまた保留です。", 0.70),
        candidate("mada", "計画はまだ保留です。", alternative_acoustic),
    )
    first_observed = observed(first_rows, "mata", {"mata": 0.55, "mada": 0.45})
    first_normalized = NormalizedTranscript.attach(
        first_observed,
        text=first_observed.text,
        mode="deterministic",
    )
    second_rows = (
        candidate("execute", "保留です。承認後に実行します。", 0.90),
        candidate("cancel", "保留です。承認後に中止します。", 0.70),
    )
    second_observed = observed(second_rows, "execute", {"execute": 0.80, "cancel": 0.20})
    second_normalized = NormalizedTranscript.attach(
        second_observed,
        text=second_observed.text,
        mode="deterministic",
    )
    segments = (
        LongformSegment(
            window=Window(index=0, start_ms=0, end_ms=1_000),
            observed=first_observed,
            normalized=first_normalized,
            diagnostics={"topPosterior": 0.55},
        ),
        LongformSegment(
            window=Window(index=1, start_ms=700, end_ms=1_700),
            observed=second_observed,
            normalized=second_normalized,
            diagnostics={"topPosterior": 0.80},
        ),
    )
    observed_text = "".join(segment.observed.text for segment in segments)
    normalized_text = "".join(segment.normalized.text for segment in segments)
    evidence_sha256 = sha256_json(
        {
            "sourceAudioSha256": AUDIO,
            "durationMs": 1_700,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in segments],
        }
    )
    return LongformResult(
        source_name="meeting.wav",
        source_audio_sha256=AUDIO,
        duration_ms=1_700,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=segments,
        evidence_sha256=evidence_sha256,
        diagnostics={"provisionalWindowCount": 0},
    )


def document_scorer() -> CallableGlobalSequenceScorer:
    def score(path, context):
        text = "".join(arc.text for arc in path)
        if "まだ" in text and "実行します" in text and text.count("保留です") == 1:
            return 1.0
        return -1.0

    return CallableGlobalSequenceScorer(
        score,
        source="test-complete-document-scorer",
        profile_digest=frozen_profile_digest(
            "test-complete-document-scorer",
            "r1",
            {"fixture": True},
        ),
    )


def local_policy(*, maximum_span_audio_regression: float = 0.30) -> DeliberationPolicy:
    return DeliberationPolicy(
        channel_weights=(
            ("first_pass", 0.7),
            ("asr_acoustic", 1.0),
            ("mora_shadow", 0.1),
            ("transition", 0.1),
        ),
        global_context_weight=0.0,
        retention_bonus=0.0,
        maximum_span_audio_regression=maximum_span_audio_regression,
        maximum_mean_audio_regression=maximum_span_audio_regression,
        minimum_final_margin=0.0,
    )


def document_config(**overrides):
    values = {
        "local_paths_per_window": 4,
        "document_beam_size": 16,
        "overlap_weight": 0.5,
        "global_document_weight": 2.0,
        "maximum_document_audio_regression": 0.30,
        "maximum_changed_windows": 2,
        "maximum_changed_ratio": 1.0,
        "minimum_document_margin": 0.01,
    }
    values.update(overrides)
    return DocumentDeliberationConfig(**values)


def test_joint_beam_selects_coherent_windows_and_deduplicates_overlap() -> None:
    raw = first_pass()

    result = apply_joint_document_deliberation(
        raw,
        config=document_config(),
        local_policy=local_policy(),
        document_scorer=document_scorer(),
    )

    assert isinstance(result, DocumentDeliberatedResult)
    assert result.document_decision.applied
    assert result.document_decision.status == "accepted"
    assert result.segments[0].observed.full_window_text == "計画はまだ保留です。"
    assert result.segments[1].observed.full_window_text == "保留です。承認後に実行します。"
    assert result.segments[1].observed.text == "承認後に実行します。"
    assert result.observed_text.count("保留です") == 1
    assert result.segments[1].observed.overlap_receipt.method == "exact-suffix-prefix"
    assert result.segments[0].observed.candidates == raw.segments[0].observed.candidates
    assert result.segments[0].diagnostics["topPosterior"] is None
    result.verify()


def test_document_scorer_sees_overlap_resolved_emission_text() -> None:
    seen = []

    def score(path, context):
        text = "".join(arc.text for arc in path)
        seen.append(text)
        return 0.0

    scorer = CallableGlobalSequenceScorer(
        score,
        source="capture-document-text",
        profile_digest="1" * 64,
    )
    result = apply_joint_document_deliberation(
        first_pass(),
        config=document_config(global_document_weight=0.0),
        local_policy=local_policy(),
        document_scorer=scorer,
    )

    assert isinstance(result, DocumentDeliberatedResult)
    assert seen
    assert any(text.count("保留です") == 1 for text in seen)


def test_large_audio_regression_blocks_contextual_override() -> None:
    raw = first_pass(alternative_acoustic=-5.0)

    result = apply_joint_document_deliberation(
        raw,
        config=document_config(maximum_document_audio_regression=0.05),
        local_policy=local_policy(maximum_span_audio_regression=0.05),
        document_scorer=document_scorer(),
    )

    assert isinstance(result, DocumentDeliberatedResult)
    assert "また" in result.observed_text
    assert "まだ" not in result.observed_text


def test_ambiguous_overlap_marks_document_provisional_and_is_not_applied() -> None:
    emitted, receipt = resolve_window_overlap(
        "今日は晴れです。",
        "今日は雨です。続けます。",
        left_window_index=0,
        right_window_index=1,
        overlap_ms=400,
        policy=OverlapPolicy(ambiguous_similarity_threshold=0.45),
    )

    assert emitted == "今日は雨です。続けます。"
    assert receipt.method == "ambiguous-conflict"
    assert receipt.ambiguous


def test_overlap_resolution_is_deterministic_and_hash_bound() -> None:
    arguments = {
        "left_text": "説明を続けます。",
        "right_text": "説明を続けます。次です。",
        "left_window_index": 0,
        "right_window_index": 1,
        "overlap_ms": 300,
        "policy": OverlapPolicy(),
    }

    first = resolve_window_overlap(**arguments)
    second = resolve_window_overlap(**arguments)

    assert first == second
    assert first[0] == "次です。"
    assert first[1].digest == second[1].digest


def test_document_output_uses_emitted_segments_without_duplicate_subtitles(
    tmp_path: Path,
) -> None:
    result = apply_joint_document_deliberation(
        first_pass(),
        config=document_config(),
        local_policy=local_policy(),
        document_scorer=document_scorer(),
    )
    assert isinstance(result, DocumentDeliberatedResult)

    outputs = write_outputs(result, tmp_path, formats={"json", "observed", "srt"})

    observed_text = Path(outputs["observed"]).read_text(encoding="utf-8")
    srt_text = Path(outputs["srt"]).read_text(encoding="utf-8")
    assert observed_text.count("保留です") == 1
    assert srt_text.count("保留です") == 1
    assert "document_decision_digest" in Path(outputs["json"]).read_text(encoding="utf-8")


def test_changed_document_does_not_reuse_candidate_timestamp_spans() -> None:
    result = apply_joint_document_deliberation(
        first_pass(),
        config=document_config(),
        local_policy=local_policy(),
        document_scorer=document_scorer(),
    )
    assert isinstance(result, DocumentDeliberatedResult)
    facade = tuple(
        TranscriptSegment(
            index=index + 1,
            start_ms=segment.window.start_ms,
            end_ms=segment.window.end_ms,
            observed=segment.observed.text,
            normalized=segment.normalized.text,
            status=segment.observed.decision,
            confidence=None,
        )
        for index, segment in enumerate(result.segments)
    )

    utterances = utterances_from_segments(result, facade)

    assert utterances[0].text == result.segments[0].observed.text
    assert utterances[1].text == result.segments[1].observed.text


def test_wrapper_runs_first_pass_exactly_once() -> None:
    raw = first_pass()

    class FakeFirstPass:
        runtime_profile_name = "cpu-ja-v1"
        runtime_profile_digest = "c" * 64

        def __init__(self):
            self.calls = 0

        def transcribe(self, audio_path, **kwargs):
            self.calls += 1
            return raw

    base = FakeFirstPass()
    wrapped = with_joint_document_deliberation(
        base,  # type: ignore[arg-type]
        document_scorer=document_scorer(),
        config=document_config(),
        local_policy=local_policy(),
    )

    result = wrapped.transcribe("meeting.wav")

    assert base.calls == 1
    assert wrapped.runtime_profile_name == "cpu-ja-v1"
    assert isinstance(result, DocumentDeliberatedResult)


def test_missing_required_document_scorer_fails_before_work() -> None:
    with pytest.raises(ValueError, match="requires an explicit document scorer"):
        apply_joint_document_deliberation(first_pass(), document_scorer=None)
