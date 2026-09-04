from __future__ import annotations

from pathlib import Path

from semantic_asr.api import TranscriptSegment, utterances_from_segments
from semantic_asr.contracts import (
    CandidateEvidence,
    GateDecision,
    NormalizedTranscript,
    ObservedTranscript,
    RankedCandidate,
    sha256_json,
)
from semantic_asr.global_deliberation import DeliberationPolicy
from semantic_asr.global_scorer import CallableGlobalSequenceScorer, frozen_profile_digest
from semantic_asr.longform import LongformResult, LongformSegment, Window
from semantic_asr.longform_deliberation import (
    DeliberatedLongformResult,
    LongformDeliberationConfig,
    apply_longform_deliberation,
    with_global_deliberation,
)
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


def first_pass() -> LongformResult:
    first_rows = (
        candidate("mata", "レビュー完了まではまたマージしません。", 0.70),
        candidate("mada", "レビュー完了まではまだマージしません。", 0.68),
    )
    first_observed = observed(first_rows, "mata", {"mata": 0.55, "mada": 0.45})
    first_normalized = NormalizedTranscript.attach(
        first_observed,
        text=first_observed.text,
        mode="deterministic",
    )
    second_rows = (candidate("approved", "承認後に統合します。", 0.90),)
    second_observed = observed(second_rows, "approved", {"approved": 1.0})
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
            window=Window(index=1, start_ms=1_000, end_ms=2_000),
            observed=second_observed,
            normalized=second_normalized,
            diagnostics={"topPosterior": 1.0},
        ),
    )
    observed_text = "".join(segment.observed.text for segment in segments)
    normalized_text = "".join(segment.normalized.text for segment in segments)
    evidence_sha256 = sha256_json(
        {
            "sourceAudioSha256": AUDIO,
            "durationMs": 2_000,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in segments],
        }
    )
    return LongformResult(
        source_name="meeting.wav",
        source_audio_sha256=AUDIO,
        duration_ms=2_000,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=segments,
        evidence_sha256=evidence_sha256,
        diagnostics={"provisionalWindowCount": 0},
    )


def scorer() -> CallableGlobalSequenceScorer:
    return CallableGlobalSequenceScorer(
        lambda path, context: (
            1.0
            if "まだ" in "".join(arc.text for arc in path) and "承認後" in context.right_context
            else -1.0
        ),
        source="test-bidirectional-document-scorer",
        profile_digest=frozen_profile_digest(
            "test-document-scorer",
            "r1",
            {"fixture": True},
        ),
    )


def policy(*, minimum_final_margin: float = 0.02) -> DeliberationPolicy:
    return DeliberationPolicy(
        channel_weights=(
            ("first_pass", 0.8),
            ("asr_acoustic", 1.0),
            ("mora_shadow", 0.1),
            ("transition", 0.1),
        ),
        global_context_weight=1.0,
        retention_bonus=0.0,
        maximum_span_audio_regression=0.3,
        maximum_mean_audio_regression=0.3,
        minimum_final_margin=minimum_final_margin,
    )


def test_opt_in_second_pass_changes_contextually_incoherent_window() -> None:
    raw = first_pass()

    result = apply_longform_deliberation(
        raw,
        sequence_scorer=scorer(),
        policy=policy(),
    )

    assert isinstance(result, DeliberatedLongformResult)
    assert result.segments[0].observed.text == "レビュー完了まではまだマージしません。"
    assert result.segments[0].changed
    assert result.segments[0].trace.applied
    assert result.segments[0].diagnostics["topPosterior"] is None
    assert result.segments[0].normalized.observed_evidence_sha256 == (
        result.segments[0].observed.evidence_sha256
    )
    assert result.first_pass.observed_text == raw.observed_text
    assert result.evidence_sha256 != raw.evidence_sha256
    result.verify()


def test_disabled_second_pass_returns_the_exact_first_pass_object() -> None:
    raw = first_pass()

    result = apply_longform_deliberation(
        raw,
        config=LongformDeliberationConfig(enabled=False),
        sequence_scorer=scorer(),
    )

    assert result is raw


def test_provisional_change_is_recorded_but_not_applied_by_default() -> None:
    raw = first_pass()

    result = apply_longform_deliberation(
        raw,
        config=LongformDeliberationConfig(apply_provisional=False),
        sequence_scorer=scorer(),
        policy=policy(minimum_final_margin=10.0),
    )

    assert isinstance(result, DeliberatedLongformResult)
    assert result.segments[0].observed.text == raw.segments[0].observed.text
    assert not result.segments[0].trace.applied
    assert result.segments[0].trace.reason == "provisional-not-applied"
    assert result.diagnostics["globalDeliberation"]["proposedButNotAppliedCount"] == 1


def test_changed_path_does_not_reuse_stale_first_pass_timestamps() -> None:
    result = apply_longform_deliberation(
        first_pass(),
        sequence_scorer=scorer(),
        policy=policy(),
    )
    assert isinstance(result, DeliberatedLongformResult)
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

    assert utterances[0].start_ms == 0
    assert utterances[0].end_ms == 1_000
    assert utterances[0].text == "レビュー完了まではまだマージしません。"


def test_standard_outputs_use_final_observed_text(tmp_path: Path) -> None:
    result = apply_longform_deliberation(
        first_pass(),
        sequence_scorer=scorer(),
        policy=policy(),
    )
    assert isinstance(result, DeliberatedLongformResult)

    outputs = write_outputs(result, tmp_path, formats={"json", "observed", "srt"})

    assert "まだマージ" in Path(outputs["observed_txt"]).read_text(encoding="utf-8")
    assert "まだマージ" in Path(outputs["srt"]).read_text(encoding="utf-8")
    assert "deliberation_evidence_sha256" in Path(outputs["json"]).read_text(encoding="utf-8")


def test_wrapper_runs_first_pass_once_and_preserves_profile_attributes() -> None:
    raw = first_pass()

    class FakeFirstPass:
        runtime_profile_name = "cpu-ja-v1"
        runtime_profile_digest = "c" * 64

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path, **kwargs):
            self.calls += 1
            return raw

    base = FakeFirstPass()
    wrapped = with_global_deliberation(
        base,  # type: ignore[arg-type]
        sequence_scorer=scorer(),
        policy=policy(),
    )

    result = wrapped.transcribe("meeting.wav")

    assert base.calls == 1
    assert wrapped.runtime_profile_name == "cpu-ja-v1"
    assert isinstance(result, DeliberatedLongformResult)
