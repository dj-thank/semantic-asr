from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_asr.contracts import (
    CandidateEvidence,
    GateDecision,
    NormalizedTranscript,
    ObservedTranscript,
    RankedCandidate,
)
from semantic_asr.deliberation_evidence import BoundedUtility
from semantic_asr.deliberation_lattice import DocumentContext
from semantic_asr.document_deliberation import (
    DocumentBeamConfig,
    DocumentDeliberationPlan,
    FrozenWindowContext,
    _coverage_attribution,
    apply_document_deliberation,
    build_frozen_window_contexts,
    plan_document_deliberation,
)
from semantic_asr.global_deliberation import DeliberationPolicy
from semantic_asr.global_scorer import (
    CallableGlobalSequenceScorer,
    GlobalPathScore,
    frozen_profile_digest,
)
from semantic_asr.longform import LongformResult, LongformSegment, Window
from semantic_asr.longform_deliberation import DeliberatedLongformResult
from semantic_asr.semantic_deliberation import VerifiedSpanProposal

AUDIO = "a" * 64
PROFILE = "1" * 64
INPUT = "2" * 64


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
    ranked = tuple(
        RankedCandidate(
            candidate=row,
            final_score=posterior[row.candidate_id],
            posterior=posterior[row.candidate_id],
            calibrated_scores={"acoustic": posterior[row.candidate_id]},
            gate=gate,
        )
        for row in sorted(rows, key=lambda item: item.candidate_id != selected_id)
    )
    selected = next(row for row in ranked if row.candidate.candidate_id == selected_id)
    return ObservedTranscript.create(
        selected=selected,
        ranked=ranked,
        uncertainty_spans=[],
        source_audio_sha256=AUDIO,
    )


def segment(
    index: int,
    start_ms: int,
    end_ms: int,
    rows: tuple[CandidateEvidence, ...],
    selected_id: str,
    posterior: dict[str, float],
) -> LongformSegment:
    raw = observed(rows, selected_id, posterior)
    return LongformSegment(
        window=Window(index=index, start_ms=start_ms, end_ms=end_ms),
        observed=raw,
        normalized=NormalizedTranscript.attach(
            raw,
            text=raw.text,
            mode="deterministic",
        ),
        diagnostics={"topPosterior": max(posterior.values())},
    )


def longform(segments: tuple[LongformSegment, ...]) -> LongformResult:
    return LongformResult.create(
        source_name="meeting.wav",
        source_audio_sha256=AUDIO,
        duration_ms=max(row.window.end_ms for row in segments),
        segments=segments,
        diagnostics={"fixture": True},
    )


def local_policy() -> DeliberationPolicy:
    return DeliberationPolicy(
        channel_weights=(
            ("first_pass", 0.8),
            ("asr_acoustic", 1.0),
            ("phone", 0.8),
            ("mora_shadow", 0.1),
            ("transition", 0.1),
        ),
        beam_size=32,
        global_context_weight=0.0,
        retention_bonus=0.0,
        maximum_span_audio_regression=0.5,
        maximum_mean_audio_regression=0.5,
        minimum_final_margin=0.0,
    )


def coherent_fixture() -> LongformResult:
    return longform(
        (
            segment(
                0,
                0,
                1_000,
                (
                    candidate("mata", "レビュー完了まではまたマージしません。", 0.70),
                    candidate("mada", "レビュー完了まではまだマージしません。", 0.68),
                ),
                "mata",
                {"mata": 0.55, "mada": 0.45},
            ),
            segment(
                1,
                1_000,
                2_000,
                (candidate("approved", "承認後に統合します。", 0.90),),
                "approved",
                {"approved": 1.0},
            ),
        )
    )


def whole_document_scorer() -> CallableGlobalSequenceScorer:
    return CallableGlobalSequenceScorer(
        lambda path, context: (
            1.0 if "まだマージ" in path[0].text and "承認後" in path[0].text else -1.0
        ),
        source="test-whole-document-scorer",
        profile_digest=frozen_profile_digest(
            "test-whole-document-scorer",
            "r1",
            {"fixture": True},
        ),
    )


def document_config(**overrides: object) -> DocumentBeamConfig:
    values: dict[str, object] = {
        "local_paths_per_window": 8,
        "beam_size": 32,
        "global_rescore_paths": 32,
        "global_context_weight": 1.0,
        "change_penalty": 0.0,
        "generated_penalty": 0.0,
        "maximum_document_audio_regression": 0.5,
        "minimum_final_margin": 0.01,
    }
    values.update(overrides)
    return DocumentBeamConfig(**values)  # type: ignore[arg-type]


def test_joint_document_beam_selects_contextually_coherent_window_sequence() -> None:
    first_pass = coherent_fixture()

    result = apply_document_deliberation(
        first_pass,
        config=document_config(),
        local_policy=local_policy(),
        sequence_scorer=whole_document_scorer(),
    )

    assert isinstance(result, DeliberatedLongformResult)
    assert "まだマージ" in result.observed_text
    assert result.segments[0].changed
    assert not result.segments[1].changed
    assert result.diagnostics["globalDeliberation"]["mode"] == "document-joint-beam-v1"
    assert result.diagnostics["globalDeliberation"]["changedWindowCount"] == 1
    assert result.first_pass is first_pass
    result.verify()


def test_document_plan_keeps_multiple_local_paths_until_whole_document_scoring() -> None:
    plan = plan_document_deliberation(
        coherent_fixture(),
        config=document_config(),
        local_policy=local_policy(),
        sequence_scorer=whole_document_scorer(),
    )

    assert isinstance(plan, DocumentDeliberationPlan)
    assert len(plan.window_sets[0].options) >= 2
    assert len(plan.decision.alternatives) >= 2
    assert plan.decision.selected.options[0].text != plan.decision.retained.options[0].text
    assert plan.decision.scorer_source == "test-whole-document-scorer"


def overlap_fixture() -> LongformResult:
    return longform(
        (
            segment(
                0,
                0,
                1_000,
                (
                    candidate("school", "今日は学校です", 0.70),
                    candidate("company", "今日は会社です", 0.69),
                ),
                "school",
                {"school": 0.51, "company": 0.49},
            ),
            segment(
                1,
                800,
                1_800,
                (candidate("continuation", "学校です。次へ進みます", 0.90),),
                "continuation",
                {"continuation": 1.0},
            ),
        )
    )


def prefer_company_scorer() -> CallableGlobalSequenceScorer:
    return CallableGlobalSequenceScorer(
        lambda path, context: 1.0 if "会社" in path[0].text else -1.0,
        source="adversarial-company-scorer",
        profile_digest=frozen_profile_digest(
            "adversarial-company-scorer",
            "r1",
            {"fixture": True},
        ),
    )


def test_overlap_regression_guard_blocks_document_scorer_from_breaking_boundary() -> None:
    plan = plan_document_deliberation(
        overlap_fixture(),
        config=document_config(
            maximum_overlap_similarity_regression=0.05,
            overlap_consistency_weight=1.0,
        ),
        local_policy=local_policy(),
        sequence_scorer=prefer_company_scorer(),
    )

    assert "学校" in plan.decision.selected.text
    assert plan.decision.selected.overlap_receipts
    assert all(receipt.compatible for receipt in plan.decision.selected.overlap_receipts)


def test_audio_coverage_is_attributed_once_across_overlapping_windows() -> None:
    fixture = overlap_fixture()

    coverage = _coverage_attribution(fixture.segments)

    assert coverage == pytest.approx((900.0, 900.0))
    assert sum(coverage) == pytest.approx(1_800.0)


def test_nonoverlapping_identical_speech_is_preserved_as_repetition() -> None:
    fixture = longform(
        (
            segment(
                0,
                0,
                500,
                (candidate("yes-0", "はい。", 0.9),),
                "yes-0",
                {"yes-0": 1.0},
            ),
            segment(
                1,
                600,
                1_100,
                (candidate("yes-1", "はい。", 0.9),),
                "yes-1",
                {"yes-1": 1.0},
            ),
        )
    )

    plan = plan_document_deliberation(
        fixture,
        config=document_config(require_sequence_scorer=False),
        local_policy=local_policy(),
        sequence_scorer=None,
    )

    assert plan.decision.selected.text == "はい。はい。"
    assert not plan.decision.selected.overlap_receipts


def context_fixture() -> LongformResult:
    return longform(
        tuple(
            segment(
                index,
                index * 1_000,
                (index + 1) * 1_000,
                (candidate(f"c{index}", text, 0.9),),
                f"c{index}",
                {f"c{index}": 1.0},
            )
            for index, text in enumerate(("一番目。", "二番目。", "三番目。", "四番目。"))
        )
    )


def test_context_control_arms_exclude_target_and_future_from_left_only() -> None:
    fixture = context_fixture()
    none = build_frozen_window_contexts(fixture, arm="none")
    left = build_frozen_window_contexts(fixture, arm="left-only")
    bidirectional = build_frozen_window_contexts(
        fixture,
        arm="bidirectional-offline",
    )
    shuffled = build_frozen_window_contexts(
        fixture,
        arm="shuffled-context",
        shuffle_seed="fixed-seed",
    )

    assert all(not row.source_window_indices for row in none)
    assert all(not row.context.left_context and not row.context.right_context for row in none)
    assert left[2].source_window_indices == (0, 1)
    assert not left[2].context.right_context
    assert all(
        source < row.target_window_index
        for row in left
        for source in row.source_window_indices
    )
    assert bidirectional[1].source_window_indices == (0, 2, 3)
    assert bidirectional[1].context.metadata["usesFutureFirstPass"] is True
    assert all(
        row.target_window_index not in row.source_window_indices
        for row in (*none, *left, *bidirectional, *shuffled)
    )
    assert shuffled == build_frozen_window_contexts(
        fixture,
        arm="shuffled-context",
        shuffle_seed="fixed-seed",
    )
    assert shuffled[1].shuffle_seed_digest is not None


def test_context_receipt_rejects_target_window_injection() -> None:
    fixture = context_fixture()
    row = build_frozen_window_contexts(fixture, arm="left-only")[2]

    with pytest.raises(ValueError, match="target window"):
        FrozenWindowContext(
            target_window_index=row.target_window_index,
            arm=row.arm,
            context=row.context,
            source_window_indices=(*row.source_window_indices, row.target_window_index),
            first_pass_evidence_sha256=row.first_pass_evidence_sha256,
        )


def phone_utility(value: float = 0.9) -> BoundedUtility:
    return BoundedUtility(
        channel="phone",
        value=value,
        source="test-phone",
        profile_digest=PROFILE,
        input_digest=INPUT,
    )


def generated_provider(**kwargs):
    build = kwargs["build"]
    targets = [
        span for span in build.lattice.spans if bool(span.metadata["isContradiction"])
    ]
    if not targets:
        return {}
    target = targets[0]
    return {
        target.span_id: (
            VerifiedSpanProposal(
                proposal_id="phone-nao",
                text="なお",
                utilities=(phone_utility(),),
                source_audio_sha256=kwargs["source_audio_sha256"],
            ),
        )
    }


def prefer_generated_scorer() -> CallableGlobalSequenceScorer:
    return CallableGlobalSequenceScorer(
        lambda path, context: 1.0 if "なお" in path[0].text else -1.0,
        source="prefer-generated",
        profile_digest=frozen_profile_digest(
            "prefer-generated",
            "r1",
            {"fixture": True},
        ),
    )


def test_generated_document_path_remains_provisional_and_unapplied_by_default() -> None:
    first_pass = coherent_fixture()

    result = apply_document_deliberation(
        first_pass,
        config=document_config(apply_provisional=False),
        local_policy=local_policy(),
        sequence_scorer=prefer_generated_scorer(),
        proposal_provider=generated_provider,
    )

    assert isinstance(result, DeliberatedLongformResult)
    assert result.observed_text == first_pass.observed_text
    assert result.diagnostics["globalDeliberation"]["documentStatus"] == "provisional"
    assert result.diagnostics["globalDeliberation"]["changedWindowCount"] == 0


class WrongDigestScorer:
    source = "wrong-digest"
    profile_digest = PROFILE

    def score_many(self, paths, *, context):
        return tuple(
            GlobalPathScore(
                value=1.0,
                source=self.source,
                profile_digest=self.profile_digest,
                path_digest="f" * 64,
                context_digest=context.digest,
            )
            for _ in paths
        )

    def score(self, path, *, context):
        return self.score_many((path,), context=context)[0]


def test_document_scorer_digest_mismatch_fails_closed_or_raises() -> None:
    first_pass = coherent_fixture()

    assert (
        apply_document_deliberation(
            first_pass,
            config=document_config(fail_closed_to_first_pass=True),
            local_policy=local_policy(),
            sequence_scorer=WrongDigestScorer(),
        )
        is first_pass
    )
    with pytest.raises(ValueError, match="unknown path digest"):
        apply_document_deliberation(
            first_pass,
            config=document_config(fail_closed_to_first_pass=False),
            local_policy=local_policy(),
            sequence_scorer=WrongDigestScorer(),
        )


def test_tampered_first_pass_is_rejected_before_context_construction() -> None:
    fixture = coherent_fixture()
    tampered = replace(fixture, observed_text="改変")

    with pytest.raises(ValueError):
        build_frozen_window_contexts(tampered, arm="bidirectional-offline")
