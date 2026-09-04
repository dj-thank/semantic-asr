from __future__ import annotations

import pytest

from semantic_asr.multilevel_lattice import (
    BoundedUtility,
    CallableGlobalSequenceScorer,
    DeliberationLattice,
    DeliberationPolicy,
    DeliberationSpan,
    DocumentContext,
    GlobalPathScore,
    LatticeArc,
    UtilityCalibrationProfile,
    decode_global_lattice,
    frozen_profile_digest,
    path_digest,
)
from semantic_asr.score_semantics import EvidenceScore, ScoreKind

PROFILE = "1" * 64
INPUT = "2" * 64
AUDIO = "a" * 64


def utility(channel: str, value: float) -> BoundedUtility:
    return BoundedUtility(
        channel=channel,  # type: ignore[arg-type]
        value=value,
        source=f"test-{channel}",
        profile_digest=PROFILE,
        input_digest=INPUT,
    )


def arc(
    arc_id: str,
    span_id: str,
    text: str,
    audio: float,
    *,
    retained: bool = False,
    origin: str = "first-pass",
    pronunciation_key: str | None = None,
    extra: tuple[BoundedUtility, ...] = (),
    eligible: bool = True,
) -> LatticeArc:
    del retained
    return LatticeArc(
        arc_id=arc_id,
        span_id=span_id,
        text=text,
        origin=origin,  # type: ignore[arg-type]
        utilities=(utility("asr_acoustic", audio), *extra),
        observed_eligible=eligible,
        pronunciation_key=pronunciation_key,
    )


def policy(**overrides: object) -> DeliberationPolicy:
    values: dict[str, object] = {
        "channel_weights": (
            ("asr_acoustic", 1.0),
            ("phone", 0.8),
            ("mora", 0.8),
            ("semantic", 0.1),
            ("transition", 0.2),
        ),
        "global_context_weight": 0.8,
        "retention_bonus": 0.0,
        "maximum_span_audio_regression": 0.2,
        "maximum_mean_audio_regression": 0.2,
        "minimum_final_margin": 0.01,
    }
    values.update(overrides)
    return DeliberationPolicy(**values)  # type: ignore[arg-type]


def two_choice_lattice(retained_audio: float, alternative_audio: float) -> DeliberationLattice:
    prefix = DeliberationSpan(
        span_id="prefix",
        index=0,
        start_ms=0,
        end_ms=100,
        retained_arc_id="prefix-retained",
        arcs=(arc("prefix-retained", "prefix", "レビュー完了までは", 0.9),),
    )
    choice = DeliberationSpan(
        span_id="choice",
        index=1,
        start_ms=100,
        end_ms=200,
        retained_arc_id="mata",
        arcs=(
            arc("mata", "choice", "また", retained_audio),
            arc("mada", "choice", "まだ", alternative_audio),
        ),
    )
    suffix = DeliberationSpan(
        span_id="suffix",
        index=2,
        start_ms=200,
        end_ms=300,
        retained_arc_id="suffix-retained",
        arcs=(arc("suffix-retained", "suffix", "マージしません", 0.9),),
    )
    return DeliberationLattice(
        document_id="doc",
        source_audio_sha256=AUDIO,
        spans=(prefix, choice, suffix),
    )


def context_scorer():
    return CallableGlobalSequenceScorer(
        lambda path, context: 1.0 if "まだ" in "".join(arc.text for arc in path) else -1.0,
        source="full-document-test",
        profile_digest=frozen_profile_digest("full-context", "r1", {"model": "test"}),
    )


def test_global_scorer_can_choose_contextually_coherent_path() -> None:
    decision = decode_global_lattice(
        two_choice_lattice(0.60, 0.55),
        policy=policy(),
        context=DocumentContext(
            left_context="レビューが終わるまで保留です。",
            right_context="承認後に統合します。",
        ),
        sequence_scorer=context_scorer(),
    )

    assert decision.observed_text == "レビュー完了まではまだマージしません"
    assert "global-context-applied" in decision.reasons
    assert decision.status == "accepted"


def test_context_cannot_override_large_audio_regression() -> None:
    decision = decode_global_lattice(
        two_choice_lattice(0.85, 0.10),
        policy=policy(maximum_span_audio_regression=0.2),
        sequence_scorer=context_scorer(),
    )

    assert "また" in decision.observed_text
    assert all(resolution.selected_arc_id != "mada" for resolution in decision.resolutions)


def test_generated_context_only_arc_is_not_observed_eligible() -> None:
    retained = arc("retained", "choice", "また", 0.4)
    context_only = LatticeArc(
        arc_id="context-only",
        span_id="choice",
        text="まだ",
        origin="context-proposal",
        utilities=(utility("semantic", 1.0),),
        observed_eligible=True,
    )
    lattice = DeliberationLattice(
        document_id="doc",
        source_audio_sha256=AUDIO,
        spans=(
            DeliberationSpan(
                span_id="choice",
                index=0,
                start_ms=0,
                end_ms=100,
                arcs=(retained, context_only),
                retained_arc_id="retained",
            ),
        ),
    )
    scorer = CallableGlobalSequenceScorer(
        lambda path, context: 1.0 if path[0].arc_id == "context-only" else -1.0,
        source="context",
        profile_digest=PROFILE,
    )

    decision = decode_global_lattice(lattice, policy=policy(), sequence_scorer=scorer)

    assert decision.selected.arcs[0].arc_id == "retained"
    assert decision.margin == 0.0
    assert "single-surviving-path" in decision.reasons


def test_phone_verified_generated_arc_can_enter_but_remains_provisional() -> None:
    retained = arc("retained", "choice", "また", 0.4)
    generated = LatticeArc(
        arc_id="generated",
        span_id="choice",
        text="まだ",
        origin="phonetic-proposal",
        utilities=(utility("phone", 0.55),),
        observed_eligible=True,
        pronunciation_key="mada",
    )
    lattice = DeliberationLattice(
        document_id="doc",
        source_audio_sha256=AUDIO,
        spans=(
            DeliberationSpan(
                span_id="choice",
                index=0,
                start_ms=0,
                end_ms=100,
                arcs=(retained, generated),
                retained_arc_id="retained",
            ),
        ),
    )
    scorer = CallableGlobalSequenceScorer(
        lambda path, context: 1.0 if path[0].arc_id == "generated" else -1.0,
        source="context",
        profile_digest=PROFILE,
    )

    decision = decode_global_lattice(
        lattice,
        policy=policy(maximum_span_audio_regression=1.0, maximum_mean_audio_regression=1.0),
        sequence_scorer=scorer,
    )

    assert decision.selected.arcs[0].arc_id == "generated"
    assert decision.status == "provisional"
    assert "selected-generated-proposal" in decision.reasons


def test_homophone_selection_is_marked_context_resolved_orthography() -> None:
    pronunciation_key = "shiyou"
    retained = arc(
        "use",
        "word",
        "使用",
        0.5,
        pronunciation_key=pronunciation_key,
    )
    specification = arc(
        "specification",
        "word",
        "仕様",
        0.49,
        pronunciation_key=pronunciation_key,
    )
    lattice = DeliberationLattice(
        document_id="doc",
        source_audio_sha256=AUDIO,
        spans=(
            DeliberationSpan(
                span_id="word",
                index=0,
                start_ms=0,
                end_ms=100,
                arcs=(retained, specification),
                retained_arc_id="use",
            ),
        ),
    )
    scorer = CallableGlobalSequenceScorer(
        lambda path, context: 1.0 if path[0].text == "仕様" else -1.0,
        source="bidirectional-context",
        profile_digest=PROFILE,
    )

    decision = decode_global_lattice(
        lattice,
        policy=policy(),
        context=DocumentContext(right_context="を変更して再ビルドします。"),
        sequence_scorer=scorer,
    )

    assert decision.selected.text == "仕様"
    assert decision.resolutions[0].mode == "context-resolved-orthography"


def test_global_score_must_be_bound_to_exact_path() -> None:
    class BadScorer:
        def score(self, path, *, context):
            return GlobalPathScore(
                value=1.0,
                source="bad",
                profile_digest=PROFILE,
                path_digest="f" * 64,
                context_digest=context.digest,
            )

    with pytest.raises(ValueError, match="different path"):
        decode_global_lattice(
            two_choice_lattice(0.6, 0.55),
            policy=policy(),
            sequence_scorer=BadScorer(),
        )


def test_utility_calibration_preserves_score_semantics() -> None:
    profile = UtilityCalibrationProfile(
        channel="phone",
        score_source="ctc-phone:model@r1:phones-r1",
        score_kind=ScoreKind.LOG_LIKELIHOOD,
        center=-1.0,
        scale=0.5,
        fitted_manifest_sha256="b" * 64,
        revision="cal-r1",
    )
    score = EvidenceScore(
        value=-0.5,
        kind=ScoreKind.LOG_LIKELIHOOD,
        source="ctc-phone:model@r1:phones-r1",
    )

    transformed = profile.transform(score)

    assert transformed.channel == "phone"
    assert 0 < transformed.value < 1

    with pytest.raises(ValueError, match="source"):
        profile.transform(
            EvidenceScore(
                value=-0.5,
                kind=ScoreKind.LOG_LIKELIHOOD,
                source="another-source",
            )
        )


def test_path_digest_changes_when_surface_changes() -> None:
    left = arc("a", "s", "仕様", 0.5)
    right = arc("a", "s", "使用", 0.5)

    assert path_digest((left,)) != path_digest((right,))
