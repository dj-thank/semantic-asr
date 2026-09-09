"""Regression for an experimental decoder bypassing acoustic retention."""

import hashlib
from dataclasses import replace

import pytest

from semantic_asr.phonetic_refinement import (
    PhoneContextCandidate,
    guard_decoder_fallback,
    project_decoder_display_edits,
)


def candidate(identifier, text, score, phones=("a",)):
    return PhoneContextCandidate(
        identifier,
        text,
        phones,
        score,
        0.0,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        hashlib.sha256(text.encode()).hexdigest(),
    )


def test_unscored_fallback_cannot_replace_density_with_waterworks():
    base = candidate("base", "2.密度、3.速度", -0.05)
    proposal = candidate("decoder", "2. 水道、3. 速度", -0.10)
    # The previous hybrid's unconditional fallback returned proposal.text.
    decision = guard_decoder_fallback(base, proposal)
    assert decision.text == base.text
    assert not decision.changed
    assert decision.reason == "decoder-phone-regression"


def test_supported_decoder_correction_is_not_discarded():
    base = candidate("base", "記述を示します", -0.2)
    proposal = candidate("decoder", "期日を示します", -0.1)
    decision = guard_decoder_fallback(base, proposal)
    assert decision.text == proposal.text
    assert decision.status == "provisional"
    assert decision.reason == "decoder-not-vetoed"


def test_homophone_tie_is_only_a_veto_result_not_acoustic_spelling_proof():
    base = candidate("base", "密度", -0.1)
    proposal = candidate("decoder", "蜜度", -0.1)
    decision = guard_decoder_fallback(base, proposal)
    assert decision.reason == "decoder-not-vetoed"
    assert decision.status == "provisional"
    assert decision.phone_delta == 0


def test_missing_proposal_score_retains_baseline():
    base = candidate("base", "行かない、行かない", -0.1)
    decision = guard_decoder_fallback(base, None)
    assert decision.text == base.text
    assert decision.reason == "decoder-score-unavailable"


@pytest.mark.parametrize("field", ["profile_digest", "source_audio_sha256", "posterior_digest"])
def test_cross_recording_window_and_profile_are_rejected(field):
    base = candidate("base", "十二", -0.2)
    proposal = candidate("decoder", "二十", -0.1)
    with pytest.raises(ValueError):
        guard_decoder_fallback(base, replace(proposal, **{field: "d" * 64}))


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        guard_decoder_fallback(candidate("same", "はい", -0.2), candidate("same", "いいえ", -0.1))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_score_cannot_enter_guard(value):
    with pytest.raises(ValueError):
        candidate("decoder", "十二", value)


def test_decision_digest_binds_both_proposal_and_baseline():
    base = candidate("base", "密度", -0.05)
    a = guard_decoder_fallback(base, candidate("decoder", "水道", -0.1))
    b = guard_decoder_fallback(base, candidate("decoder", "水道", -0.2))
    assert a.text == b.text == base.text
    assert a.candidate_evidence_digest != b.candidate_evidence_digest


def test_development_tolerance_retains_small_uncertain_difference():
    base = candidate("base", "最小", -0.01)
    proposal = candidate("decoder", "最初", -0.013)
    decision = guard_decoder_fallback(base, proposal, maximum_phone_regression=0.005)
    assert decision.text == proposal.text
    assert decision.status == "provisional"


def test_poor_baseline_fit_is_unresolved_instead_of_trusted_reversion():
    base = candidate("base", "地名の候補A", -0.10)
    proposal = candidate("decoder", "地名の候補B", -0.20)
    decision = guard_decoder_fallback(base, proposal, minimum_baseline_phone_score=-0.08)
    assert decision.text == proposal.text
    assert decision.reason == "decoder-unresolved-low-baseline-fit"
    assert decision.status == "provisional"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["maximum_phone_regression", "minimum_baseline_phone_score"])
def test_invalid_guard_configuration_fails_closed(field, value):
    with pytest.raises(ValueError, match="finite"):
        guard_decoder_fallback(candidate("base", "はい", -0.1), None, **{field: value})


def test_negative_regression_tolerance_fails_closed():
    with pytest.raises(ValueError, match="non-negative"):
        guard_decoder_fallback(candidate("base", "はい", -0.1), None, maximum_phone_regression=-1)


def test_policy_digest_binds_the_development_thresholds():
    base = candidate("base", "はい", -0.1)
    a = guard_decoder_fallback(base, None)
    b = guard_decoder_fallback(base, None, maximum_phone_regression=0.005)
    c = guard_decoder_fallback(base, None, minimum_baseline_phone_score=-0.08)
    assert len({a.policy_digest, b.policy_digest, c.policy_digest}) == 3


def test_same_phone_malformed_spelling_can_be_vetoed_by_context():
    base = replace(candidate("base", "打ち伸ばす", -0.1), language_score=-1.0)
    proposal = replace(candidate("decoder", "内伸す", -0.1), language_score=-2.0)
    decision = guard_decoder_fallback(base, proposal, require_same_phone_language_agreement=True)
    assert decision.text == base.text
    assert decision.reason == "decoder-same-phone-language-regression"


def test_language_veto_does_not_revert_quote_only_edits():
    base = replace(candidate("base", "感謝しますと述べた", -0.1), language_score=-1.0)
    proposal = replace(candidate("decoder", "感謝します」と述べた", -0.1), language_score=-2.0)
    decision = guard_decoder_fallback(base, proposal, require_same_phone_language_agreement=True)
    assert decision.text == proposal.text


def test_same_phone_language_veto_does_not_claim_to_resolve_different_readings():
    base = replace(candidate("base", "最小", -0.1, ("sh", "o", "o")), language_score=-1.0)
    proposal = replace(candidate("decoder", "最初", -0.1, ("sh", "o")), language_score=-2.0)
    decision = guard_decoder_fallback(base, proposal, require_same_phone_language_agreement=True)
    assert decision.text == proposal.text


def test_language_guard_is_opt_in_and_strictly_boolean():
    base = candidate("base", "打ち伸ばす", -0.1)
    proposal = replace(candidate("decoder", "内伸す", -0.1), language_score=-2.0)
    assert guard_decoder_fallback(base, proposal).text == proposal.text
    with pytest.raises(TypeError):
        guard_decoder_fallback(base, proposal, require_same_phone_language_agreement=1)


def test_foreign_name_g2p_conflict_preserves_decoder_proposal_as_unresolved():
    base = replace(candidate("base", "セジュン王", -0.047), language_score=-1.0)
    proposal = replace(candidate("decoder", "世宗王", -0.057), language_score=-0.72)
    decision = guard_decoder_fallback(base, proposal, language_weight=0.2)
    assert decision.text == proposal.text
    assert decision.reason == "decoder-unresolved-phone-language-conflict"
    assert decision.status == "provisional"


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_invalid_joint_preference_weight_fails_closed(value):
    with pytest.raises(ValueError):
        guard_decoder_fallback(candidate("base", "はい", -0.1), None, language_weight=value)


@pytest.mark.parametrize(
    "baseline,proposal",
    [
        ("1 2", "12"),
        ("1、2", "12"),
        ("1。5", "15"),
        ("12", "1 2"),
        ("１２", "１、２"),
        ("一、二", "一二"),
        ("a b", "ab"),
        ("1.5", "15"),
        ("-1.5", "1.5"),
        ("「はい」", "はい"),
        ("行かない", "行く"),
        ("はい、はい", "はい"),
    ],
)
def test_display_projection_preserves_quantity_tokens_quotes_and_speech(baseline, proposal):
    assert project_decoder_display_edits(baseline, proposal) == baseline


def test_display_projection_keeps_equivalent_numbers_and_vetoed_words():
    assert project_decoder_display_edits("二百二十キロの道", "220キロの水道") == "220キロの道"
    assert project_decoder_display_edits("密度、速度", "水道、 速度") == "密度、 速度"


def test_display_projection_rejects_changed_numeric_value():
    assert project_decoder_display_edits("二百二十キロ", "240キロ") == "二百二十キロ"
