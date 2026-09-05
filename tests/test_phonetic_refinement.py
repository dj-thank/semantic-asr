import hashlib
from dataclasses import replace

import pytest

from semantic_asr.phonetic_refinement import FrozenPhoneContextPolicy, PhoneContextCandidate


def candidate(identifier, text, phone_score, language_score, phones=("a",)):
    return PhoneContextCandidate(
        identifier,
        text,
        phones,
        phone_score,
        language_score,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        hashlib.sha256(text.encode()).hexdigest(),
    )


def policy(**kwargs):
    return FrozenPhoneContextPolicy("a" * 64, "d" * 64, **kwargs)


def test_ties_and_unresolved_homophones_retain_baseline():
    rows = [candidate("z", "仕様", -1.0, -2.0), candidate("a", "使用", -1.0, -2.0)]
    d = policy(language_weight=1.0).select(rows, baseline_id="z")
    assert d.selected_id == "z" and not d.changed and d.status == "retained"


def test_context_cannot_overrule_phone_regression_guard():
    rows = [
        candidate("a", "昨日学校を行った", -0.1, -10),
        candidate("b", "昨日学校に行った", -0.5, 10),
    ]
    d = policy(language_weight=100).select(rows, baseline_id="a")
    assert d.selected_id == "a"


def test_same_phone_context_selection_is_provisional_not_acoustic_spelling_proof():
    rows = [candidate("a", "公聴会", -0.1, -1), candidate("b", "校長会", -0.1, -0.5)]
    d = policy(language_weight=1, maximum_edit_ratio=1).select(rows, baseline_id="a")
    assert d.selected_id == "b" and d.reason == "context-resolved-homophone"
    assert d.status == "provisional"


def test_audio_only_improvement_can_abstain_on_language_disagreement():
    rows = [candidate("a", "マタ", -1, -1), candidate("b", "マダ", -0.1, -10)]
    d = policy(maximum_edit_ratio=1, require_language_agreement=True).select(rows, baseline_id="a")
    assert not d.changed


def test_evidence_cannot_cross_model_recording_or_window_boundaries():
    a = candidate("a", "はい", -0.1, -1)
    b = candidate("b", "いいえ", -0.01, -0.1)
    for change in (
        {"profile_digest": "e" * 64},
        {"source_audio_sha256": "e" * 64},
        {"posterior_digest": "e" * 64},
    ):
        with pytest.raises(ValueError):
            policy().select([a, replace(b, **change)], baseline_id="a")
    with pytest.raises(ValueError, match="baseline"):
        policy().select([b], baseline_id="a")


def test_text_tampering_and_nan_scores_fail_closed():
    a = candidate("a", "はい", -0.1, -1)
    for change in ({"text": "いいえ"}, {"phone_score": float("nan")}, {"language_score": True}):
        with pytest.raises(ValueError):
            replace(a, **change)


@pytest.mark.parametrize("bad", [True, -1, float("inf"), float("nan")])
def test_policy_rejects_invalid_weights(bad):
    with pytest.raises(ValueError):
        policy(language_weight=bad)
