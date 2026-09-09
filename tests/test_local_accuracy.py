import pytest

from semantic_asr.local_accuracy import measure, normalized_trial, summarize, surface_review


def test_strict_counts_punctuation_and_number_representation():
    assert measure("一人。", "1人")["strict_errors"] == 2
    assert not measure("一人。", "1人")["raw_exact"]
    assert measure("あ 。", "あ。")["strict_exact"]
    assert not measure("あ 。", "あ。")["raw_exact"]


def test_provisional_errors_remain_in_denominator():
    rows = [
        {**measure("あ。", "い。"), "status": "provisional"},
        {**measure("う。", "う。"), "status": "accepted"},
    ]
    score = summarize(rows)
    assert score["strict_cer"] == 0.25
    assert score["raw_exact_rate"] == 0.5
    assert score["accepted_strict_cer"] == 0
    assert score["provisional_count"] == 1


def test_cached_unknown_status_is_not_accepted():
    result = summarize([{**measure("あ", "あ"), "status": "unknown"}])
    assert result["accepted_strict_cer"] is None
    assert result["accepted_count"] == 0


def test_normalized_trial_preserves_input_and_does_not_double_stop():
    observed = "二〇二四万円です"
    assert normalized_trial(observed, terminal_stop=False) == "2024万円です"
    assert normalized_trial(observed, terminal_stop=True) == "2024万円です。"
    assert observed == "二〇二四万円です"
    assert normalized_trial("行きません！", terminal_stop=True) == "行きません！"
    assert normalized_trial("", terminal_stop=True) == ""


def test_missing_reference_is_an_error():
    with pytest.raises(ValueError):
        measure(" ", "あ")


def test_surface_variants_are_not_semantic_errors():
    assert surface_review("Six Days。", "SIX DAYS").startswith("表記差のみ")
    assert surface_review("ほんじつ", "ホンジツ").startswith("表記差のみ")
    assert surface_review("行きます", "行きません").startswith("要聴取")
    assert surface_review("Six Days in Fallujah", "デースインフルジュ").startswith("要聴取")
