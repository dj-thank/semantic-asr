from __future__ import annotations

import pytest

from semantic_asr.japanese_phonetic_targets import (
    JapanesePronunciationPolicy,
    japanese_pronunciation_target,
)


def test_gakkou_preserves_sokuon_and_long_vowel() -> None:
    policy = JapanesePronunciationPolicy()

    target = japanese_pronunciation_target("がっこう。", policy=policy)

    assert target.normalized_reading == "ガッコウ"
    assert target.mora_symbols == ("ガ", "ッ", "コ", "ウ")
    assert target.phone_symbols == ("g", "a", "q", "k", "o", "u")
    assert target.policy_digest == policy.digest


def test_yoon_and_explicit_long_mark_are_distinct_moras() -> None:
    target = japanese_pronunciation_target("キョー")

    assert target.mora_symbols == ("キョ", "ー")
    assert target.phone_symbols == ("ky", "o", ":")


def test_nasal_and_foreign_sound_mapping_are_explicit() -> None:
    target = japanese_pronunciation_target("コンピュータ")

    assert "ン" in target.mora_symbols
    assert "N" in target.phone_symbols
    assert "ピュ" in target.mora_symbols
    assert "py" in target.phone_symbols


def test_inventory_and_target_ids_share_one_policy_digest() -> None:
    policy = JapanesePronunciationPolicy()
    phone, mora = policy.inventories()
    target = japanese_pronunciation_target("まだ", policy=policy)

    phone_ids, mora_ids = target.target_ids(phone, mora)

    assert phone.source_manifest_sha256 == policy.digest
    assert mora.source_manifest_sha256 == policy.digest
    assert tuple(phone.labels[index] for index in phone_ids) == target.phone_symbols
    assert tuple(mora.labels[index] for index in mora_ids) == target.mora_symbols
    assert phone.blank_index not in phone_ids
    assert mora.blank_index not in mora_ids


def test_kanji_and_latin_text_are_not_silently_pronounced() -> None:
    with pytest.raises(ValueError, match="unsupported or non-kana"):
        japanese_pronunciation_target("学校")

    with pytest.raises(ValueError, match="unsupported or non-kana"):
        japanese_pronunciation_target("ASR")


def test_unknown_kana_mapping_fails_instead_of_guessing() -> None:
    with pytest.raises(ValueError, match="explicit mapping revision"):
        japanese_pronunciation_target("ヷ")


def test_punctuation_can_be_made_strict() -> None:
    policy = JapanesePronunciationPolicy(ignore_punctuation=False)

    with pytest.raises(ValueError, match="punctuation"):
        japanese_pronunciation_target("まだ。", policy=policy)
