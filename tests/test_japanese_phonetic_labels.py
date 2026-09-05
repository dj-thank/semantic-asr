from __future__ import annotations

import pytest

from semantic_asr.phonetic_runtime.japanese_labels import (
    JapanesePhoneticLabelProfile,
    JapanesePronunciationSource,
    build_japanese_pronunciation_lexicon,
    normalize_japanese_reading,
    segment_japanese_moras,
)


def test_hiragana_normalizes_to_explicit_katakana_labels() -> None:
    profile = JapanesePhoneticLabelProfile()

    labels = profile.label("がっこう")

    assert labels.normalized_reading == "ガッコウ"
    assert labels.moras == ("ガ", "ッ", "コ", "ウ")
    assert labels.phones == ("g", "a", "q", "k", "o", "u")
    assert labels.profile_digest == profile.digest


def test_long_vowels_nasal_and_palatalized_mora_have_distinct_symbols() -> None:
    profile = JapanesePhoneticLabelProfile()

    labels = profile.label("キョーン")

    assert labels.moras == ("キョ", "ー", "ン")
    assert labels.phones == ("ky", "o", ":", "N")


def test_common_foreign_katakana_is_explicitly_supported() -> None:
    profile = JapanesePhoneticLabelProfile()

    labels = profile.label("ファイル・ティー")

    assert labels.moras == ("ファ", "イ", "ル", "ティ", "ー")
    assert labels.phones == ("f", "a", "i", "r", "u", "t", "i", ":")


def test_kanji_and_unsupported_small_kana_fail_closed() -> None:
    with pytest.raises(ValueError, match="Kanji readings must be supplied explicitly"):
        normalize_japanese_reading("学校")

    with pytest.raises(ValueError, match="no valid base mora"):
        segment_japanese_moras("ャ")


def test_invalid_geminate_and_long_mark_fail_closed() -> None:
    profile = JapanesePhoneticLabelProfile()

    with pytest.raises(ValueError, match="long-vowel mark"):
        profile.label("ーア")
    with pytest.raises(ValueError, match="geminate marker"):
        profile.label("アッ")
    with pytest.raises(ValueError, match="consonant-bearing"):
        profile.label("ッア")


def test_frozen_inventories_cover_every_emitted_symbol() -> None:
    profile = JapanesePhoneticLabelProfile()
    phone = profile.phone_inventory()
    mora = profile.mora_inventory()
    labels = profile.label("シンブンヲヨム")

    assert phone.decode(phone.encode(labels.phones)) == labels.phones
    assert mora.decode(mora.encode(labels.moras)) == labels.moras
    assert phone.blank_id == mora.blank_id == 0


def test_pronunciation_lexicon_requires_explicit_readings_and_is_profile_bound() -> None:
    profile = JapanesePhoneticLabelProfile()
    lexicon = build_japanese_pronunciation_lexicon(
        (
            JapanesePronunciationSource(
                entry_id="specification",
                text="仕様",
                reading="しよう",
                tags=("technical",),
            ),
            JapanesePronunciationSource(
                entry_id="use",
                text="使用",
                reading="しよう",
                tags=("general",),
            ),
        ),
        name="homophones",
        revision="r1",
        profile=profile,
    )

    assert len(lexicon.entries) == 2
    assert lexicon.entries[0].phone_symbols == lexicon.entries[1].phone_symbols
    assert profile.digest[:16] in lexicon.revision
