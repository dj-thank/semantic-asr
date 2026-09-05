import pytest

from semantic_asr.mora_phonology import kana_to_phone_moras, phones_to_moras


def test_foreign_yoon_geminate_nasal_and_long_vowels():
    rows = kana_to_phone_moras("きゃ ティ ファール しんぶん がっこう")
    assert [u.kana_options[0] for u in rows] == [
        "キャ",
        "ティ",
        "ファ",
        "ー",
        "ル",
        "シ",
        "ン",
        "ブ",
        "ン",
        "ガ",
        "ッ",
        "コ",
        "ウ",
    ]
    assert rows[0].phones == ("ky", "a")
    assert rows[1].phones == ("t", "i")
    assert rows[3].phones == ("a",)
    assert rows[-3].phones == ("cl",)


def test_audio_phone_projection_retains_voicing_repetitions_and_ambiguity():
    phones = ("sil", "k", "I", "t", "a", "a", "pau", "j", "i", "cl", "N", "q")
    units = phones_to_moras(phones)
    assert tuple(p for unit in units for p in unit.phones) == phones
    assert units[1].devoiced and units[1].kana_options == ("キ",)
    assert units[3].possible_long_vowel
    assert set(units[5].kana_options) == {"ジ", "ヂ"}
    assert units[-1].kind == "unresolved"
    assert sum(u.kind not in {"pause", "unresolved"} for u in units) == 6


def test_orthography_cannot_invent_phonetic_long_vowels():
    assert [u.phones for u in kana_to_phone_moras("えいおう")] == [("e",), ("i",), ("o",), ("u",)]
    assert [u.phones for u in kana_to_phone_moras("えーおー")] == [("e",), ("e",), ("o",), ("o",)]


@pytest.mark.parametrize("text", ["ー", "キャ。ー", "ンー", "キャャ", "カィ", "き、ゃ", "日本"])
def test_ambiguous_or_malformed_kana_requires_an_explicit_reading(text):
    with pytest.raises(ValueError):
        kana_to_phone_moras(text)
