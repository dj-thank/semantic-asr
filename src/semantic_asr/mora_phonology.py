"""Japanese mora/phone mapping without claiming that spelling proves pronunciation.

The inventory uses Open JTalk phone symbols. It is a pronunciation *proposal* for
kana, not an acoustic model. Audio-derived phones keep devoicing and ambiguity;
/ji/ cannot prove ジ vs ヂ, and /o o/ cannot prove a long vowel vs a hiatus.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

# Foreign combinations are deliberately explicit: a small kana does not attach to
# every preceding character, nor across punctuation, unknown text, or a pause.
_BASE_ROWS = (
    ("アイウエオ", ("a", "i", "u", "e", "o")),
    ("カキクケコ", ("k a", "k i", "k u", "k e", "k o")),
    ("ガギグゲゴ", ("g a", "g i", "g u", "g e", "g o")),
    ("サシスセソ", ("s a", "sh i", "s u", "s e", "s o")),
    ("ザジズゼゾ", ("z a", "j i", "z u", "z e", "z o")),
    ("タチツテト", ("t a", "ch i", "ts u", "t e", "t o")),
    ("ダヂヅデド", ("d a", "j i", "z u", "d e", "d o")),
    ("ナニヌネノ", ("n a", "n i", "n u", "n e", "n o")),
    ("ハヒフヘホ", ("h a", "h i", "f u", "h e", "h o")),
    ("バビブベボ", ("b a", "b i", "b u", "b e", "b o")),
    ("パピプペポ", ("p a", "p i", "p u", "p e", "p o")),
    ("マミムメモ", ("m a", "m i", "m u", "m e", "m o")),
    ("ヤユヨ", ("y a", "y u", "y o")),
    ("ラリルレロ", ("r a", "r i", "r u", "r e", "r o")),
    ("ワヰヱヲヴ", ("w a", "i", "e", "o", "v u")),
)
_inventory = {
    kana: tuple(phones.split())
    for row, values in _BASE_ROWS
    for kana, phones in zip(row, values, strict=True)
}
for base, onset in (
    ("キ", "ky"),
    ("ギ", "gy"),
    ("シ", "sh"),
    ("ジ", "j"),
    ("チ", "ch"),
    ("ヂ", "j"),
    ("ニ", "ny"),
    ("ヒ", "hy"),
    ("ビ", "by"),
    ("ピ", "py"),
    ("ミ", "my"),
    ("リ", "ry"),
):
    for small, vowel in (("ャ", "a"), ("ュ", "u"), ("ョ", "o"), ("ェ", "e")):
        _inventory[base + small] = (onset, vowel)
for onset, prefix, endings in (
    ("f", "フ", "ァィェォ"),
    ("v", "ヴ", "ァィェォ"),
    ("ts", "ツ", "ァィェォ"),
    ("t", "テ", "ィ"),
    ("d", "デ", "ィ"),
    ("t", "ト", "ゥ"),
    ("d", "ド", "ゥ"),
    ("s", "ス", "ィ"),
    ("z", "ズ", "ィ"),
    ("w", "ウ", "ィェォ"),
    ("y", "イ", "ェ"),
    ("kw", "ク", "ァィェォヮ"),
    ("gw", "グ", "ァィェォヮ"),
    ("fy", "フ", "ャュョ"),
    ("ty", "テ", "ュ"),
    ("dy", "デ", "ュ"),
):
    for small in endings:
        _inventory[prefix + small] = (
            onset,
            {
                "ァ": "a",
                "ィ": "i",
                "ゥ": "u",
                "ェ": "e",
                "ォ": "o",
                "ヮ": "a",
                "ャ": "a",
                "ュ": "u",
                "ョ": "o",
            }[small],
        )
_inventory.update({"ン": ("N",), "ッ": ("cl",)})
MORA_PHONES = MappingProxyType(_inventory)
MORA_COMBINATIONS = frozenset(k for k in MORA_PHONES if len(k) == 2)
VOWELS = frozenset(("a", "i", "u", "e", "o", "I", "U"))
PAUSE_PHONES = frozenset(("sil", "pau"))
_reverse: dict[tuple[str, ...], tuple[str, ...]] = {}
for kana, phones in MORA_PHONES.items():
    _reverse[phones] = tuple(sorted((*_reverse.get(phones, ()), kana)))
PHONE_KANA = MappingProxyType(_reverse)


@dataclass(frozen=True, slots=True)
class MoraPhoneUnit:
    index: int
    kana_options: tuple[str, ...]
    phones: tuple[str, ...]
    phone_start: int
    phone_end: int
    kind: str
    devoiced: bool = False
    possible_long_vowel: bool = False


def phones_to_moras(phones: Sequence[str]) -> tuple[MoraPhoneUnit, ...]:
    """Losslessly group decoded phones; unresolved phones remain explicit units.

    Pauses are returned as ``kind='pause'`` and are not Japanese morae. A vowel
    following the same vowel may be lengthening OR a morpheme boundary. No kanji
    or definite long-vowel spelling is inferred from that fact.
    """
    values = tuple(phones)
    if any(not isinstance(x, str) or not x for x in values):
        raise ValueError("phones must be non-empty strings")
    units: list[MoraPhoneUnit] = []
    offset = 0
    previous_vowel = None
    while offset < len(values):
        phone = values[offset]
        end = offset + 1
        kind = "regular"
        if phone in PAUSE_PHONES:
            kind = "pause"
        elif phone == "N":
            kind = "moraic-nasal"
        elif phone == "cl":
            kind = "geminate"
        elif phone not in VOWELS:
            if end < len(values) and values[end] in VOWELS:
                end += 1
            else:
                kind = "unresolved"
        raw = values[offset:end]
        canonical = tuple(p.lower() if p in ("I", "U") else p for p in raw)
        kana = PHONE_KANA.get(canonical, ())
        if not kana and kind == "regular":
            kind = "unresolved"
        vowel = canonical[-1] if canonical[-1] in ("a", "i", "u", "e", "o") else None
        possible_long = len(raw) == 1 and vowel is not None and vowel == previous_vowel
        units.append(
            MoraPhoneUnit(
                len(units),
                kana,
                raw,
                offset,
                end,
                kind,
                any(p in ("I", "U") for p in raw),
                possible_long,
            )
        )
        previous_vowel = vowel
        offset = end
    assert tuple(p for unit in units for p in unit.phones) == values
    return tuple(units)


def kana_to_phone_moras(kana: str) -> tuple[MoraPhoneUnit, ...]:
    """Strict kana pronunciation proposal; kanji needs an explicit G2P/lexicon.

    Orthographic エイ/オウ are NOT universally rewritten to /ee/ or /oo/.
    Only an explicit ー inherits its preceding vowel. Devoicing is not invented.
    """
    text = unicodedata.normalize("NFKC", kana)
    text = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)
    result = []
    index, phone_index = 0, 0
    previous_vowel = None
    while index < len(text):
        char = text[index]
        if char.isspace() or unicodedata.category(char).startswith("P"):
            previous_vowel = None
            index += 1
            continue
        piece = text[index : index + 2] if text[index : index + 2] in MORA_COMBINATIONS else char
        if piece == "ー":
            if previous_vowel is None:
                raise ValueError("long-vowel mark has no preceding vowel")
            phones = (previous_vowel,)
            kind = "long-vowel"
        else:
            if piece not in MORA_PHONES:
                raise ValueError(f"unsupported kana mora: {piece!r}; provide an explicit reading")
            phones = MORA_PHONES[piece]
            kind = "moraic-nasal" if piece == "ン" else "geminate" if piece == "ッ" else "regular"
        result.append(
            MoraPhoneUnit(
                len(result), (piece,), phones, phone_index, phone_index + len(phones), kind
            )
        )
        previous_vowel = phones[-1] if phones[-1] in VOWELS else None
        phone_index += len(phones)
        index += len(piece)
    return tuple(result)
