"""Deterministic Japanese reading-to-mora/phone labeling.

The labeler accepts explicit hiragana or katakana readings. It does not guess readings for Kanji.
Its closed mapping is embedded in the profile digest so training, calibration, inference lexicons,
and evaluation cannot silently use different phonetic conventions.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..phonetic_bridge import FrozenPronunciationLexicon, PronunciationLexiconEntry
from .contracts import PhoneticInventory

_PUNCTUATION = frozenset(
    " \t\r\n、。！？!?・，,．.：:；;（）()［］[]｛｝{}「」『』【】〈〉《》…‥〜～―—-"
)
_SMALL_FOLLOWERS = frozenset("ャュョァィゥェォヮ")
_SMALL_STANDALONE = {"ヵ": "カ", "ヶ": "ケ"}
_SPECIAL_MORAS = frozenset({"ン", "ッ", "ー"})


def _base_row(onset: str, kana: str, vowels: str = "aiueo") -> dict[str, tuple[str, ...]]:
    if len(kana) != len(vowels):
        raise ValueError("kana and vowel rows must have equal length")
    return {
        symbol: ((onset, vowel) if onset else (vowel,))
        for symbol, vowel in zip(kana, vowels, strict=True)
    }


def _mapping() -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    for onset, kana in (
        ("", "アイウエオ"),
        ("k", "カキクケコ"),
        ("g", "ガギグゲゴ"),
        ("n", "ナニヌネノ"),
        ("b", "バビブベボ"),
        ("p", "パピプペポ"),
        ("m", "マミムメモ"),
        ("r", "ラリルレロ"),
    ):
        rows.update(_base_row(onset, kana))
    rows.update(
        {
            "サ": ("s", "a"),
            "シ": ("sh", "i"),
            "ス": ("s", "u"),
            "セ": ("s", "e"),
            "ソ": ("s", "o"),
            "ザ": ("z", "a"),
            "ジ": ("j", "i"),
            "ズ": ("z", "u"),
            "ゼ": ("z", "e"),
            "ゾ": ("z", "o"),
            "タ": ("t", "a"),
            "チ": ("ch", "i"),
            "ツ": ("ts", "u"),
            "テ": ("t", "e"),
            "ト": ("t", "o"),
            "ダ": ("d", "a"),
            "ヂ": ("j", "i"),
            "ヅ": ("z", "u"),
            "デ": ("d", "e"),
            "ド": ("d", "o"),
            "ハ": ("h", "a"),
            "ヒ": ("h", "i"),
            "フ": ("f", "u"),
            "ヘ": ("h", "e"),
            "ホ": ("h", "o"),
            "ヤ": ("y", "a"),
            "ユ": ("y", "u"),
            "ヨ": ("y", "o"),
            "ワ": ("w", "a"),
            "ヰ": ("w", "i"),
            "ヱ": ("w", "e"),
            "ヲ": ("o",),
            "ヴ": ("v", "u"),
            "ン": ("N",),
            "ッ": ("q",),
            "ー": (":",),
        }
    )
    rows.update(
        {
            "キャ": ("ky", "a"),
            "キュ": ("ky", "u"),
            "キョ": ("ky", "o"),
            "キェ": ("ky", "e"),
            "ギャ": ("gy", "a"),
            "ギュ": ("gy", "u"),
            "ギョ": ("gy", "o"),
            "ギェ": ("gy", "e"),
            "シャ": ("sh", "a"),
            "シュ": ("sh", "u"),
            "ショ": ("sh", "o"),
            "シェ": ("sh", "e"),
            "ジャ": ("j", "a"),
            "ジュ": ("j", "u"),
            "ジョ": ("j", "o"),
            "ジェ": ("j", "e"),
            "チャ": ("ch", "a"),
            "チュ": ("ch", "u"),
            "チョ": ("ch", "o"),
            "チェ": ("ch", "e"),
            "ニャ": ("ny", "a"),
            "ニュ": ("ny", "u"),
            "ニョ": ("ny", "o"),
            "ニェ": ("ny", "e"),
            "ヒャ": ("hy", "a"),
            "ヒュ": ("hy", "u"),
            "ヒョ": ("hy", "o"),
            "ヒェ": ("hy", "e"),
            "ビャ": ("by", "a"),
            "ビュ": ("by", "u"),
            "ビョ": ("by", "o"),
            "ビェ": ("by", "e"),
            "ピャ": ("py", "a"),
            "ピュ": ("py", "u"),
            "ピョ": ("py", "o"),
            "ピェ": ("py", "e"),
            "ミャ": ("my", "a"),
            "ミュ": ("my", "u"),
            "ミョ": ("my", "o"),
            "ミェ": ("my", "e"),
            "リャ": ("ry", "a"),
            "リュ": ("ry", "u"),
            "リョ": ("ry", "o"),
            "リェ": ("ry", "e"),
            "イェ": ("y", "e"),
            "ウァ": ("w", "a"),
            "ウィ": ("w", "i"),
            "ウェ": ("w", "e"),
            "ウォ": ("w", "o"),
            "クァ": ("kw", "a"),
            "クィ": ("kw", "i"),
            "クェ": ("kw", "e"),
            "クォ": ("kw", "o"),
            "クヮ": ("kw", "a"),
            "グァ": ("gw", "a"),
            "グィ": ("gw", "i"),
            "グェ": ("gw", "e"),
            "グォ": ("gw", "o"),
            "グヮ": ("gw", "a"),
            "ツァ": ("ts", "a"),
            "ツィ": ("ts", "i"),
            "ツェ": ("ts", "e"),
            "ツォ": ("ts", "o"),
            "スィ": ("s", "i"),
            "ズィ": ("z", "i"),
            "ティ": ("t", "i"),
            "トゥ": ("t", "u"),
            "テュ": ("ty", "u"),
            "ディ": ("d", "i"),
            "ドゥ": ("d", "u"),
            "デュ": ("dy", "u"),
            "ファ": ("f", "a"),
            "フィ": ("f", "i"),
            "フェ": ("f", "e"),
            "フォ": ("f", "o"),
            "フャ": ("fy", "a"),
            "フュ": ("fy", "u"),
            "フョ": ("fy", "o"),
            "ヴァ": ("v", "a"),
            "ヴィ": ("v", "i"),
            "ヴェ": ("v", "e"),
            "ヴォ": ("v", "o"),
            "ヴャ": ("vy", "a"),
            "ヴュ": ("vy", "u"),
            "ヴョ": ("vy", "o"),
            "トァ": ("t", "a"),
            "トィ": ("t", "i"),
            "トェ": ("t", "e"),
            "トォ": ("t", "o"),
            "ドァ": ("d", "a"),
            "ドィ": ("d", "i"),
            "ドェ": ("d", "e"),
            "ドォ": ("d", "o"),
        }
    )
    rows["ヵ"] = rows["カ"]
    rows["ヶ"] = rows["ケ"]
    return rows


_MORA_TO_PHONES = _mapping()


def _to_katakana(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    output: list[str] = []
    for character in normalized:
        codepoint = ord(character)
        if 0x3041 <= codepoint <= 0x3096:
            output.append(chr(codepoint + 0x60))
        elif character in {"ゝ", "ゞ"}:
            raise ValueError("iteration marks must be expanded before phonetic labeling")
        else:
            output.append(character)
    return "".join(output)


def normalize_japanese_reading(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Japanese reading must be a non-empty string")
    katakana = _to_katakana(value)
    output: list[str] = []
    for character in katakana:
        if character in _PUNCTUATION:
            continue
        if character in _SMALL_STANDALONE:
            output.append(_SMALL_STANDALONE[character])
            continue
        if not (0x30A1 <= ord(character) <= 0x30FA or character in {"ー", "ヵ", "ヶ"}):
            raise ValueError(
                f"unsupported character in explicit Japanese reading: {character!r}; "
                "Kanji readings must be supplied explicitly"
            )
        output.append(character)
    if not output:
        raise ValueError("Japanese reading contains no phonetic symbols")
    return "".join(output)


def segment_japanese_moras(normalized_reading: str) -> tuple[str, ...]:
    if not normalized_reading:
        raise ValueError("normalized_reading must not be empty")
    moras: list[str] = []
    for character in normalized_reading:
        if character in _SMALL_FOLLOWERS:
            if not moras or moras[-1] in _SPECIAL_MORAS:
                raise ValueError(f"small kana {character!r} has no valid base mora")
            combined = moras[-1] + character
            if combined not in _MORA_TO_PHONES:
                raise ValueError(f"unsupported Japanese mora combination: {combined!r}")
            moras[-1] = combined
        else:
            if character not in _MORA_TO_PHONES:
                raise ValueError(f"unsupported Japanese mora: {character!r}")
            moras.append(character)
    for index, mora in enumerate(moras):
        if mora == "ー":
            if index == 0 or moras[index - 1] in _SPECIAL_MORAS:
                raise ValueError("long-vowel mark requires a preceding vowel-bearing mora")
        if mora == "ッ":
            if index + 1 >= len(moras) or moras[index + 1] in _SPECIAL_MORAS:
                raise ValueError("geminate marker requires a following consonant-bearing mora")
            following = _MORA_TO_PHONES[moras[index + 1]][0]
            if following in {"a", "i", "u", "e", "o", "N", ":"}:
                raise ValueError("geminate marker must precede a consonant-bearing mora")
    return tuple(moras)


@dataclass(frozen=True, slots=True)
class JapanesePhoneticLabels:
    source_reading_sha256: str
    normalized_reading: str
    moras: tuple[str, ...]
    phones: tuple[str, ...]
    profile_digest: str

    def __post_init__(self) -> None:
        for digest in (self.source_reading_sha256, self.profile_digest):
            if len(digest) != 64:
                raise ValueError("Japanese phonetic label digests must be SHA-256 values")
        if not self.normalized_reading or not self.moras or not self.phones:
            raise ValueError("Japanese phonetic labels must not be empty")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class JapanesePhoneticLabelProfile:
    name: str = "semantic-asr-ja-phones"
    revision: str = "ja-phonetic-labels-v1"
    long_vowel_phone: str = ":"
    moraic_nasal_phone: str = "N"
    geminate_phone: str = "q"
    punctuation_policy: str = "drop-declared-punctuation-v1"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("Japanese phonetic label profile requires name and revision")
        if self.long_vowel_phone != ":":
            raise ValueError("v1 long-vowel phone is fixed to ':'")
        if self.moraic_nasal_phone != "N" or self.geminate_phone != "q":
            raise ValueError("v1 special phone symbols are fixed to N and q")

    @property
    def mapping_digest(self) -> str:
        return sha256_json(tuple(sorted(_MORA_TO_PHONES.items())))

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "mappingDigest": self.mapping_digest,
                "punctuation": tuple(sorted(_PUNCTUATION)),
            }
        )

    def label(self, reading: str) -> JapanesePhoneticLabels:
        normalized = normalize_japanese_reading(reading)
        moras = segment_japanese_moras(normalized)
        phones = tuple(phone for mora in moras for phone in _MORA_TO_PHONES[mora])
        return JapanesePhoneticLabels(
            source_reading_sha256=hashlib.sha256(reading.encode("utf-8")).hexdigest(),
            normalized_reading=normalized,
            moras=moras,
            phones=phones,
            profile_digest=self.digest,
        )

    def phone_inventory(self) -> PhoneticInventory:
        phones = tuple(sorted({phone for values in _MORA_TO_PHONES.values() for phone in values}))
        return PhoneticInventory(
            kind="phone",
            symbols=("<blk>", *phones),
            blank_symbol="<blk>",
            language="ja",
            revision=f"{self.revision}:phones:{self.mapping_digest[:16]}",
        )

    def mora_inventory(self) -> PhoneticInventory:
        return PhoneticInventory(
            kind="mora",
            symbols=("<blk>", *tuple(sorted(_MORA_TO_PHONES))),
            blank_symbol="<blk>",
            language="ja",
            revision=f"{self.revision}:moras:{self.mapping_digest[:16]}",
        )


@dataclass(frozen=True, slots=True)
class JapanesePronunciationSource:
    entry_id: str
    text: str
    reading: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entry_id or not self.text or not self.reading:
            raise ValueError("pronunciation source requires entry_id, text, and reading")
        if len(self.tags) != len(set(self.tags)) or any(not tag for tag in self.tags):
            raise ValueError("pronunciation source tags must be unique and non-empty")


def build_japanese_pronunciation_lexicon(
    entries: tuple[JapanesePronunciationSource, ...],
    *,
    name: str,
    revision: str,
    profile: JapanesePhoneticLabelProfile | None = None,
) -> FrozenPronunciationLexicon:
    if not entries:
        raise ValueError("pronunciation lexicon requires entries")
    if len({entry.entry_id for entry in entries}) != len(entries):
        raise ValueError("pronunciation source entry IDs must be unique")
    if len({entry.text for entry in entries}) != len(entries):
        raise ValueError("pronunciation source surface texts must be unique")
    profile = profile or JapanesePhoneticLabelProfile()
    output: list[PronunciationLexiconEntry] = []
    for entry in entries:
        labels = profile.label(entry.reading)
        output.append(
            PronunciationLexiconEntry(
                entry_id=entry.entry_id,
                text=entry.text,
                phone_symbols=labels.phones,
                mora_symbols=labels.moras,
                reading=labels.normalized_reading,
                tags=entry.tags,
            )
        )
    return FrozenPronunciationLexicon(
        name=name,
        revision=f"{revision}:{profile.digest[:16]}",
        entries=tuple(output),
    )
