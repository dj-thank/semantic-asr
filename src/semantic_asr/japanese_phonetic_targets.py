"""Deterministic Japanese kana-to-mora and kana-to-phone training targets.

This is an orthographic pronunciation baseline, not a learned pronunciation model. Input must be an
explicit kana reading fixed before evaluation. Kanji, Latin text, accent, devoicing, and contextual
allophones are never guessed silently.
"""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

from .contracts import sha256_json
from .japanese import split_mora, to_katakana
from .phonetic_training import PhoneticLabelInventory

_BASIC = {
    "ア": ("a",),
    "イ": ("i",),
    "ウ": ("u",),
    "エ": ("e",),
    "オ": ("o",),
    "カ": ("k", "a"),
    "キ": ("k", "i"),
    "ク": ("k", "u"),
    "ケ": ("k", "e"),
    "コ": ("k", "o"),
    "ガ": ("g", "a"),
    "ギ": ("g", "i"),
    "グ": ("g", "u"),
    "ゲ": ("g", "e"),
    "ゴ": ("g", "o"),
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
    "ナ": ("n", "a"),
    "ニ": ("n", "i"),
    "ヌ": ("n", "u"),
    "ネ": ("n", "e"),
    "ノ": ("n", "o"),
    "ハ": ("h", "a"),
    "ヒ": ("h", "i"),
    "フ": ("f", "u"),
    "ヘ": ("h", "e"),
    "ホ": ("h", "o"),
    "バ": ("b", "a"),
    "ビ": ("b", "i"),
    "ブ": ("b", "u"),
    "ベ": ("b", "e"),
    "ボ": ("b", "o"),
    "パ": ("p", "a"),
    "ピ": ("p", "i"),
    "プ": ("p", "u"),
    "ペ": ("p", "e"),
    "ポ": ("p", "o"),
    "マ": ("m", "a"),
    "ミ": ("m", "i"),
    "ム": ("m", "u"),
    "メ": ("m", "e"),
    "モ": ("m", "o"),
    "ヤ": ("y", "a"),
    "ユ": ("y", "u"),
    "ヨ": ("y", "o"),
    "ラ": ("r", "a"),
    "リ": ("r", "i"),
    "ル": ("r", "u"),
    "レ": ("r", "e"),
    "ロ": ("r", "o"),
    "ワ": ("w", "a"),
    "ヰ": ("w", "i"),
    "ヱ": ("w", "e"),
    "ヲ": ("o",),
    "ン": ("N",),
    "ッ": ("q",),
    "ー": (":",),
    "ヵ": ("k", "a"),
    "ヶ": ("k", "e"),
}

_DIGRAPHS = {
    "キャ": ("ky", "a"),
    "キュ": ("ky", "u"),
    "キョ": ("ky", "o"),
    "ギャ": ("gy", "a"),
    "ギュ": ("gy", "u"),
    "ギョ": ("gy", "o"),
    "シャ": ("sh", "a"),
    "シュ": ("sh", "u"),
    "ショ": ("sh", "o"),
    "ジャ": ("j", "a"),
    "ジュ": ("j", "u"),
    "ジョ": ("j", "o"),
    "チャ": ("ch", "a"),
    "チュ": ("ch", "u"),
    "チョ": ("ch", "o"),
    "ニャ": ("ny", "a"),
    "ニュ": ("ny", "u"),
    "ニョ": ("ny", "o"),
    "ヒャ": ("hy", "a"),
    "ヒュ": ("hy", "u"),
    "ヒョ": ("hy", "o"),
    "ビャ": ("by", "a"),
    "ビュ": ("by", "u"),
    "ビョ": ("by", "o"),
    "ピャ": ("py", "a"),
    "ピュ": ("py", "u"),
    "ピョ": ("py", "o"),
    "ミャ": ("my", "a"),
    "ミュ": ("my", "u"),
    "ミョ": ("my", "o"),
    "リャ": ("ry", "a"),
    "リュ": ("ry", "u"),
    "リョ": ("ry", "o"),
    "イェ": ("y", "e"),
    "ウィ": ("w", "i"),
    "ウェ": ("w", "e"),
    "ウォ": ("w", "o"),
    "クァ": ("kw", "a"),
    "クィ": ("kw", "i"),
    "クェ": ("kw", "e"),
    "クォ": ("kw", "o"),
    "グァ": ("gw", "a"),
    "グィ": ("gw", "i"),
    "グェ": ("gw", "e"),
    "グォ": ("gw", "o"),
    "シェ": ("sh", "e"),
    "ジェ": ("j", "e"),
    "チェ": ("ch", "e"),
    "ティ": ("t", "i"),
    "トゥ": ("t", "u"),
    "ディ": ("d", "i"),
    "ドゥ": ("d", "u"),
    "ツァ": ("ts", "a"),
    "ツィ": ("ts", "i"),
    "ツェ": ("ts", "e"),
    "ツォ": ("ts", "o"),
    "ファ": ("f", "a"),
    "フィ": ("f", "i"),
    "フェ": ("f", "e"),
    "フォ": ("f", "o"),
    "フャ": ("fy", "a"),
    "フュ": ("fy", "u"),
    "フョ": ("fy", "o"),
    "ヴァ": ("v", "a"),
    "ヴィ": ("v", "i"),
    "ヴ": ("v", "u"),
    "ヴェ": ("v", "e"),
    "ヴォ": ("v", "o"),
    "ヴャ": ("vy", "a"),
    "ヴュ": ("vy", "u"),
    "ヴョ": ("vy", "o"),
}

_PHONE_MAP = {**_BASIC, **_DIGRAPHS}
_IGNORABLE = frozenset(" \t\r\n、。・，．,.!?！？「」『』（）()［］[]【】")


@dataclass(frozen=True, slots=True)
class JapanesePronunciationPolicy:
    blank_symbol: str = "<blk>"
    nasal_symbol: str = "N"
    sokuon_symbol: str = "q"
    long_vowel_symbol: str = ":"
    ignore_punctuation: bool = True
    mapping_revision: str = "ja-kana-mora-phone-v1"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.blank_symbol:
            raise ValueError("blank_symbol is required")
        if not self.mapping_revision:
            raise ValueError("mapping_revision is required")
        if self.nasal_symbol != "N" or self.sokuon_symbol != "q":
            raise ValueError("v1 freezes nasal=N and sokuon=q")
        if self.long_vowel_symbol != ":":
            raise ValueError("v1 freezes long_vowel_symbol=':'")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "phoneMap": tuple(sorted(_PHONE_MAP.items())),
                "ignorable": tuple(sorted(_IGNORABLE)),
            }
        )

    def inventories(self) -> tuple[PhoneticLabelInventory, PhoneticLabelInventory]:
        phone_labels = (self.blank_symbol, *sorted({p for row in _PHONE_MAP.values() for p in row}))
        mora_labels = (self.blank_symbol, *sorted(_PHONE_MAP))
        source = self.digest
        return (
            PhoneticLabelInventory(
                kind="phone",
                labels=phone_labels,
                blank_symbol=self.blank_symbol,
                revision=self.mapping_revision,
                source_manifest_sha256=source,
            ),
            PhoneticLabelInventory(
                kind="mora",
                labels=mora_labels,
                blank_symbol=self.blank_symbol,
                revision=self.mapping_revision,
                source_manifest_sha256=source,
            ),
        )


@dataclass(frozen=True, slots=True)
class JapanesePronunciationTarget:
    normalized_reading: str
    mora_symbols: tuple[str, ...]
    phone_symbols: tuple[str, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if not self.normalized_reading or not self.mora_symbols or not self.phone_symbols:
            raise ValueError("pronunciation target must be non-empty")
        if len(self.policy_digest) != 64:
            raise ValueError("policy_digest must be a SHA-256 value")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def target_ids(
        self,
        phone_inventory: PhoneticLabelInventory,
        mora_inventory: PhoneticLabelInventory,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if phone_inventory.source_manifest_sha256 != self.policy_digest:
            raise ValueError("phone inventory is bound to a different pronunciation policy")
        if mora_inventory.source_manifest_sha256 != self.policy_digest:
            raise ValueError("mora inventory is bound to a different pronunciation policy")
        phone_index = {label: index for index, label in enumerate(phone_inventory.labels)}
        mora_index = {label: index for index, label in enumerate(mora_inventory.labels)}
        try:
            return (
                tuple(phone_index[value] for value in self.phone_symbols),
                tuple(mora_index[value] for value in self.mora_symbols),
            )
        except KeyError as exc:
            raise ValueError("pronunciation symbol is absent from the frozen inventory") from exc


def _normalize_reading(reading: str, policy: JapanesePronunciationPolicy) -> str:
    value = unicodedata.normalize("NFKC", reading)
    output = []
    for character in to_katakana(value):
        if character in _IGNORABLE:
            if policy.ignore_punctuation:
                continue
            raise ValueError(f"punctuation is not allowed by this policy: {character!r}")
        output.append(character)
    normalized = "".join(output)
    if not normalized:
        raise ValueError("reading contains no pronounceable kana")
    return normalized


def japanese_pronunciation_target(
    reading: str,
    *,
    policy: JapanesePronunciationPolicy | None = None,
) -> JapanesePronunciationTarget:
    policy = policy or JapanesePronunciationPolicy()
    normalized = _normalize_reading(reading, policy)
    moras = tuple(split_mora(normalized))
    phones = []
    for mora in moras:
        try:
            phones.extend(_PHONE_MAP[mora])
        except KeyError as exc:
            raise ValueError(
                f"unsupported or non-kana pronunciation unit {mora!r}; "
                "provide an explicit mapping revision rather than guessing"
            ) from exc
    return JapanesePronunciationTarget(
        normalized_reading=normalized,
        mora_symbols=moras,
        phone_symbols=tuple(phones),
        policy_digest=policy.digest,
    )
