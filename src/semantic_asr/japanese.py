from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from .contracts import MoraUnit

SMALL_KANA = frozenset("ァィゥェォャュョヮヵヶ")
PUNCTUATION = frozenset(" \t\r\n、。,.!?！？・「」『』（）()［］[]【】…‥:：;；\"'“”‘’")
KATAKANA_RANGE = range(0x30A0, 0x3100)
_MULTI_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([、。！？!?：:；;）\]】」』])")
_SPACE_AFTER_OPEN = re.compile(r"([（\[【「『])\s+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


def to_katakana(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    output: list[str] = []
    for character in normalized:
        code = ord(character)
        output.append(chr(code + 0x60) if 0x3041 <= code <= 0x3096 else character)
    return "".join(output)


def classify_mora(kana: str) -> str:
    if kana == "ン":
        return "moraic-nasal"
    if kana == "ッ":
        return "geminate"
    if kana == "ー":
        return "long-vowel"
    return "regular"


def _is_katakana(character: str) -> bool:
    return len(character) == 1 and ord(character) in KATAKANA_RANGE


def split_mora(value: str, *, include_unknown: bool = False) -> list[MoraUnit]:
    normalized = to_katakana(value)
    units: list[MoraUnit] = []
    for offset, character in enumerate(normalized):
        if character in PUNCTUATION:
            continue
        if character in SMALL_KANA and units and units[-1].kind == "regular":
            previous = units[-1]
            units[-1] = replace(
                previous,
                kana=previous.kana + character,
                surface=(previous.surface or previous.kana) + character,
                char_end=offset + 1,
            )
            continue
        if not _is_katakana(character) and not include_unknown:
            continue
        units.append(
            MoraUnit(
                index=len(units),
                kana=character,
                surface=character,
                kind=classify_mora(character),
                char_start=offset,
                char_end=offset + 1,
            )
        )
    return units


def mora_sequence(value: str) -> list[str]:
    return [unit.kana for unit in split_mora(value)]


def count_mora(value: str) -> int:
    return len(split_mora(value))


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _confidence(value: Any) -> float | None:
    number = _finite_number(value)
    return None if number is None else min(1.0, max(0.0, number))


def merge_character_alignment(rows: Iterable[Mapping[str, Any]]) -> list[MoraUnit]:
    """Merge timed character CTC rows into canonical timed mora units."""

    units: list[MoraUnit] = []
    for row in rows:
        raw = row.get("char", row.get("surface", row.get("text", "")))
        for character in to_katakana(str(raw)):
            if character in PUNCTUATION or not _is_katakana(character):
                continue
            start_ms = _finite_number(row.get("startMs", row.get("start_ms")))
            end_ms = _finite_number(row.get("endMs", row.get("end_ms")))
            confidence = _confidence(row.get("confidence"))
            if character in SMALL_KANA and units and units[-1].kind == "regular":
                previous = units[-1]
                combined_confidence = previous.confidence
                if confidence is not None:
                    combined_confidence = (
                        confidence
                        if combined_confidence is None
                        else min(combined_confidence, confidence)
                    )
                units[-1] = replace(
                    previous,
                    kana=previous.kana + character,
                    surface=(previous.surface or previous.kana) + character,
                    start_ms=previous.start_ms if previous.start_ms is not None else start_ms,
                    end_ms=end_ms if end_ms is not None else previous.end_ms,
                    confidence=combined_confidence,
                )
                continue
            units.append(
                MoraUnit(
                    index=len(units),
                    kana=character,
                    surface=character,
                    kind=classify_mora(character),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=confidence,
                )
            )
    return units


def contains_japanese(text: str) -> bool:
    return bool(_JAPANESE_RE.search(str(text or "")))


def _ascii_boundary(left: str, right: str) -> bool:
    return bool(
        left
        and right
        and left[-1].isascii()
        and right[0].isascii()
        and (left[-1].isalnum() or left[-1] in "_+-/#@.%")
        and (right[0].isalnum() or right[0] in "_+-/#@.%")
    )


def join_japanese_fragments(fragments: Iterable[str], *, max_overlap: int = 160) -> str:
    output = ""
    for raw in fragments:
        fragment = str(raw or "").strip()
        if not fragment:
            continue
        if not output:
            output = fragment
            continue
        overlap = 0
        for length in range(min(max_overlap, len(output), len(fragment)), 1, -1):
            if output[-length:] == fragment[:length]:
                overlap = length
                break
        suffix = fragment[overlap:]
        if not suffix:
            continue
        if output[-1:] in "、。！？!?" and suffix[:1] == output[-1:]:
            suffix = suffix[1:]
        output += (" " if _ascii_boundary(output, suffix) else "") + suffix
    return output


def deterministic_normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _MULTI_SPACE.sub(" ", value)
    value = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", value)
    value = _SPACE_AFTER_OPEN.sub(r"\1", value)
    if contains_japanese(value):
        value = value.replace("!", "！").replace("?", "？")
    value = re.sub(r"([、。！？])\1{2,}", r"\1\1", value)
    value = _MULTI_NEWLINE.sub("\n\n", value)
    return value.strip()


def optional_reading(text: str, *, use_pyopenjtalk: bool = False) -> str | None:
    value = str(text or "").strip()
    if not value:
        return None
    katakana = to_katakana(value)
    if all(character in PUNCTUATION or _is_katakana(character) for character in katakana):
        return katakana
    if not use_pyopenjtalk:
        return None
    try:
        import pyopenjtalk
    except ImportError:
        return None
    try:
        reading = pyopenjtalk.g2p(value, kana=True)
    except Exception:
        return None
    return to_katakana(reading) if reading else None
