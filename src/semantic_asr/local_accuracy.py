"""Reference-aware measurement helpers; never used by runtime candidate selection."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

from .evaluation import edit_distance, normalize_characters, normalize_characters_lenient
from .hayamimi_itn import convert


def normalized_trial(text: str, *, terminal_stop: bool) -> str:
    """A separate presentation layer: CJK numbers and an optional sentence stop.

    No reference text, dictionary learned from evaluation, or observed mutation.
    Adding a stop is a readability hypothesis, not acoustic punctuation evidence.
    """
    value = convert(text, "ja")
    if terminal_stop and value.strip() and value.rstrip()[-1] not in "。！？.!?…」』）)]":
        value = value.rstrip() + "。"
    return value


def measure(reference: str, text: str) -> dict[str, Any]:
    strict_ref, strict_text = normalize_characters(reference), normalize_characters(text)
    if not strict_ref:
        raise ValueError("a nonempty reference is required; missing labels cannot be skipped")
    return {
        "strict_errors": edit_distance(strict_ref, strict_text),
        "strict_units": len(strict_ref),
        "strict_exact": strict_ref == strict_text,
        "raw_errors": edit_distance(reference, text),
        "raw_units": len(reference),
        "raw_exact": reference == text,
        "lenient_errors": edit_distance(
            normalize_characters_lenient(reference), normalize_characters_lenient(text)
        ),
    }


def surface_review(reference: str, text: str) -> str:
    """Classify safe surface equivalence, never invent a semantic accuracy score.

    Latin proper names versus katakana require a reviewed alias or listening;
    a pronunciation generator alone cannot certify equivalence.
    """
    if reference == text:
        return "文字列一致"

    def key(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return "".join(
            chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c
            for c in value
            if not c.isspace() and not unicodedata.category(c).startswith(("P", "S"))
        )

    if key(reference) == key(text):
        return "表記差のみ（空白・記号・大小文字・かな）"
    return "要聴取・意味確認（文字差だけでは誤認識と判定しない）"


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Include provisional rows. Unknown cached statuses are not assumed accepted."""
    if not rows:
        raise ValueError("cannot summarize an empty cohort")
    accepted = [r for r in rows if r["status"] == "accepted"]
    units = sum(r["strict_units"] for r in rows)
    errors = sum(r["strict_errors"] for r in rows)
    return {
        "count": len(rows),
        "strict_errors": errors,
        "strict_units": units,
        "strict_cer": errors / units,
        "strict_exact_count": sum(r["strict_exact"] for r in rows),
        "strict_exact_rate": sum(r["strict_exact"] for r in rows) / len(rows),
        "raw_cer": sum(r["raw_errors"] for r in rows) / sum(r["raw_units"] for r in rows),
        "raw_exact_count": sum(r["raw_exact"] for r in rows),
        "raw_exact_rate": sum(r["raw_exact"] for r in rows) / len(rows),
        "primary_metric": "literal-codepoint-CER",
        "primary_cer": sum(r["raw_errors"] for r in rows) / sum(r["raw_units"] for r in rows),
        "provisional_rate": sum(r["status"] == "provisional" for r in rows) / len(rows),
        "accepted_raw_cer": (
            sum(r["raw_errors"] for r in accepted) / sum(r["raw_units"] for r in accepted)
            if accepted
            else None
        ),
        "accepted_raw_exact_rate": (
            sum(r["raw_exact"] for r in accepted) / len(accepted) if accepted else None
        ),
        "provisional_count": sum(r["status"] == "provisional" for r in rows),
        "unknown_status_count": sum(r["status"] == "unknown" for r in rows),
        "accepted_count": len(accepted),
        "accepted_strict_cer": (
            sum(r["strict_errors"] for r in accepted) / sum(r["strict_units"] for r in accepted)
            if accepted
            else None
        ),
        "all_exact": errors == 0 and all(r["raw_exact"] for r in rows),
    }
