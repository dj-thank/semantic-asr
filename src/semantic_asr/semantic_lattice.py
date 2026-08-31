from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from .contracts import CandidateEvidence
from .japanese import mora_sequence, to_katakana

AlignmentLevel = Literal["surface", "mora"]

_PARTICLES = frozenset("はがをにへとでのもやかねよぞさなってからまでよりしかば")
_NUMBER_RE = re.compile(r"[0-9０-９〇一二三四五六七八九十百千万億兆]")
_DATE_TIME_RE = re.compile(r"(?:\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,2})?|\d{1,2}:\d{2}|[年月日時分秒])")
_CURRENCY_RE = re.compile(r"[¥￥$€£]|円|ドル|ユーロ|ポンド")
_PERCENT_RE = re.compile(r"%|％|パーセント")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+._/#-]*")
_NEGATION_RE = re.compile(r"(?:ない|なく|なかった|ません|ませんでした|じゃない|ではない|ぬ|ず)")
_MODALITY_RE = re.compile(r"(?:かもしれない|はず|べき|予定|つもり|と思う|と思います)")
_DISFLUENCY_RE = re.compile(r"(?:えー+|えっと|ええと|あの+|その+|まあ+|うーん|んー)")
_KANJI_RE = re.compile(r"[\u3400-\u9fff]{2,}")
_KATAKANA_TERM_RE = re.compile(r"[\u30A0-\u30FFー]{3,}")


@dataclass(frozen=True, slots=True)
class Alternative:
    candidate_id: str
    units: tuple[str, ...]
    surface_text: str
    posterior: float
    source: str


@dataclass(frozen=True, slots=True)
class SemanticIsland:
    start: int
    end: int
    pivot_units: tuple[str, ...]
    alternatives: tuple[Alternative, ...]
    kinds: tuple[str, ...]
    posterior_ambiguity: float
    semantic_criticality: float
    expected_information_gain: float
    start_ms: int | None = None
    end_ms: int | None = None
    timing_source: Literal["exact", "proportional", "unavailable"] = "unavailable"


@dataclass(frozen=True, slots=True)
class ConsensusSpan:
    start: int
    end: int
    units: tuple[str, ...]
    support: float = 1.0


@dataclass(frozen=True, slots=True)
class SemanticLattice:
    pivot_candidate_id: str
    alignment_level: AlignmentLevel
    pivot_units: tuple[str, ...]
    locked_consensus: tuple[ConsensusSpan, ...]
    contradiction_islands: tuple[SemanticIsland, ...]


def _reading(candidate: CandidateEvidence) -> str | None:
    if candidate.reading:
        return candidate.reading
    metadata_reading = candidate.metadata.get("reading")
    return str(metadata_reading) if metadata_reading else None


def _has_mora_shadow(candidate: CandidateEvidence) -> bool:
    return bool(candidate.mora_units or _reading(candidate))


def _surface_units(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    return tuple(character for character in normalized if not character.isspace())


def _mora_units(candidate: CandidateEvidence) -> tuple[str, ...]:
    if candidate.mora_units:
        return tuple(unit.kana for unit in candidate.mora_units)
    reading = _reading(candidate)
    return tuple(mora_sequence(reading)) if reading else ()


def _candidate_units(candidate: CandidateEvidence, level: AlignmentLevel) -> tuple[str, ...]:
    return _mora_units(candidate) if level == "mora" else _surface_units(candidate.text)


def _difference_intervals(pivot: Sequence[str], candidate: Sequence[str]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 == i2:
            if i1 < len(pivot):
                i2 = i1 + 1
            elif i1 > 0:
                i1 -= 1
        intervals.append((i1, i2))
    return intervals


def _merge_intervals(
    intervals: Iterable[tuple[int, int]], *, join_gap: int
) -> list[tuple[int, int]]:
    ordered = sorted((max(0, start), max(start, end)) for start, end in intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + join_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _candidate_slice(
    pivot: Sequence[str], candidate: Sequence[str], start: int, end: int
) -> tuple[str, ...]:
    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    output: list[str] = []
    for _tag, i1, i2, j1, j2 in matcher.get_opcodes():
        overlaps = max(i1, start) < min(i2, end)
        insertion_at_boundary = i1 == i2 and start <= i1 <= end
        if overlaps or insertion_at_boundary:
            output.extend(candidate[j1:j2])
    return tuple(output)


def _classify(pivot_units: Sequence[str], alternatives: Iterable[Sequence[str]]) -> tuple[str, ...]:
    texts = ["".join(pivot_units), *("".join(value) for value in alternatives)]
    combined = "".join(texts)
    kinds: list[str] = []
    if _NUMBER_RE.search(combined):
        kinds.append("number-or-quantity")
    if _DATE_TIME_RE.search(combined):
        kinds.append("date-or-time")
    if _CURRENCY_RE.search(combined):
        kinds.append("currency")
    if _PERCENT_RE.search(combined):
        kinds.append("percentage")
    if _NEGATION_RE.search(combined):
        kinds.append("negation-meaning-flip")
    if _MODALITY_RE.search(combined):
        kinds.append("modality-or-intent")
    if _DISFLUENCY_RE.search(combined):
        kinds.append("disfluency-or-repair")
    if any(len(text) <= 4 and any(character in _PARTICLES for character in text) for text in texts):
        kinds.append("particle-or-functional")
    if _LATIN_RE.search(combined):
        kinds.append("latin-acronym-or-term")
    if _KANJI_RE.search(combined) or _KATAKANA_TERM_RE.search(combined):
        kinds.append("entity-or-domain-term")
    if any(character in "ンッー" for character in to_katakana(combined)):
        kinds.append("special-mora")
    if not kinds:
        kinds.append("phonetic-or-punctuation")
    return tuple(dict.fromkeys(kinds))


def _criticality(kinds: Sequence[str]) -> float:
    weights = {
        "negation-meaning-flip": 1.00,
        "number-or-quantity": 1.00,
        "date-or-time": 1.00,
        "currency": 1.00,
        "percentage": 0.98,
        "entity-or-domain-term": 0.93,
        "latin-acronym-or-term": 0.91,
        "modality-or-intent": 0.90,
        "special-mora": 0.82,
        "particle-or-functional": 0.76,
        "disfluency-or-repair": 0.72,
        "phonetic-or-punctuation": 0.45,
    }
    return max(weights.get(kind, 0.45) for kind in kinds)


def _ambiguity(alternatives: Sequence[Alternative]) -> float:
    mass_by_units: dict[tuple[str, ...], float] = {}
    for alternative in alternatives:
        mass_by_units[alternative.units] = (
            mass_by_units.get(alternative.units, 0.0) + alternative.posterior
        )
    total = sum(mass_by_units.values())
    if total <= 0 or len(mass_by_units) <= 1:
        return 0.0
    probabilities = [mass / total for mass in mass_by_units.values()]
    entropy = -sum(probability * math.log(probability + 1e-12) for probability in probabilities)
    return min(1.0, entropy / math.log(len(probabilities)))


def _map_exact_time(
    start: int,
    end: int,
    timeline: list[dict[str, object]] | None,
    *,
    level: AlignmentLevel,
) -> tuple[int | None, int | None]:
    if not timeline:
        return None, None
    overlapping: list[tuple[int, int]] = []
    for row in timeline:
        try:
            if level == "mora":
                unit_start = int(row.get("moraStart", row.get("unitStart", row.get("index"))))
                unit_end = int(row.get("moraEnd", row.get("unitEnd", unit_start + 1)))
            else:
                unit_start = int(row["charStart"])
                unit_end = int(row["charEnd"])
            start_ms = int(row["startMs"])
            end_ms = int(row["endMs"])
        except (KeyError, TypeError, ValueError):
            continue
        if max(start, unit_start) < min(end, unit_end):
            overlapping.append((start_ms, end_ms))
    if not overlapping:
        return None, None
    return min(value[0] for value in overlapping), max(value[1] for value in overlapping)


def _time_range(
    start: int,
    end: int,
    unit_count: int,
    *,
    timeline: list[dict[str, object]] | None,
    level: AlignmentLevel,
    segment_start_ms: int | None,
    segment_end_ms: int | None,
) -> tuple[int | None, int | None, str]:
    exact_start, exact_end = _map_exact_time(start, end, timeline, level=level)
    if exact_start is not None and exact_end is not None:
        return exact_start, exact_end, "exact"
    if (
        segment_start_ms is not None
        and segment_end_ms is not None
        and segment_end_ms > segment_start_ms
        and unit_count > 0
    ):
        duration = segment_end_ms - segment_start_ms
        approximate_start = segment_start_ms + round(duration * start / unit_count)
        approximate_end = segment_start_ms + round(duration * end / unit_count)
        return approximate_start, max(approximate_start + 1, approximate_end), "proportional"
    return None, None, "unavailable"


def build_semantic_lattice(
    candidates: list[CandidateEvidence],
    *,
    posterior: Mapping[str, float] | None = None,
    pivot_candidate_id: str | None = None,
    timeline: list[dict[str, object]] | None = None,
    segment_start_ms: int | None = None,
    segment_end_ms: int | None = None,
    join_gap_units: int = 1,
) -> SemanticLattice:
    if not candidates:
        raise ValueError("at least one candidate is required")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    level: AlignmentLevel = (
        "mora" if all(_has_mora_shadow(candidate) for candidate in candidates) else "surface"
    )
    if pivot_candidate_id is None:
        pivot = max(
            candidates,
            key=lambda candidate: (
                float("-inf") if candidate.acoustic is None else candidate.acoustic,
                candidate.candidate_id,
            ),
        )
    else:
        try:
            pivot = next(
                candidate
                for candidate in candidates
                if candidate.candidate_id == pivot_candidate_id
            )
        except StopIteration as exc:
            raise ValueError("pivot candidate is absent") from exc
    units_by_id = {
        candidate.candidate_id: _candidate_units(candidate, level) for candidate in candidates
    }
    pivot_units = units_by_id[pivot.candidate_id]
    intervals = _merge_intervals(
        (
            interval
            for candidate in candidates
            if candidate.candidate_id != pivot.candidate_id
            for interval in _difference_intervals(pivot_units, units_by_id[candidate.candidate_id])
        ),
        join_gap=join_gap_units,
    )

    if posterior is None:
        probability = 1.0 / len(candidates)
        posterior_map = {candidate.candidate_id: probability for candidate in candidates}
    else:
        total = sum(
            max(0.0, float(posterior.get(candidate.candidate_id, 0.0))) for candidate in candidates
        )
        if total <= 0:
            raise ValueError("posterior mass must be positive")
        posterior_map = {
            candidate.candidate_id: max(0.0, float(posterior.get(candidate.candidate_id, 0.0)))
            / total
            for candidate in candidates
        }

    islands: list[SemanticIsland] = []
    for start, end in intervals:
        alternatives = tuple(
            Alternative(
                candidate_id=candidate.candidate_id,
                units=_candidate_slice(
                    pivot_units, units_by_id[candidate.candidate_id], start, end
                ),
                surface_text=candidate.text,
                posterior=posterior_map[candidate.candidate_id],
                source=candidate.evidence_source,
            )
            for candidate in candidates
        )
        kinds = _classify(
            pivot_units[start:end],
            (alternative.units for alternative in alternatives),
        )
        ambiguity = _ambiguity(alternatives)
        criticality = _criticality(kinds)
        start_ms, end_ms, timing_source = _time_range(
            start,
            end,
            len(pivot_units),
            timeline=timeline,
            level=level,
            segment_start_ms=segment_start_ms,
            segment_end_ms=segment_end_ms,
        )
        islands.append(
            SemanticIsland(
                start=start,
                end=end,
                pivot_units=tuple(pivot_units[start:end]),
                alternatives=alternatives,
                kinds=kinds,
                posterior_ambiguity=ambiguity,
                semantic_criticality=criticality,
                expected_information_gain=ambiguity * criticality,
                start_ms=start_ms,
                end_ms=end_ms,
                timing_source=timing_source,  # type: ignore[arg-type]
            )
        )

    consensus: list[ConsensusSpan] = []
    cursor = 0
    for start, end in intervals:
        if cursor < start:
            consensus.append(ConsensusSpan(cursor, start, tuple(pivot_units[cursor:start])))
        cursor = max(cursor, end)
    if cursor < len(pivot_units):
        consensus.append(ConsensusSpan(cursor, len(pivot_units), tuple(pivot_units[cursor:])))
    return SemanticLattice(
        pivot_candidate_id=pivot.candidate_id,
        alignment_level=level,
        pivot_units=tuple(pivot_units),
        locked_consensus=tuple(span for span in consensus if span.units),
        contradiction_islands=tuple(islands),
    )


def semantic_change_warnings(observed_text: str, normalized_text: str) -> tuple[str, ...]:
    if unicodedata.normalize("NFKC", observed_text) == unicodedata.normalize(
        "NFKC", normalized_text
    ):
        return ()
    lattice = build_semantic_lattice(
        [
            CandidateEvidence("observed", observed_text, acoustic=1.0),
            CandidateEvidence("normalized", normalized_text, acoustic=0.0),
        ],
        posterior={"observed": 0.5, "normalized": 0.5},
        pivot_candidate_id="observed",
    )
    warnings = [
        kind
        for island in lattice.contradiction_islands
        for kind in island.kinds
        if kind
        in {
            "negation-meaning-flip",
            "number-or-quantity",
            "date-or-time",
            "currency",
            "percentage",
            "entity-or-domain-term",
            "modality-or-intent",
            "particle-or-functional",
        }
    ]
    return tuple(dict.fromkeys(warnings))
