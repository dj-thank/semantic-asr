from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .contracts import CandidateEvidence
from .mbr import semantic_loss
from .ranker_training import RankerExample


@dataclass(frozen=True, slots=True)
class Corruption:
    text: str
    kind: str
    severity: float
    source_span: tuple[int, int] | None = None


_DIGIT_PATTERN = re.compile(r"\d")
_FILLER_PATTERN = re.compile(r"(?:えー|えっと|あの|その|まあ|うーん)")
_NEGATION_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ません", "ます"),
    ("ない", "ある"),
    ("なかった", "あった"),
    ("できない", "できる"),
    ("行かない", "行く"),
)
_PARTICLE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("を", "に"),
    ("に", "で"),
    ("は", "が"),
    ("へ", "に"),
    ("が", "を"),
)
_MORA_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("っ", "", "geminate-delete"),
    ("ー", "", "long-vowel-delete"),
    ("ん", "", "moraic-nasal-delete"),
    ("きゃ", "きや", "contracted-sound-expand"),
    ("きゅ", "きゆ", "contracted-sound-expand"),
    ("きょ", "きよ", "contracted-sound-expand"),
    ("しゃ", "しや", "contracted-sound-expand"),
    ("しゅ", "しゆ", "contracted-sound-expand"),
    ("しょ", "しよ", "contracted-sound-expand"),
)


def _replace_once(text: str, old: str, new: str) -> tuple[str, tuple[int, int]] | None:
    index = text.find(old)
    if index < 0:
        return None
    return text[:index] + new + text[index + len(old) :], (index, index + len(old))


def generate_hard_negatives(
    reference: str,
    *,
    maximum: int = 12,
    seed: int = 17,
) -> list[Corruption]:
    if not reference:
        raise ValueError("reference must not be empty")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    output: list[Corruption] = []
    seen = {reference}

    for old, new in _NEGATION_REPLACEMENTS:
        result = _replace_once(reference, old, new)
        if result is not None and result[0] not in seen:
            seen.add(result[0])
            output.append(Corruption(result[0], "negation-flip", 1.0, result[1]))

    for old, new in _PARTICLE_REPLACEMENTS:
        result = _replace_once(reference, old, new)
        if result is not None and result[0] not in seen:
            seen.add(result[0])
            output.append(Corruption(result[0], "particle", 0.45, result[1]))

    for old, new, kind in _MORA_REPLACEMENTS:
        result = _replace_once(reference, old, new)
        if result is not None and result[0] not in seen:
            seen.add(result[0])
            output.append(Corruption(result[0], kind, 0.60, result[1]))

    digit_match = _DIGIT_PATTERN.search(reference)
    if digit_match:
        original = int(digit_match.group(0))
        for delta in (1, -1, 10):
            replacement = str((original + delta) % 10)
            text = (
                reference[: digit_match.start()]
                + replacement
                + reference[digit_match.end() :]
            )
            if text not in seen:
                seen.add(text)
                output.append(
                    Corruption(
                        text,
                        "number",
                        1.0,
                        (digit_match.start(), digit_match.end()),
                    )
                )

    filler_match = _FILLER_PATTERN.search(reference)
    if filler_match:
        text = reference[: filler_match.start()] + reference[filler_match.end() :]
        if text not in seen:
            seen.add(text)
            output.append(
                Corruption(
                    text,
                    "filler-delete",
                    0.55,
                    (filler_match.start(), filler_match.end()),
                )
            )

    kana_pairs = (
        ("か", "が"),
        ("た", "だ"),
        ("さ", "ざ"),
        ("は", "ば"),
        ("し", "ち"),
        ("つ", "す"),
    )
    for old, new in kana_pairs:
        result = _replace_once(reference, old, new)
        if result is not None and result[0] not in seen:
            seen.add(result[0])
            output.append(Corruption(result[0], "phonetic-neighbour", 0.35, result[1]))

    rng = random.Random(seed)
    rng.shuffle(output)
    output.sort(key=lambda row: (-row.severity, row.kind, row.text))
    return output[:maximum]


def synthetic_ranker_example(
    reference: str,
    *,
    example_id: str = "synthetic",
    maximum_negatives: int = 8,
    seed: int = 17,
    context: str = "",
) -> RankerExample:
    reference_candidate = CandidateEvidence(
        candidate_id="reference",
        text=reference,
        acoustic=0.92,
        mora=0.94,
        preservation=0.95,
        rank=1,
        hypothesis_count=maximum_negatives + 1,
        avg_logprob=-0.08,
        source="synthetic-reference",
        metadata={"synthetic": True, "corruptionKind": "none"},
    )
    negatives = generate_hard_negatives(
        reference,
        maximum=maximum_negatives,
        seed=seed,
    )
    candidates = [reference_candidate]
    for index, corruption in enumerate(negatives, 2):
        candidates.append(
            CandidateEvidence(
                candidate_id=f"negative-{index:03d}",
                text=corruption.text,
                acoustic=max(0.05, 0.85 - 0.35 * corruption.severity - index * 0.01),
                mora=max(0.05, 0.86 - 0.40 * corruption.severity),
                preservation=max(
                    0.05,
                    0.90
                    - (0.55 if corruption.kind == "filler-delete" else 0.20)
                    * corruption.severity,
                ),
                rank=index,
                hypothesis_count=len(negatives) + 1,
                avg_logprob=-0.12 - 0.28 * corruption.severity - index * 0.01,
                source="synthetic-corruption",
                metadata={
                    "synthetic": True,
                    "corruptionKind": corruption.kind,
                    "corruptionSeverity": corruption.severity,
                    "sourceSpan": list(corruption.source_span)
                    if corruption.source_span is not None
                    else None,
                },
            )
        )
    reference_for_loss = CandidateEvidence("__reference__", reference)
    losses = {
        candidate.candidate_id: semantic_loss(candidate, reference_for_loss)[0]
        for candidate in candidates
    }
    return RankerExample(
        example_id=example_id,
        candidates=tuple(candidates),
        losses=losses,
        context=context,
    )
