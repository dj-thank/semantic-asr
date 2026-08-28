#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import apply_pr_review_fixes_v2 as review

ROOT = Path(__file__).resolve().parents[1]


def patch_semantic_lattice() -> None:
    path = ROOT / "src/semantic_asr/semantic_lattice.py"
    text = path.read_text(encoding="utf-8")
    old = '''def _candidate_slice(
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
'''
    new = '''def _candidate_slice(
    pivot: Sequence[str], candidate: Sequence[str], start: int, end: int
) -> tuple[str, ...]:
    """Project a pivot-coordinate island onto one candidate sequence.

    `SequenceMatcher` opcodes can span far beyond a contradiction island. Copying
    the whole candidate side of every overlapping opcode therefore leaks locked
    consensus (and, for the pivot itself, the entire sentence) into a local
    hypothesis. Equal blocks are clipped exactly; unequal replacement blocks are
    projected proportionally so merged islands remain local.
    """

    if start < 0 or end < start or end > len(pivot):
        raise ValueError("candidate slice is outside pivot coordinates")
    matcher = SequenceMatcher(a=list(pivot), b=list(candidate), autojunk=False)
    output: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            at_right_edge = i1 == len(pivot) == end and start < end
            if start <= i1 < end or at_right_edge:
                output.extend(candidate[j1:j2])
            continue

        overlap_start = max(i1, start)
        overlap_end = min(i2, end)
        if overlap_start >= overlap_end or tag == "delete":
            continue

        if tag == "equal":
            candidate_start = j1 + (overlap_start - i1)
            candidate_end = j1 + (overlap_end - i1)
        else:  # replace
            pivot_width = max(1, i2 - i1)
            candidate_width = j2 - j1
            candidate_start = j1 + math.floor(
                (overlap_start - i1) * candidate_width / pivot_width
            )
            candidate_end = j1 + math.ceil(
                (overlap_end - i1) * candidate_width / pivot_width
            )
            if candidate_width and candidate_end <= candidate_start:
                candidate_end = min(j2, candidate_start + 1)

        output.extend(candidate[candidate_start:candidate_end])
    return tuple(output)
'''
    if old not in text:
        raise RuntimeError("semantic lattice candidate-slice anchor is missing")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def main() -> int:
    patch_semantic_lattice()
    review.main()
    Path(__file__).unlink(missing_ok=True)
    print("semantic island projection and PR review fixes applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
