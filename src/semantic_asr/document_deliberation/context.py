"""Leakage-auditable context arms for document deliberation."""

from __future__ import annotations

from collections.abc import Sequence

from ..audio import require_integer
from ..contracts import sha256_json
from ..deliberation_lattice import DocumentContext
from ..longform import LongformResult
from .context_types import ContextArm, FrozenWindowContext


def _bounded(value: str, limit: int, *, suffix: bool) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[-limit:] if suffix else value[:limit]


def _deterministic_shuffle(indices: Sequence[int], seed: str, target: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            indices,
            key=lambda index: sha256_json(
                {"seed": seed, "targetWindowIndex": target, "sourceWindowIndex": index}
            ),
        )
    )


def build_frozen_window_contexts(
    first_pass: LongformResult,
    *,
    arm: ContextArm,
    declared_context: DocumentContext | None = None,
    maximum_left_windows: int = 4,
    maximum_right_windows: int = 4,
    maximum_context_characters: int = 12_000,
    shuffle_seed: str = "semantic-asr-shuffled-context-v1",
) -> tuple[FrozenWindowContext, ...]:
    """Construct control-arm contexts without target-window or corrected-text leakage."""

    first_pass.verify()
    require_integer(maximum_left_windows, name="maximum_left_windows")
    require_integer(maximum_right_windows, name="maximum_right_windows")
    require_integer(maximum_context_characters, name="maximum_context_characters")
    if arm not in {
        "none",
        "declared-only",
        "left-only",
        "bidirectional-offline",
        "shuffled-context",
    }:
        raise ValueError("unknown document context arm")
    declared = declared_context or DocumentContext()
    texts = tuple(segment.observed.text for segment in first_pass.segments)
    seed_digest = sha256_json({"shuffleSeed": shuffle_seed}) if arm == "shuffled-context" else None
    output: list[FrozenWindowContext] = []
    for target in range(len(texts)):
        left_start = max(0, target - maximum_left_windows)
        right_end = min(len(texts), target + 1 + maximum_right_windows)
        left_indices = tuple(range(left_start, target))
        right_indices = tuple(range(target + 1, right_end))
        if arm in {"none", "declared-only"}:
            source_indices: tuple[int, ...] = ()
            left_rows: tuple[str, ...] = ()
            right_rows: tuple[str, ...] = ()
        elif arm == "left-only":
            source_indices = left_indices
            left_rows = tuple(texts[index] for index in source_indices)
            right_rows = ()
        elif arm == "bidirectional-offline":
            source_indices = (*left_indices, *right_indices)
            left_rows = tuple(texts[index] for index in left_indices)
            right_rows = tuple(texts[index] for index in right_indices)
        else:
            candidates = (*left_indices, *right_indices)
            source_indices = _deterministic_shuffle(candidates, shuffle_seed, target)
            midpoint = min(len(left_indices), len(source_indices))
            left_rows = tuple(texts[index] for index in source_indices[:midpoint])
            right_rows = tuple(texts[index] for index in source_indices[midpoint:])

        include_declared = arm != "none"
        left = "\n".join(
            (
                *((declared.left_context,) if include_declared and declared.left_context else ()),
                *left_rows,
            )
        )
        right = "\n".join(
            (
                *right_rows,
                *((declared.right_context,) if include_declared and declared.right_context else ()),
            )
        )
        half = maximum_context_characters // 2
        left = _bounded(left, half, suffix=True)
        right = _bounded(
            right,
            maximum_context_characters - len(left),
            suffix=False,
        )
        context = DocumentContext(
            left_context=left,
            right_context=right,
            topic_summary=declared.topic_summary if include_declared else "",
            entity_ids=declared.entity_ids if include_declared else (),
            metadata={
                **(declared.metadata if include_declared else {}),
                "contextArm": arm,
                "contextTextRole": "untrusted-evidence",
                "targetWindowIndex": target,
                "sourceWindowIndices": source_indices,
                "windowCount": len(texts),
                "firstPassEvidenceSha256": first_pass.evidence_sha256,
                "offline": arm in {"bidirectional-offline", "shuffled-context"},
                "usesFutureFirstPass": any(index > target for index in source_indices),
                "shuffleSeedDigest": seed_digest,
            },
        )
        output.append(
            FrozenWindowContext(
                target_window_index=target,
                arm=arm,
                context=context,
                source_window_indices=source_indices,
                first_pass_evidence_sha256=first_pass.evidence_sha256,
                shuffle_seed_digest=seed_digest,
            )
        )
    return tuple(output)
