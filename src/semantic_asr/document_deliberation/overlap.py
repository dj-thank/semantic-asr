"""Overlap projection and compatibility scoring."""

from __future__ import annotations

from collections.abc import Sequence
from difflib import SequenceMatcher

from ..candidate_pool import lenient_surface_key
from ..contracts import sha256_json
from .config import DocumentBeamConfig
from .decision_types import _BeamState
from .overlap_types import OverlapCompatibility
from .path_types import DocumentPathHypothesis
from .window_types import WindowPathOption, WindowPathSet


def _normalized_overlap(value: str) -> str:
    return lenient_surface_key(value)


def _overlap_receipt(
    left: WindowPathOption,
    right: WindowPathOption,
    retained_left: WindowPathOption,
    retained_right: WindowPathOption,
    *,
    config: DocumentBeamConfig,
) -> OverlapCompatibility | None:
    start = max(left.start_ms, right.start_ms)
    end = min(left.end_ms, right.end_ms)
    if end <= start:
        return None
    left_text = _normalized_overlap(left.overlap_text(start, end))
    right_text = _normalized_overlap(right.overlap_text(start, end))
    retained_left_text = _normalized_overlap(retained_left.overlap_text(start, end))
    retained_right_text = _normalized_overlap(retained_right.overlap_text(start, end))

    compared = min(len(left_text), len(right_text))
    retained_compared = min(len(retained_left_text), len(retained_right_text))
    similarity = (
        SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
        if compared >= config.minimum_overlap_characters
        else None
    )
    retained_similarity = (
        SequenceMatcher(
            None,
            retained_left_text,
            retained_right_text,
            autojunk=False,
        ).ratio()
        if retained_compared >= config.minimum_overlap_characters
        else None
    )
    delta = (
        0.0
        if similarity is None or retained_similarity is None
        else similarity - retained_similarity
    )
    compatible = not (
        similarity is not None
        and retained_similarity is not None
        and retained_similarity - similarity > config.maximum_overlap_similarity_regression
    )
    return OverlapCompatibility(
        left_window_index=left.window_index,
        right_window_index=right.window_index,
        overlap_start_ms=start,
        overlap_end_ms=end,
        left_text_sha256=sha256_json({"text": left_text}),
        right_text_sha256=sha256_json({"text": right_text}),
        similarity=similarity,
        retained_similarity=retained_similarity,
        similarity_delta=delta,
        compared_characters=compared,
        compatible=compatible,
    )


def _mean_audio_support(options: Sequence[WindowPathOption]) -> float:
    rows = [
        (option.coverage_ms, option.path.mean_audio_support)
        for option in options
        if option.path.mean_audio_support > -1.0
    ]
    if not rows:
        return -1.0
    total = sum(weight for weight, _ in rows)
    return sum(weight * value for weight, value in rows) / total


def _document_hypothesis(state: _BeamState) -> DocumentPathHypothesis:
    return DocumentPathHypothesis(
        options=state.options,
        overlap_receipts=state.overlap_receipts,
        base_score=state.base_score,
        mean_audio_support=_mean_audio_support(state.options),
        final_score=state.base_score,
    )


def _total_pair_overlap(window_sets: Sequence[WindowPathSet]) -> int:
    return sum(
        max(
            0,
            min(left.retained.end_ms, right.retained.end_ms)
            - max(left.retained.start_ms, right.retained.start_ms),
        )
        for left_index, left in enumerate(window_sets)
        for right in window_sets[left_index + 1 :]
    )
