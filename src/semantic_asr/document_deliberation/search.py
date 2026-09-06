"""Bounded beam search over window-path alternatives."""

from __future__ import annotations

from collections.abc import Sequence

from .config import DocumentBeamConfig
from .decision_types import _BeamState
from .overlap import (
    _document_hypothesis,
    _mean_audio_support,
    _overlap_receipt,
    _total_pair_overlap,
)
from .overlap_types import OverlapCompatibility
from .path_types import DocumentPathHypothesis
from .window_types import WindowPathSet


def _base_document_paths(
    window_sets: Sequence[WindowPathSet],
    *,
    config: DocumentBeamConfig,
) -> tuple[DocumentPathHypothesis, ...]:
    total_coverage = sum(window.retained.coverage_ms for window in window_sets)
    total_pair_overlap = max(1, _total_pair_overlap(window_sets))
    retained = tuple(window.retained for window in window_sets)
    beam = [_BeamState((), (), 0.0, 0, 0)]
    for index, window in enumerate(window_sets):
        expanded: list[_BeamState] = []
        for state in beam:
            for option in window.options:
                changed = state.changed_windows + int(option.changed)
                generated = state.generated_windows + int(option.generated)
                if (
                    config.maximum_changed_windows is not None
                    and changed > config.maximum_changed_windows
                ):
                    continue
                if generated > config.maximum_generated_windows:
                    continue
                receipts = state.overlap_receipts
                overlap_delta = 0.0
                compatible = True
                new_receipts: list[OverlapCompatibility] = []
                for prior_index, prior in enumerate(state.options):
                    receipt = _overlap_receipt(
                        prior,
                        option,
                        retained[prior_index],
                        retained[index],
                        config=config,
                    )
                    if receipt is None:
                        continue
                    if not receipt.compatible:
                        compatible = False
                        break
                    new_receipts.append(receipt)
                    overlap_delta += (
                        config.overlap_consistency_weight
                        * (receipt.duration_ms / total_pair_overlap)
                        * receipt.similarity_delta
                    )
                if not compatible:
                    continue
                receipts = (*receipts, *new_receipts)
                score = state.base_score
                score += (option.coverage_ms / total_coverage) * option.local_score_delta
                score += overlap_delta
                if option.changed:
                    score -= config.change_penalty
                if option.generated:
                    score -= config.generated_penalty
                expanded.append(
                    _BeamState(
                        options=(*state.options, option),
                        overlap_receipts=receipts,
                        base_score=score,
                        changed_windows=changed,
                        generated_windows=generated,
                    )
                )
        if not expanded:
            raise ValueError(f"document beam has no compatible path at window {index}")
        expanded.sort(
            key=lambda row: (
                -row.base_score,
                row.changed_windows,
                row.generated_windows,
                tuple(option.digest for option in row.options),
            )
        )
        beam = expanded[: config.beam_size]

    paths = [_document_hypothesis(state) for state in beam]
    retained_path = _retained_hypothesis_with_config(window_sets, config=config)
    if retained_path.digest not in {path.digest for path in paths}:
        paths.append(retained_path)
    paths = sorted(
        {path.digest: path for path in paths}.values(),
        key=lambda path: (-path.base_score, path.changed_window_count, path.digest),
    )
    guarded = [
        path
        for path in paths
        if retained_path.mean_audio_support - path.mean_audio_support
        <= config.maximum_document_audio_regression
    ]
    if retained_path.digest not in {path.digest for path in guarded}:
        guarded.append(retained_path)
    return tuple(
        sorted(
            guarded,
            key=lambda path: (-path.base_score, path.changed_window_count, path.digest),
        )
    )


def _retained_hypothesis_with_config(
    window_sets: Sequence[WindowPathSet],
    *,
    config: DocumentBeamConfig,
) -> DocumentPathHypothesis:
    options = tuple(window.retained for window in window_sets)
    receipts = tuple(
        receipt
        for left_index, left in enumerate(window_sets)
        for right in window_sets[left_index + 1 :]
        if (
            receipt := _overlap_receipt(
                left.retained,
                right.retained,
                left.retained,
                right.retained,
                config=config,
            )
        )
        is not None
    )
    return DocumentPathHypothesis(
        options=options,
        overlap_receipts=receipts,
        base_score=0.0,
        mean_audio_support=_mean_audio_support(options),
        final_score=0.0,
    )
