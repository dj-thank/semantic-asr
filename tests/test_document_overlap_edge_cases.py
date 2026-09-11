from __future__ import annotations

from semantic_asr.document_joint_deliberation import OverlapPolicy, resolve_window_overlap


def test_full_window_duplicate_is_retained_by_conservative_default() -> None:
    emitted, receipt = resolve_window_overlap(
        "前置き。同じ発話です。",
        "同じ発話です。",
        left_window_index=0,
        right_window_index=1,
        overlap_ms=500,
        policy=OverlapPolicy(),
    )

    assert emitted == "同じ発話です。"
    assert receipt.method == "full-window-duplicate-retained"
    assert receipt.right_trim_characters == 0


def test_full_window_duplicate_can_be_explicitly_suppressed() -> None:
    emitted, receipt = resolve_window_overlap(
        "前置き。同じ発話です。",
        "同じ発話です。",
        left_window_index=0,
        right_window_index=1,
        overlap_ms=500,
        policy=OverlapPolicy(suppress_full_window_duplicate=True),
    )

    assert emitted == ""
    assert receipt.method == "full-window-duplicate-suppressed"
    assert receipt.right_trim_characters == len("同じ発話です。")


def test_short_generic_suffix_is_not_deleted() -> None:
    emitted, receipt = resolve_window_overlap(
        "前の説明です。",
        "です。次の説明です。",
        left_window_index=0,
        right_window_index=1,
        overlap_ms=300,
        policy=OverlapPolicy(minimum_exact_characters=4),
    )

    assert emitted == "です。次の説明です。"
    assert receipt.method in {"no-safe-match", "ambiguous-conflict"}
