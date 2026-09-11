from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic_asr.deliberation_lattice import DocumentContext, LatticeArc
from semantic_asr.document_joint_deliberation import (
    DocumentDeliberationConfig,
    DocumentDeliberationDecision,
    DocumentPathCandidate,
    OverlapPolicy,
    OverlapReceipt,
    WindowPathOption,
)
from semantic_asr.document_ranker_dataset import (
    label_document_decision,
    rank_input_from_candidate,
    write_labeled_groups,
)
from semantic_asr.global_deliberation import DeliberationPolicy, PathHypothesis
from semantic_asr.longform import Window
from semantic_asr.semantic_deliberation import SemanticDeliberationBuild

AUDIO = "a" * 64


def fake_build() -> SemanticDeliberationBuild:
    # Constructing a full valid build is unnecessary for the conversion helper; the option only
    # exposes the source-audio digest through its existing immutable build field at runtime.
    return object.__new__(SemanticDeliberationBuild)


def option(index: int, text: str, *, retained: bool) -> WindowPathOption:
    arc = LatticeArc(
        arc_id=f"arc-{index}-{text}",
        span_id=f"span-{index}",
        text=text,
        origin="first-pass",
        utilities=(),
        source_audio_sha256=AUDIO,
    )
    path = PathHypothesis(
        arcs=(arc,),
        base_score=0.5,
        mean_audio_support=0.7,
        final_score=0.5,
    )
    build = fake_build()
    object.__setattr__(build, "lattice", type("Lattice", (), {"source_audio_sha256": AUDIO})())
    object.__setattr__(build, "digest", "b" * 64)
    return WindowPathOption(
        segment_index=index,
        window=Window(index=index, start_ms=index * 1000, end_ms=(index + 1) * 1000),
        build=build,
        path=path,
        retained_path_digest=path.digest if retained else "c" * 64,
        option_rank=0,
    )


def receipt(index: int, text: str) -> OverlapReceipt:
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return OverlapReceipt(
        left_window_index=None if index == 0 else index - 1,
        right_window_index=index,
        overlap_ms=0,
        method="first-window" if index == 0 else "no-window-overlap",
        right_trim_characters=0,
        matched_characters=0,
        normalized_matched_characters=0,
        similarity=0.0,
        utility=0.0,
        left_text_sha256=None,
        right_text_sha256=digest,
        emitted_text_sha256=digest,
        policy_digest=OverlapPolicy().digest,
    )


def candidate(text: str, *, retained: bool) -> DocumentPathCandidate:
    row = option(0, text, retained=retained)
    return DocumentPathCandidate(
        options=(row,),
        emitted_texts=(text,),
        overlap_receipts=(receipt(0, text),),
        local_score=0.5,
        overlap_score=0.0,
        mean_audio_support=0.7,
        final_score=0.5,
    )


def decision() -> DocumentDeliberationDecision:
    retained = candidate("三千円です。", retained=True)
    alternative = candidate("三万円です。", retained=False)
    context = DocumentContext(topic_summary="費用確認")
    return DocumentDeliberationDecision(
        selected=retained,
        retained=retained,
        alternatives=(retained, alternative),
        status="accepted",
        applied=False,
        margin=0.1,
        reasons=("retained-first-pass-document",),
        first_pass_evidence_sha256="d" * 64,
        config_digest=DocumentDeliberationConfig().digest,
        local_policy_digest=DeliberationPolicy.conservative_default().digest,
        context_digest=context.digest,
        source_audio_sha256=AUDIO,
    )


def test_offline_labeler_uses_runtime_feature_shape_and_reference_only_for_labels() -> None:
    context = DocumentContext(topic_summary="費用確認")
    group = label_document_decision(
        decision(),
        group_id="recording-1",
        reference="三千円です。",
        first_pass_text="三千円です。",
        context=context,
        critical_tokens=("三千円",),
    )

    assert len(group.examples) == 2
    retained = next(row for row in group.examples if row.rank_input.retained_path)
    changed = next(row for row in group.examples if not row.rank_input.retained_path)
    assert retained.character_error_rate == 0.0
    assert retained.critical_error_count == 0
    assert changed.character_error_rate > 0.0
    assert changed.critical_error_count == 1
    assert "三千円です。" not in json.dumps(retained.rank_input.metadata, ensure_ascii=False)


def test_labeler_rejects_context_mismatch() -> None:
    with pytest.raises(ValueError, match="context digests differ"):
        label_document_decision(
            decision(),
            group_id="recording-1",
            reference="三千円です。",
            first_pass_text="三千円です。",
            context=DocumentContext(topic_summary="別の文脈"),
        )


def test_labeled_group_writes_jsonl_and_reference_free_manifest(tmp_path: Path) -> None:
    context = DocumentContext(topic_summary="費用確認")
    group = label_document_decision(
        decision(),
        group_id="recording-1",
        reference="三千円です。",
        first_pass_text="三千円です。",
        context=context,
        critical_tokens=("三千円",),
    )

    output = write_labeled_groups((group,), tmp_path / "train.jsonl")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert len(rows) == 2
    assert manifest["referenceBoundary"] == "references-used-offline-only"
    assert "reference" not in json.dumps(rows, ensure_ascii=False).lower()
