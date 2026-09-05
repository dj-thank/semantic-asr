from __future__ import annotations

import pytest

from semantic_asr.context_phonetic_experiment.context_scorer import (
    ContextCandidate,
    GlobalSequenceCandidateContextAdapter,
)
from semantic_asr.context_phonetic_experiment.protocol import FrozenContextSnapshot
from semantic_asr.global_scorer import CallableGlobalSequenceScorer


def context() -> FrozenContextSnapshot:
    return FrozenContextSnapshot(
        context_id="ordered:case-a",
        left_context="保留中です。",
        right_context="承認後に統合します。",
        topic_summary="変更管理",
        entity_ids=("entity-a",),
        source_case_id="case-a",
    )


def test_complete_path_scorer_adapter_binds_each_candidate_and_context() -> None:
    scorer = CallableGlobalSequenceScorer(
        lambda path, document: (
            1.0
            if "まだ" in "".join(arc.text for arc in path) and "承認後" in document.right_context
            else -1.0
        ),
        source="complete-path-fixture",
        profile_digest="a" * 64,
    )
    adapter = GlobalSequenceCandidateContextAdapter(scorer)
    candidates = (
        ContextCandidate(candidate_id="mada", text="まだ"),
        ContextCandidate(candidate_id="mata", text="また"),
    )

    rows = adapter.score_many(candidates, context=context())

    assert {row.candidate_id for row in rows} == {"mada", "mata"}
    assert next(row for row in rows if row.candidate_id == "mada").value == 1.0
    assert next(row for row in rows if row.candidate_id == "mata").value == -1.0
    assert all(row.context_digest == context().digest for row in rows)
    assert all(row.scorer_profile_digest == "a" * 64 for row in rows)


def test_adapter_requires_immutable_scorer_identity() -> None:
    class UnidentifiedScorer:
        def score(self, path, *, context):
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="frozen scorer source"):
        GlobalSequenceCandidateContextAdapter(UnidentifiedScorer())  # type: ignore[arg-type]
