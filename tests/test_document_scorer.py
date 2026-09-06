from __future__ import annotations

import pytest

from semantic_asr.deliberation_lattice import DocumentContext, LatticeArc
from semantic_asr.document_scorer import (
    DocumentPromptFormat,
    GlobalScoreNormalization,
    SequenceScorerGlobalAdapter,
)
from semantic_asr.score_types import EvidenceScore, ScoreSemantics
from semantic_asr.sequence_scorers import SequenceScoreResult

MANIFEST = "a" * 64
IDENTITY = "b" * 64


class FakeSequenceScorer:
    name = "fake-causal"

    def __init__(self) -> None:
        self.calls = 0
        self.last_context = ""

    def score(self, candidates, *, context=""):
        self.calls += 1
        self.last_context = context
        output = []
        for candidate in candidates:
            total = -0.2 * len(candidate.text)
            count = max(1, len(candidate.text))
            output.append(
                SequenceScoreResult(
                    candidate_id=candidate.candidate_id,
                    cumulative=EvidenceScore.raw(
                        total,
                        semantics=ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
                        scorer=self.name,
                    ),
                    average=EvidenceScore.raw(
                        total / count,
                        semantics=ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
                        scorer=self.name,
                    ),
                    token_count=count,
                )
            )
        return output


def path(identifier: str, text: str):
    return (
        LatticeArc(
            arc_id=identifier,
            span_id=identifier,
            text=text,
            origin="first-pass",
            utilities=(),
        ),
    )


def normalization() -> GlobalScoreNormalization:
    return GlobalScoreNormalization(
        center=-0.3,
        scale=0.2,
        fitted_manifest_sha256=MANIFEST,
        revision="held-out-r1",
    )


def test_adapter_batches_complete_paths_and_binds_context() -> None:
    scorer = FakeSequenceScorer()
    adapter = SequenceScorerGlobalAdapter(
        scorer,
        normalization(),
        scorer_identity_digest=IDENTITY,
    )
    context = DocumentContext(
        left_context="レビュー待ちです。",
        right_context="承認後に統合します。",
    )

    rows = adapter.score_many(
        (path("a", "まだマージしない"), path("b", "またマージしない")),
        context=context,
    )

    assert scorer.calls == 1
    assert len(rows) == 2
    assert all(-1.0 <= row.value <= 1.0 for row in rows)
    assert all(row.context_digest == context.digest for row in rows)
    assert all(row.profile_digest == adapter.profile_digest for row in rows)
    assert "leftContext" in scorer.last_context
    assert "rightContext" in scorer.last_context


def test_model_identity_must_be_immutable() -> None:
    with pytest.raises(ValueError, match="identity is not immutable"):
        SequenceScorerGlobalAdapter(FakeSequenceScorer(), normalization())


def test_prompt_format_respects_exact_character_budget() -> None:
    prompt = DocumentPromptFormat(maximum_context_characters=90)
    rendered = prompt.render(
        DocumentContext(
            left_context="左" * 1_000,
            right_context="右" * 1_000,
        )
    )

    assert len(rendered) <= 90


def test_adapter_rejects_non_cumulative_score_semantics() -> None:
    class BadScorer(FakeSequenceScorer):
        def score(self, candidates, *, context=""):
            rows = super().score(candidates, context=context)
            row = rows[0]
            rows[0] = SequenceScoreResult(
                candidate_id=row.candidate_id,
                cumulative=EvidenceScore.raw(
                    row.cumulative.value,
                    semantics=ScoreSemantics.LOGIT,
                    scorer=self.name,
                ),
                average=row.average,
                token_count=row.token_count,
            )
            return rows

    adapter = SequenceScorerGlobalAdapter(
        BadScorer(),
        normalization(),
        scorer_identity_digest=IDENTITY,
    )

    with pytest.raises(ValueError, match="cumulative log-likelihood"):
        adapter.score(path("a", "候補"), context=DocumentContext())
