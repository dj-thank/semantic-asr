from __future__ import annotations

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.distillation import (
    MultiTeacherConfig,
    TeacherJudgment,
    aggregate_teacher_judgments,
    candidate_set_digest,
    consensus_to_ranker_example,
)


def _candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence("a", "料金は3000円です"),
        CandidateEvidence("b", "料金は30000円です"),
        CandidateEvidence("c", "料金は3000円でした"),
    ]


def test_multi_teacher_consensus_is_candidate_locked() -> None:
    candidates = _candidates()
    digest = candidate_set_digest(candidates)
    judgments = [
        TeacherJudgment(
            teacher="teacher-a",
            candidate_set_sha256=digest,
            scores={"a": 3.0, "b": -1.0, "c": 0.2},
            score_kind="logit",
        ),
        TeacherJudgment(
            teacher="teacher-b",
            candidate_set_sha256=digest,
            scores={"a": -0.1, "b": -2.0, "c": -0.4},
            score_kind="log_likelihood",
            reliability=0.8,
        ),
    ]
    consensus = aggregate_teacher_judgments(candidates, judgments)
    assert consensus.usable_for_distillation
    assert consensus.preference_distribution["a"] > consensus.preference_distribution["b"]
    assert sum(consensus.preference_distribution.values()) == pytest.approx(1.0)
    assert sum(consensus.teacher_weights.values()) == pytest.approx(1.0)
    example = consensus_to_ranker_example(
        example_id="distilled",
        candidates=candidates,
        consensus=consensus,
    )
    assert example.losses["a"] < example.losses["b"]


def test_teacher_cannot_score_a_different_candidate_set() -> None:
    candidates = _candidates()
    judgment = TeacherJudgment(
        teacher="teacher-a",
        candidate_set_sha256="a" * 64,
        scores={"a": 1.0, "b": 0.0, "c": -1.0},
        score_kind="logit",
    )
    with pytest.raises(ValueError, match="different candidate set"):
        aggregate_teacher_judgments(candidates, [judgment])


def test_abstention_and_disagreement_block_distillation() -> None:
    candidates = _candidates()
    digest = candidate_set_digest(candidates)
    judgments = [
        TeacherJudgment(
            teacher="teacher-a",
            candidate_set_sha256=digest,
            scores={"a": 8.0, "b": -8.0, "c": -8.0},
            score_kind="logit",
        ),
        TeacherJudgment(
            teacher="teacher-b",
            candidate_set_sha256=digest,
            scores={"a": -8.0, "b": 8.0, "c": -8.0},
            score_kind="logit",
        ),
        TeacherJudgment(
            teacher="teacher-c",
            candidate_set_sha256=digest,
            scores={"a": 0.0, "b": 0.0, "c": 0.0},
            score_kind="preference",
            abstained=True,
        ),
    ]
    consensus = aggregate_teacher_judgments(
        candidates,
        judgments,
        config=MultiTeacherConfig(maximum_disagreement=0.05),
    )
    assert not consensus.usable_for_distillation
    assert "teacher-disagreement" in consensus.reasons
    assert consensus.abstained_teachers == ("teacher-c",)
    with pytest.raises(ValueError, match="not usable"):
        consensus_to_ranker_example(
            example_id="blocked",
            candidates=candidates,
            consensus=consensus,
        )


def test_teacher_must_score_every_existing_candidate_once() -> None:
    candidates = _candidates()
    digest = candidate_set_digest(candidates)
    judgment = TeacherJudgment(
        teacher="teacher-a",
        candidate_set_sha256=digest,
        scores={"a": 1.0, "b": 0.0},
        score_kind="logit",
    )
    with pytest.raises(ValueError, match="every candidate ID"):
        aggregate_teacher_judgments(candidates, [judgment])
