from dataclasses import replace

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.fusion import FusionConfig, fuse_candidates
from semantic_asr.teachers import (
    DelayedTeacherPolicy,
    OpenAICompatibleRanker,
    _validate_response,
    validate_ollama_endpoint,
    validate_openai_endpoint,
)


def test_teacher_endpoints_are_loopback_only() -> None:
    assert validate_ollama_endpoint("http://127.0.0.1:11434") == "http://127.0.0.1:11434/api/chat"
    assert (
        validate_openai_endpoint("http://localhost:8000")
        == "http://localhost:8000/v1/chat/completions"
    )
    with pytest.raises(ValueError):
        validate_ollama_endpoint("https://example.com/api/chat")
    with pytest.raises(ValueError):
        validate_openai_endpoint("http://example.com/v1/chat/completions")
    with pytest.raises(ValueError):
        validate_openai_endpoint("http://127.0.0.1:8000/other")


def test_teacher_response_requires_exact_candidate_set() -> None:
    probabilities, abstained, entropy = _validate_response(
        {
            "probabilities": [
                {"id": "a", "p": 0.6},
                {"id": "b", "p": 0.4},
            ],
            "abstain": False,
        },
        ["a", "b"],
    )
    assert probabilities == {"a": 0.6, "b": 0.4}
    assert not abstained
    assert 0 < entropy <= 1
    with pytest.raises(ValueError):
        _validate_response(
            {
                "probabilities": [
                    {"id": "a", "p": 0.5},
                    {"id": "a", "p": 0.5},
                ],
                "abstain": False,
            },
            ["a", "b"],
        )


def test_qwen38_ranker_defaults_to_no_thought_persistence_contract() -> None:
    client = OpenAICompatibleRanker()
    assert client.model == "Qwen/Qwen3.8-Flash-Next"
    assert client.endpoint.endswith("/v1/chat/completions")


def test_delayed_policy_only_queries_ambiguous_sets() -> None:
    clear = fuse_candidates(
        [
            CandidateEvidence(
                "a", "A", acoustic=1, mora=1, lexical=1, preservation=1, cross_model=1
            ),
            CandidateEvidence(
                "b", "B", acoustic=0, mora=0, lexical=0, preservation=0, cross_model=0
            ),
        ]
    )
    ambiguous = fuse_candidates(
        [
            CandidateEvidence("a", "A", acoustic=0.51),
            CandidateEvidence("b", "B", acoustic=0.50),
        ],
        replace(FusionConfig(), minimum_evidence_coverage=0.8),
    )
    policy = DelayedTeacherPolicy()
    assert not policy.should_query(clear)
    assert policy.should_query(ambiguous)
