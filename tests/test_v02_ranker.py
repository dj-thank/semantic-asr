from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.advanced_adapters import AdaptiveRerankingAdapter
from semantic_asr.cached_lm import HashedLMProbabilityCache
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.ranker_training import (
    RankerTrainingConfig,
    evaluate_ranker,
    train_pairwise_ranker,
)
from semantic_asr.rerankers import LinearCandidateRanker
from semantic_asr.score_semantics import (
    EvidenceScore,
    ScoreKind,
    probability_score,
    uncalibrated_preference,
)
from semantic_asr.synthetic import synthetic_ranker_example


def test_score_semantics_refuse_uncalibrated_probability_claims() -> None:
    preference = uncalibrated_preference(0.9, source="chat-teacher")
    assert not preference.usable_as_probability
    with pytest.raises(ValueError):
        preference.require_probability()
    calibrated = probability_score(
        0.72,
        source="heldout-ranker",
        calibration_digest="a" * 64,
    )
    assert calibrated.require_probability() == pytest.approx(0.72)
    with pytest.raises(ValueError):
        EvidenceScore(2.0, ScoreKind.PROBABILITY, "bad")


def test_pairwise_ranker_actually_learns_on_japanese_hard_negatives() -> None:
    examples = [
        synthetic_ranker_example(
            "えっと明日は行きません。料金は3000円です。",
            example_id="one",
        ),
        synthetic_ranker_example(
            "学校へ行って切符を買います。",
            example_id="two",
        ),
        synthetic_ranker_example(
            "スーパーでしんぶんを買った。",
            example_id="three",
        ),
    ]
    result = train_pairwise_ranker(
        examples,
        config=RankerTrainingConfig(
            epochs=120,
            learning_rate=0.08,
            l2=0.001,
            seed=9,
        ),
    )
    assert result.after.pairwise_accuracy > result.before.pairwise_accuracy
    assert result.after.mean_logistic_loss < result.before.mean_logistic_loss
    ranker = LinearCandidateRanker(result.profile)
    evaluated = evaluate_ranker(ranker, examples)
    assert evaluated.pairwise_accuracy >= 0.80
    assert result.profile.training_manifest_sha256 == result.training_manifest_sha256


def test_hashed_probability_cache_backoff_and_roundtrip() -> None:
    cache = HashedLMProbabilityCache(
        key=b"0123456789abcdef",
        maximum_context=3,
        backoff_penalty=0.25,
    )
    cache.put([1, 2], 3, 0.5, teacher="offline-12b")
    exact = cache.lookup([1, 2], 3)
    backed_off = cache.lookup([0, 1, 2], 3)
    assert exact.exact
    assert backed_off.matched_order == 2
    assert backed_off.log_probability == pytest.approx(exact.log_probability - 0.25)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cache.json"
        cache.export(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "1,2" not in path.read_text(encoding="utf-8")
        assert payload["entries"][0]["teacher"] == "offline-12b"
        restored = HashedLMProbabilityCache.load(
            path,
            key=b"0123456789abcdef",
        )
        assert restored.lookup([1, 2], 3).log_probability == pytest.approx(exact.log_probability)


class _FakeBase:
    name = "fake"
    model_name = "fake"

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        assert request.hypotheses >= 6
        return [
            CandidateEvidence(
                "good",
                "明日は行きません",
                acoustic=-0.10,
                avg_logprob=-0.10,
                rank=1,
                hypothesis_count=3,
                source="fake",
            ),
            CandidateEvidence(
                "natural",
                "明日は行きます",
                acoustic=-0.12,
                avg_logprob=-0.12,
                rank=2,
                hypothesis_count=3,
                source="fake",
            ),
            CandidateEvidence(
                "wrong",
                "昨日は行きます",
                acoustic=-0.80,
                avg_logprob=-0.80,
                rank=3,
                hypothesis_count=3,
                source="fake",
            ),
        ]


class _TextRanker:
    name = "text-ranker"

    def score(self, candidates, **kwargs):
        return {
            candidate.candidate_id: (
                3.0
                if "行きません" in candidate.text
                else 1.0
                if "明日は" in candidate.text
                else -2.0
            )
            for candidate in candidates
        }


def test_adaptive_reranking_adapter_injects_calibrated_language_evidence() -> None:
    adapter = AdaptiveRerankingAdapter(
        _FakeBase(),
        _TextRanker(),
        maximum_hypotheses=6,
    )
    output = adapter.decode(DecodeRequest("unused.wav", hypotheses=2))
    assert 2 <= len(output) <= 3
    assert output[0].lexical is not None
    assert output[0].metadata["rerankerSource"] == "text-ranker"
    assert output[0].metadata["adaptiveK"]["k"] == len(output)
    assert output[0].metadata["evidenceScores"][0]["kind"] == "logit"
