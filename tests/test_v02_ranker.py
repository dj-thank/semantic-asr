from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.advanced_adapters import AdaptiveRerankingAdapter
from semantic_asr.cached_lm import HashedLMProbabilityCache
from semantic_asr.calibration import CalibrationProfile
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.ranker_training import (
    RankerTrainingConfig,
    evaluate_ranker,
    train_pairwise_ranker,
)
from semantic_asr.rerankers import (
    CrossEncoderCandidateRanker,
    LinearCandidateRanker,
    Qwen3CandidateRanker,
)
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


def test_cross_encoder_requires_and_binds_model_identity(monkeypatch, tmp_path) -> None:
    model = tmp_path / "cross-encoder"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="verified local ranker artifact"):
        CrossEncoderCandidateRanker(str(model))

    torch = ModuleType("torch")
    torch_nn = ModuleType("torch.nn")

    class _Identity:
        pass

    torch_nn.Identity = _Identity  # type: ignore[attr-defined]
    torch.nn = torch_nn  # type: ignore[attr-defined]
    sentence_transformers = ModuleType("sentence_transformers")
    calls: dict[str, object] = {}

    class _CrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls.update({"model": model_name, **kwargs})

    sentence_transformers.CrossEncoder = _CrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    revision = "0123456789abcdef0123456789abcdef01234567"
    ranker = CrossEncoderCandidateRanker(
        "publisher/cross-encoder",
        model_revision=revision,
        runtime_revision="runtime-cross",
        device="cpu",
        batch_size=4,
    )

    assert calls["revision"] == revision
    assert ranker.model_revision == revision
    assert ranker.model_artifact_sha256 is None
    assert ranker.runtime_revision == "runtime-cross"
    assert len(ranker.config_digest) == 64


def test_qwen_ranker_requires_and_binds_model_identity(monkeypatch, tmp_path) -> None:
    model = tmp_path / "qwen-ranker"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="verified local ranker artifact"):
        Qwen3CandidateRanker(str(model))

    torch = ModuleType("torch")
    torch.float16 = object()  # type: ignore[attr-defined]
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.float32 = object()  # type: ignore[attr-defined]
    transformers = ModuleType("transformers")
    tokenizer_calls: dict[str, object] = {}
    model_calls: dict[str, object] = {}

    class _Tokenizer:
        def encode(self, _value: str, **_kwargs: object) -> list[int]:
            return [1]

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> _Tokenizer:
            tokenizer_calls.update({"model": model_name, **kwargs})
            return _Tokenizer()

    class _LoadedModel:
        def eval(self) -> _LoadedModel:
            return self

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> _LoadedModel:
            model_calls.update({"model": model_name, **kwargs})
            return _LoadedModel()

    transformers.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForCausalLM = _AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    revision = "fedcba9876543210fedcba9876543210fedcba98"
    ranker = Qwen3CandidateRanker(
        model="publisher/qwen-ranker",
        model_revision=revision,
        runtime_revision="runtime-qwen",
        device_map="cpu",
        dtype="auto",
    )

    assert tokenizer_calls["revision"] == revision
    assert model_calls["revision"] == revision
    assert ranker.model_revision == revision
    assert ranker.model_artifact_sha256 is None
    assert ranker.runtime_revision == "runtime-qwen"
    assert len(ranker.config_digest) == 64


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


def test_uncalibrated_reranker_can_reorder_but_not_enter_fusion() -> None:
    adapter = AdaptiveRerankingAdapter(
        _FakeBase(),
        _TextRanker(),
        maximum_hypotheses=6,
    )
    output = adapter.decode(DecodeRequest("unused.wav", hypotheses=2))
    assert 2 <= len(output) <= 3
    assert output[0].text == "明日は行きません"
    assert output[0].lexical is None
    assert output[0].metadata["rerankerEvidenceInjected"] is False
    assert output[0].metadata["adaptiveK"]["k"] == len(output)
    scores = output[0].metadata["evidenceScores"]
    assert len(scores) == 1
    assert scores[0]["kind"] == "logit"
    assert scores[0]["calibrated"] is False


def test_heldout_calibrated_reranker_may_enter_lexical_fusion() -> None:
    profile = CalibrationProfile(
        name="fixture-heldout-logit",
        input_kind="logit",
        temperature=1.0,
        version="test",
    )
    adapter = AdaptiveRerankingAdapter(
        _FakeBase(),
        _TextRanker(),
        maximum_hypotheses=6,
        calibration_profile=profile,
    )
    output = adapter.decode(DecodeRequest("unused.wav", hypotheses=2))
    assert output[0].lexical is not None
    assert output[0].metadata["rerankerEvidenceInjected"] is True
    scores = output[0].metadata["evidenceScores"]
    assert [row["kind"] for row in scores] == ["logit", "probability"]
    assert scores[1]["calibrated"] is True
    assert scores[1]["calibrationDigest"] == profile.digest
