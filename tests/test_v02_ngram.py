from __future__ import annotations

import tempfile
from pathlib import Path

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.ngram import (
    NGramCandidateRanker,
    NGramLanguageModel,
    WeightedNGramModel,
    tokenize_ngram_text,
)


def test_character_ngram_prefers_in_domain_japanese_candidate() -> None:
    model = NGramLanguageModel(order=4, mode="character", alpha=0.05).fit(
        [
            "料金は3000円です",
            "料金は3000円です",
            "合計は3000円です",
            "明日の料金は3000円です",
        ]
    )
    correct = model.score("料金は3000円です")
    wrong = model.score("料金は30000円です")
    assert correct.average_log_probability > wrong.average_log_probability
    assert correct.unknown_token_count <= wrong.unknown_token_count


def test_mora_ngram_and_subword_tokenization() -> None:
    mora = tokenize_ngram_text("きょうはがっこうへいきます", "mora")
    assert "ッ" in mora
    assert "キョ" in mora
    subwords = tokenize_ngram_text("Qwen3-ASRで東京へ行く", "subword")
    assert any(token == "東京" for token in subwords)
    assert any(token.startswith("qwen") for token in subwords)


def test_ngram_model_roundtrip_preserves_scores_and_digest() -> None:
    model = NGramLanguageModel(order=3, mode="mora", alpha=0.1).fit(
        ["きょうはがっこうへいきます", "あしたはがっこうへいきます"]
    )
    before = model.score("きょうはがっこうへいきます")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mora-ngram.json"
        model.save(path)
        restored = NGramLanguageModel.load(path)
    after = restored.score("きょうはがっこうへいきます")
    assert restored.digest == model.digest
    assert after.total_log_probability == before.total_log_probability


def test_ngram_ensemble_implements_candidate_ranker_contract() -> None:
    character = NGramLanguageModel(order=4, mode="character").fit(
        ["明日は行きません", "明日は行きません", "今日は行きます"]
    )
    mora = NGramLanguageModel(order=3, mode="mora").fit(
        ["あしたはいきません", "あしたはいきません", "きょうはいきます"]
    )
    ranker = NGramCandidateRanker(
        [
            WeightedNGramModel("character", character, 0.6),
            WeightedNGramModel("mora", mora, 0.4),
        ]
    )
    candidates = [
        CandidateEvidence("correct", "明日は行きません", reading="あしたはいきません"),
        CandidateEvidence("wrong", "明日は行きます", reading="あしたはいきます"),
    ]
    scores = ranker.score(candidates)
    assert scores["correct"] > scores["wrong"]
