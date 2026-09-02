from __future__ import annotations

import hashlib
import json

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.enrichment import (
    EnrichmentConfig,
    SecondEarHypothesis,
    agreement,
    enrich_candidates,
    enrich_manifest_rows,
    load_second_ear,
)
from semantic_asr.ngram import NGramLanguageModel


def test_agreement_ignores_punctuation_and_clips() -> None:
    assert agreement("はい、そうです。", "はいそうです") == 1.0
    assert agreement("はい", "いいえいいえいいえ") == 0.0
    assert 0.0 < agreement("東京へ行きます", "東京に行きます") < 1.0


def _candidates() -> list[CandidateEvidence]:
    return [
        CandidateEvidence(candidate_id="a", text="料金は三千円です", acoustic=-0.1, rank=1),
        CandidateEvidence(candidate_id="b", text="料金は三万円です", acoustic=-0.2, rank=2),
    ]


def test_second_ear_agreement_sets_cross_model() -> None:
    ear = SecondEarHypothesis(
        sample_id="s",
        texts=("料金は三千円です。",),
        model_revision="qwen-revision",
    )
    rows = enrich_candidates(_candidates(), second_ear=ear, config=EnrichmentConfig())
    assert rows[0].cross_model == 1.0
    assert rows[1].cross_model < 1.0
    assert "secondEarText" not in rows[0].metadata
    assert (
        rows[0].metadata["secondEarTextSha256"]
        == hashlib.sha256("料金は三千円です。".encode()).hexdigest()
    )
    assert rows[0].metadata["secondEarSampleId"] == "s"
    assert rows[0].metadata["secondEarRevision"] == "qwen-revision"


def test_second_ear_text_requires_explicit_research_opt_in() -> None:
    ear = SecondEarHypothesis(sample_id="s", texts=("料金は三千円です。",))
    rows = enrich_candidates(
        _candidates(),
        second_ear=ear,
        config=EnrichmentConfig(retain_second_ear_text=True),
    )
    assert rows[0].metadata["secondEarText"] == "料金は三千円です。"


def test_ngram_lexical_is_normalised_within_set() -> None:
    model = NGramLanguageModel(
        order=2,
        mode="character",
        source_sha256="a" * 64,
        source_revision="corpus-revision",
    ).fit(["料金は三千円です"] * 5)
    rows = enrich_candidates(
        _candidates(), second_ear=None, config=EnrichmentConfig(ngram_model=model)
    )
    assert rows[0].lexical == 1.0
    assert 0.0 <= rows[1].lexical < 1.0
    assert "ngramAverageLogProbability" in rows[0].metadata
    assert rows[0].metadata["ngramSourceSha256"] == "a" * 64
    assert rows[0].metadata["ngramSourceRevision"] == "corpus-revision"


def test_second_ear_loader_rejects_revision_mismatch(tmp_path) -> None:
    path = tmp_path / "ear.jsonl"
    path.write_text(
        json.dumps(
            {"sampleId": "s", "hypotheses": ["はい"], "modelRevision": "actual"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="revision"):
        load_second_ear(path, model_revision="claimed")


def test_second_ear_candidate_is_appended_once() -> None:
    ear = SecondEarHypothesis(sample_id="s", texts=("料金は三千円でした",))
    config = EnrichmentConfig(add_second_ear_candidate=True, second_ear_source="ear")
    rows = enrich_candidates(_candidates(), second_ear=ear, config=config)
    assert [row.candidate_id for row in rows] == ["a", "b", "ear:0001"]
    assert rows[-1].source == "ear"
    assert rows[-1].cross_model is None
    assert rows[-1].metadata["secondEarAgreement"] is None
    assert rows[-1].metadata["crossModelEligible"] is False
    assert "secondEarText" not in rows[-1].metadata
    assert len(rows[-1].metadata["secondEarTextSha256"]) == 64
    assert all(row.hypothesis_count == 3 for row in rows)
    duplicate = enrich_candidates(
        _candidates(),
        second_ear=SecondEarHypothesis(sample_id="s", texts=("料金は三千円です",)),
        config=config,
    )
    assert len(duplicate) == 2


def test_manifest_rows_round_trip() -> None:
    row = {
        "sampleId": "s",
        "reference": "料金は三千円です",
        "candidates": [candidate.as_dict() for candidate in _candidates()],
    }
    ear = {"s": SecondEarHypothesis(sample_id="s", texts=("料金は三千円です",))}
    out = enrich_manifest_rows([row], second_ear=ear, config=EnrichmentConfig())
    assert out[0]["candidates"][0]["cross_model"] == 1.0
    assert out[0]["enrichment"]["secondEar"] == "qwen3-asr"
    assert out[0]["enrichment"]["secondEarSampleId"] == "s"
    assert "secondEarText" not in out[0]["enrichment"]
    assert len(out[0]["enrichment"]["secondEarTextSha256"]) == 64
