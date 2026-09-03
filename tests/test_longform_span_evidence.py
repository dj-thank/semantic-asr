from __future__ import annotations

import tempfile
import wave
from pathlib import Path

from semantic_asr.adapters import DecodeRequest
from semantic_asr.candidate_pool import aggregate_surface_candidates, lenient_surface_key
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.longform import (
    SemanticASRTranscriber,
    apply_span_evidence,
    span_agreement,
)


def _candidate(identifier: str, text: str, *, logprob: float, rank: int) -> CandidateEvidence:
    return CandidateEvidence(
        identifier,
        text,
        acoustic=logprob,
        avg_logprob=logprob,
        sequence_score=logprob,
        rank=rank,
        hypothesis_count=3,
        source="fake",
        metadata={"scoreDomain": "fake|window"},
    )


def test_lenient_surface_key_ignores_punctuation_and_width() -> None:
    assert lenient_surface_key("はい、そうです。") == lenient_surface_key("はい そうです")
    assert lenient_surface_key("１５種類") == lenient_surface_key("15種類")
    assert lenient_surface_key("東京") != lenient_surface_key("京都")


def test_lenient_aggregation_pools_punctuation_variants_but_keeps_surface() -> None:
    rows = [
        _candidate("a", "はい、そうです。", logprob=-0.10, rank=1),
        _candidate("b", "はいそうです", logprob=-0.12, rank=2),
        _candidate("c", "いいえ", logprob=-0.90, rank=3),
    ]
    exact = aggregate_surface_candidates(rows)
    lenient = aggregate_surface_candidates(rows, policy="lenient")
    assert len(exact) == 3
    assert len(lenient) == 2
    assert lenient[0].text == "はい、そうです。"
    assert lenient[0].metadata["surfacePolicy"] == "lenient"
    assert set(lenient[0].metadata["surfaceVariants"]) == {"はい、そうです。", "はいそうです"}


def test_span_agreement_and_evidence_never_replace_window_text() -> None:
    window = [
        _candidate("w1", "NHK大阪放送局からお届けするきょうの料理", logprob=-0.2, rank=1),
        _candidate("w2", "NHK大阪放送局から送るきょうの料理", logprob=-0.3, rank=2),
    ]
    span = [_candidate("s1", "お届けする", logprob=-0.05, rank=1)]
    assert span_agreement(window[0].text, span[0].text) == 1.0
    assert span_agreement(window[1].text, span[0].text) < 1.0
    out = apply_span_evidence(window, span)
    assert [row.text for row in out] == [window[0].text, window[1].text]
    assert out[0].cross_model == 1.0
    assert out[0].metadata["spanEvidence"]["rows"] == 1
    assert apply_span_evidence(window, []) == window


class SubSpanAdapter:
    """Full-window decode returns a sentence; any sub-span decode returns a short fragment."""

    name = "subspan-fake"
    model_name = "fixture"
    allow_legacy_cache_identity = True

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        if request.start_ms not in (None, 0) or (
            request.end_ms not in (None,) and request.end_ms < 2_000
        ):
            return [_candidate("frag", "お届けする", logprob=-0.02, rank=1)]
        return [
            _candidate(
                "full-a", "NHK大阪放送局からお届けする、きょうの料理。", logprob=-0.20, rank=1
            ),
            _candidate("full-b", "NHK大阪放送局からお届けするきょうの料理", logprob=-0.21, rank=2),
            _candidate("full-c", "NHK大阪放送局から送る今日の料理", logprob=-0.60, rank=3),
        ]


def test_longform_window_keeps_full_text_when_relisten_runs() -> None:
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "clip.wav"
        with wave.open(str(audio), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(b"\x00\x00" * 32_000)
        result = SemanticASRTranscriber(SubSpanAdapter()).transcribe(audio, duration_ms=2_000)
    assert "NHK大阪放送局" in result.observed_text
    assert result.observed_text != "お届けする"
