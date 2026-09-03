from __future__ import annotations

import wave
from pathlib import Path

from semantic_asr.adapters import DecodeRequest
from semantic_asr.advanced_adapters import utterance_spans
from semantic_asr.api import load_transcriber, transcribe, transcribe_segments
from semantic_asr.contracts import CandidateEvidence

BEGIN = 1000
VOCAB = {1: "はい", 2: "そう", 3: "です", 4: "いいえ"}


def _decode(tokens):
    return "".join(VOCAB.get(int(token), "") for token in tokens)


def test_utterance_spans_parse_whisper_timestamp_groups() -> None:
    tokens = [BEGIN + 0, 1, 2, 3, BEGIN + 50, BEGIN + 50, 4, BEGIN + 80]
    spans = utterance_spans(tokens, timestamp_begin=BEGIN, decode=_decode)
    assert spans == [
        {"startMs": 0, "endMs": 1000, "text": "はいそうです"},
        {"startMs": 1000, "endMs": 1600, "text": "いいえ"},
    ]
    open_ended = utterance_spans([BEGIN + 10, 1, 2], timestamp_begin=BEGIN, decode=_decode)
    assert open_ended == [{"startMs": 200, "endMs": None, "text": "はいそう"}]
    assert utterance_spans([BEGIN + 5, BEGIN + 9], timestamp_begin=BEGIN, decode=_decode) == []


class SpanAdapter:
    name = "span-fake"
    model_name = "fixture"
    device = "cpu"
    compute_type = "int8"
    allow_legacy_cache_identity = True

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        return [
            CandidateEvidence(
                "a",
                "はいそうです いいえ",
                acoustic=0.9,
                avg_logprob=-0.05,
                rank=1,
                hypothesis_count=1,
                source=self.name,
                metadata={
                    "utteranceSpans": [
                        {"startMs": 0, "endMs": 1000, "text": "はいそうです"},
                        {"startMs": 1000, "endMs": 1600, "text": "いいえ"},
                    ]
                },
            )
        ]


def _wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * int(16_000 * seconds))


def test_facade_exposes_absolute_utterances_and_koemo_rows(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _wav(audio, 2.0)
    result = transcribe(audio, adapter=SpanAdapter())
    assert [u.text for u in result.utterances] == ["はいそうです", "いいえ"]
    assert (result.utterances[0].start_ms, result.utterances[0].end_ms) == (0, 1000)
    assert result.utterances[1].segment_index == 1
    outputs = result.write(tmp_path / "out")
    assert "utterances_srt" in outputs and "transcript_json" in outputs
    assert "はいそうです" in (tmp_path / "out" / "clip.utterances.srt").read_text("utf-8")


def test_transcribe_segments_prefers_utterances(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _wav(audio, 2.0)
    warm = load_transcriber("cpu-ja-v1", adapter=SpanAdapter())
    rows = transcribe_segments(audio, transcriber=warm)
    assert rows == [(0.0, 1.0, "はいそうです"), (1.0, 1.6, "いいえ")]
