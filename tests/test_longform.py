from __future__ import annotations

import json
import tempfile
from pathlib import Path

from semantic_asr.adapters import DecodeRequest
from semantic_asr.cache import EvidenceCache
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.longform import SemanticASRTranscriber, plan_windows
from semantic_asr.outputs import write_outputs
from semantic_asr.teachers import DelayedTeacherPolicy, TeacherResult


class FakeAdapter:
    name = "fake-whisper"
    model_name = "fixture"

    def __init__(self) -> None:
        self.requests: list[DecodeRequest] = []

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.requests.append(request)
        if request.beam_size > 5:
            return [
                CandidateEvidence(
                    "relisten",
                    "昨日学校を行きました",
                    acoustic=0.90,
                    mora=0.88,
                    preservation=0.95,
                    rank=1,
                    hypothesis_count=2,
                    avg_logprob=-0.08,
                    source=self.name,
                ),
                CandidateEvidence(
                    "relisten-clean",
                    "昨日学校に行きました",
                    acoustic=0.61,
                    mora=0.55,
                    preservation=0.40,
                    rank=2,
                    hypothesis_count=2,
                    avg_logprob=-0.52,
                    source=self.name,
                ),
            ]
        return [
            CandidateEvidence(
                "spoken",
                "昨日学校を行きました",
                acoustic=0.53,
                mora=0.52,
                preservation=0.90,
                rank=1,
                hypothesis_count=2,
                avg_logprob=-0.30,
                source=self.name,
            ),
            CandidateEvidence(
                "clean",
                "昨日学校に行きました",
                acoustic=0.51,
                mora=0.50,
                preservation=0.42,
                rank=2,
                hypothesis_count=2,
                avg_logprob=-0.32,
                source=self.name,
            ),
        ]


class FakeTeacher:
    model = "fake-qwen"

    def probabilities(self, candidates, **kwargs):
        clean = next(candidate for candidate in candidates if "学校に" in candidate.text)
        spoken = next(candidate for candidate in candidates if "学校を" in candidate.text)
        return TeacherResult(
            probabilities={clean.candidate_id: 0.8, spoken.candidate_id: 0.2},
            model=self.model,
            endpoint_origin="fake",
            protocol="fixture",
            entropy=0.72,
            abstained=False,
        )


def test_window_planner() -> None:
    assert [
        (row.start_ms, row.end_ms)
        for row in plan_windows(60_000, window_ms=28_000, overlap_ms=1_000)
    ] == [
        (0, 28_000),
        (27_000, 55_000),
        (54_000, 60_000),
    ]


def test_longform_selective_decode_and_cache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "fixture.wav"
        audio.write_bytes(b"fake-adapter-does-not-read-audio")
        adapter = FakeAdapter()
        with EvidenceCache(root / "cache.sqlite3") as cache:
            transcriber = SemanticASRTranscriber(
                adapter,
                cache=cache,
                window_ms=20_000,
                overlap_ms=1_000,
            )
            first = transcriber.transcribe(audio, duration_ms=30_000)
            call_count = len(adapter.requests)
            assert call_count > 2
            second = transcriber.transcribe(audio, duration_ms=30_000)
            assert len(adapter.requests) == call_count
            assert second.diagnostics["cacheHitCount"] > 0
            assert first.observed_text == second.observed_text
            assert "学校を" in first.observed_text


def test_teacher_changes_only_normalized_layer() -> None:
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "fixture.wav"
        audio.write_bytes(b"fixture")
        result = SemanticASRTranscriber(
            FakeAdapter(),
            teacher=FakeTeacher(),
            teacher_policy=DelayedTeacherPolicy(
                minimum_entropy=0.0,
                maximum_posterior_margin=1.0,
                minimum_disagreement=0.0,
            ),
        ).transcribe(audio, duration_ms=1_000)
        assert "学校を" in result.observed_text
        assert "学校に" in result.normalized_text
        assert result.segments[0].normalized.mode == "rank-only"
        assert "particle-or-functional" in result.segments[0].normalized.semantic_change_warnings
        result.segments[0].observed.verify()


def test_outputs_redact_absolute_source_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "fixture.wav"
        audio.write_bytes(b"fixture")
        result = SemanticASRTranscriber(FakeAdapter()).transcribe(audio, duration_ms=1_000)
        outputs = write_outputs(result, root / "out")
        payload_text = Path(outputs["json"]).read_text(encoding="utf-8")
        payload = json.loads(payload_text)
        assert payload["source_name"] == "fixture.wav"
        assert str(root) not in payload_text
        assert payload["contract"]["observedImmutable"] is True
        assert Path(outputs["srt"]).read_text(encoding="utf-8").startswith("1\n")
        assert Path(outputs["vtt"]).read_text(encoding="utf-8").startswith("WEBVTT")
