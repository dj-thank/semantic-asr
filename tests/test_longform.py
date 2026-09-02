from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.advanced_adapters import AdaptiveRerankingAdapter
from semantic_asr.cache import EvidenceCache
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.longform import SemanticASRTranscriber, plan_windows
from semantic_asr.outputs import write_outputs
from semantic_asr.teachers import DelayedTeacherPolicy, TeacherResult


class FakeAdapter:
    name = "fake-whisper"
    model_name = "fixture"
    # Explicitly marks this in-memory fixture as safe for the legacy cache identity.
    allow_legacy_cache_identity = True

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


class RevisionedFakeAdapter(FakeAdapter):
    def __init__(self, revision: str, *, length_penalty: float = 1.0) -> None:
        super().__init__()
        self.model_revision = revision
        self.runtime_revision = "runtime-r1"
        self.length_penalty = length_penalty


class FakeTeacher:
    model = "fake-qwen"
    # Explicitly marks this in-memory fixture as safe for the legacy cache identity.
    allow_legacy_cache_identity = True

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


def test_unbound_runtime_adapter_cannot_use_legacy_cache_identity() -> None:
    class UnboundAdapter:
        name = "unbound"
        model_name = "floating-model"

        def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
            raise AssertionError("cache identity must fail before inference")

    transcriber = SemanticASRTranscriber(UnboundAdapter())
    request = DecodeRequest(audio_path="fixture.wav", start_ms=0, end_ms=1_000)
    with pytest.raises(ValueError, match="model identity"):
        transcriber._cache_key(
            namespace="base-window",
            adapter=transcriber.base_adapter,
            request=request,
            audio_sha256="a" * 64,
            context="",
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


def test_longform_cache_separates_model_revisions_and_decode_settings() -> None:
    request = DecodeRequest(
        audio_path="fixture.wav",
        language="ja",
        beam_size=5,
        hypotheses=5,
        start_ms=0,
        end_ms=1_000,
    )
    with tempfile.TemporaryDirectory() as directory:
        cache_path = Path(directory) / "cache.sqlite3"
        with EvidenceCache(cache_path) as cache:
            first = SemanticASRTranscriber(
                RevisionedFakeAdapter("r1", length_penalty=1.0),
                cache=cache,
            )
            second = SemanticASRTranscriber(
                RevisionedFakeAdapter("r2", length_penalty=1.1),
                cache=cache,
            )
            first_key = first._cache_key(
                namespace="base-window",
                adapter=first.base_adapter,
                request=request,
                audio_sha256="a" * 64,
                context="",
            )
            second_key = second._cache_key(
                namespace="base-window",
                adapter=second.base_adapter,
                request=request,
                audio_sha256="a" * 64,
                context="",
            )
            assert first_key.model_revision == "r1"
            assert second_key.model_revision == "r2"
            assert first_key.runtime_revision == second_key.runtime_revision == "runtime-r1"
            assert first_key.decode_config_sha256 != second_key.decode_config_sha256
            assert first_key.score_domain != second_key.score_domain

            _, first_hit = first._decode(
                first.base_adapter,
                request,
                namespace="base-window",
                audio_sha256="a" * 64,
                context="",
            )
            _, second_hit = second._decode(
                second.base_adapter,
                request,
                namespace="base-window",
                audio_sha256="a" * 64,
                context="",
            )
            assert first_hit is False
            assert second_hit is False
            assert cache.count("base-window") == 2


def test_longform_cache_binds_ranker_revision_artifact_and_config() -> None:
    class Ranker:
        name = "ranker"
        model_name = "ranker-model"
        runtime_revision = "runtime-r1"

        def __init__(self, revision: str, config_digest: str) -> None:
            self.model_revision = revision
            self.model_artifact_sha256 = None
            self.config_digest = config_digest

        def score(self, candidates, **kwargs):
            return {candidate.candidate_id: 0.0 for candidate in candidates}

    request = DecodeRequest(
        audio_path="fixture.wav",
        language="ja",
        beam_size=5,
        hypotheses=5,
        start_ms=0,
        end_ms=1_000,
    )
    first = SemanticASRTranscriber(
        AdaptiveRerankingAdapter(
            FakeAdapter(),
            Ranker("r1", "a" * 64),
            maximum_hypotheses=2,
        )
    )
    second = SemanticASRTranscriber(
        AdaptiveRerankingAdapter(
            FakeAdapter(),
            Ranker("r2", "b" * 64),
            maximum_hypotheses=2,
        )
    )
    first_key = first._cache_key(
        namespace="base-window",
        adapter=first.base_adapter,
        request=request,
        audio_sha256="a" * 64,
        context="",
    )
    second_key = second._cache_key(
        namespace="base-window",
        adapter=second.base_adapter,
        request=request,
        audio_sha256="a" * 64,
        context="",
    )
    assert first_key.decode_config_sha256 != second_key.decode_config_sha256
    assert first_key.score_domain != second_key.score_domain


def test_legacy_adapter_cache_key_remains_deterministic() -> None:
    request = DecodeRequest(audio_path="fixture.wav", start_ms=0, end_ms=1_000)
    transcriber = SemanticASRTranscriber(FakeAdapter())
    first = transcriber._cache_key(
        namespace="base-window",
        adapter=transcriber.base_adapter,
        request=request,
        audio_sha256="a" * 64,
        context="",
    )
    second = transcriber._cache_key(
        namespace="base-window",
        adapter=transcriber.base_adapter,
        request=request,
        audio_sha256="a" * 64,
        context="",
    )
    assert first.model_revision is None
    assert first.runtime_revision is None
    assert first.digest == second.digest


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
