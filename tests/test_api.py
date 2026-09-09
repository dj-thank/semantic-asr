from __future__ import annotations

import json
import tempfile
import wave
from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.api import (
    PROFILES,
    RuntimeProfile,
    runtime_profile,
    transcribe,
    transcribe_segments,
)
from semantic_asr.context_catalog import ContextCatalog, ContextEntry
from semantic_asr.contracts import CandidateEvidence


class FakeAdapter:
    name = "fake-whisper"
    model_name = "fixture"
    device = "cpu"
    compute_type = "int8"
    allow_legacy_cache_identity = True

    def __init__(self) -> None:
        self.requests: list[DecodeRequest] = []

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.requests.append(request)
        return [
            CandidateEvidence(
                "spoken",
                "昨日学校を行きました",
                acoustic=0.9,
                mora=0.9,
                preservation=0.95,
                rank=1,
                hypothesis_count=2,
                avg_logprob=-0.05,
                source=self.name,
            ),
            CandidateEvidence(
                "clean",
                "昨日学校に行きました",
                acoustic=0.4,
                mora=0.4,
                preservation=0.3,
                rank=2,
                hypothesis_count=2,
                avg_logprob=-0.6,
                source=self.name,
            ),
        ]


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(b"\x00\x00" * int(16_000 * seconds))


def test_profiles_are_frozen_named_and_digestible() -> None:
    profile = runtime_profile("cpu-ja-v1")
    assert profile.model == "large-v3-turbo"
    assert profile.device == "cpu"
    assert profile.loop_guard is True
    assert len(profile.digest) == 64
    assert runtime_profile(profile) is profile
    assert {name for name in PROFILES} >= {
        "cpu-ja-v1",
        "cpu-ja-quality-v1",
        "reazon-ja-v1",
        "reazon-ja-research-v1",
        "gpu-ja-v1",
    }
    with pytest.raises(ValueError):
        runtime_profile("nope")
    with pytest.raises(ValueError):
        RuntimeProfile(name="bad", description="", beam_size=2, hypotheses=5)
    with pytest.raises(ValueError):
        RuntimeProfile(name="bad", description="", window_ms=40_000)


def test_reazon_profile_uses_native_top_one_budget_without_confidence() -> None:
    profile = runtime_profile("reazon-ja-v1")
    assert profile.model == "reazon-research/reazonspeech-k2-v2"
    assert profile.beam_size == 4
    assert profile.hypotheses == 1
    assert profile.confidence_calibration is None


def test_reazon_without_local_adapter_never_loads_whisper(monkeypatch):
    from semantic_asr import advanced_adapters
    from semantic_asr.api import build_adapter

    def wrong_backend(*args, **kwargs):
        pytest.fail("Reazon profile must not load or download a Whisper backend")

    monkeypatch.setattr(advanced_adapters, "PathPreservingFasterWhisperAdapter", wrong_backend)
    with pytest.raises(ValueError, match="verified local artifact"):
        build_adapter(runtime_profile("reazon-ja-v1"))


def test_reazon_profile_routes_native_decode_budget(tmp_path: Path) -> None:
    audio = tmp_path / "reazon-profile.wav"
    _write_wav(audio, 1.0)
    adapter = FakeAdapter()
    result = transcribe(audio, profile="reazon-ja-v1", adapter=adapter)
    assert adapter.requests[0].beam_size == 4
    assert adapter.requests[0].hypotheses == 1
    assert result.profile.name == "reazon-ja-v1"
    assert result.provenance["confidenceCalibrationApplied"] is False


def test_reazon_research_profile_has_bounded_evidence_budget() -> None:
    profile = runtime_profile("reazon-ja-research-v1")
    assert profile.effort == "research"
    assert profile.beam_size == 4
    assert profile.hypotheses == 1


def test_second_ear_is_bound_and_provenance_is_explicit(tmp_path: Path) -> None:
    audio = tmp_path / "second-ear.wav"
    _write_wav(audio, 1.0)
    primary = FakeAdapter()
    second = FakeAdapter()
    result = transcribe(
        audio,
        profile="cpu-ja-quality-v1",
        adapter=primary,
        second_ear=second,
    )
    assert result.provenance["secondEar"]["adapter"] == "fake-whisper"
    assert result.provenance["secondEar"]["model"] == "fixture"
    assert result.provenance["secondEar"]["modelRevision"] is None


def test_second_ear_cannot_be_silently_ignored_with_warm_transcriber(tmp_path: Path) -> None:
    audio = tmp_path / "warm-second-ear.wav"
    _write_wav(audio, 1.0)
    with pytest.raises(ValueError, match="second_ear"):
        transcribe(
            audio,
            profile="cpu-ja-quality-v1",
            transcriber=_warm(FakeAdapter()),
            second_ear=FakeAdapter(),
        )


def test_adapter_capabilities_fail_closed_before_unsupported_hotword_decode(tmp_path: Path) -> None:
    class NoHotwordAdapter(FakeAdapter):
        name = "no-hotword"
        supports_hotwords = False

    audio = tmp_path / "no-hotword.wav"
    _write_wav(audio, 1.0)
    with pytest.raises(ValueError, match="does not support hotwords"):
        transcribe(audio, adapter=NoHotwordAdapter(), hotwords=("固有名詞",))


def test_adapter_capabilities_fail_closed_before_unsupported_prompt_decode(tmp_path: Path) -> None:
    class NoPromptAdapter(FakeAdapter):
        name = "no-prompt"
        supports_initial_prompt = False

    audio = tmp_path / "no-prompt.wav"
    _write_wav(audio, 1.0)
    with pytest.raises(ValueError, match="does not support initial_prompt"):
        transcribe(audio, adapter=NoPromptAdapter(), initial_prompt="文脈")


def test_context_catalog_hotword_fails_closed_for_unsupported_adapter(tmp_path: Path) -> None:
    class NoHotwordAdapter(FakeAdapter):
        name = "no-hotword-catalog"
        supports_hotwords = False

    audio = tmp_path / "catalog-no-hotword.wav"
    _write_wav(audio, 1.0)
    catalog = ContextCatalog(
        name="meeting",
        revision="agenda-v1",
        entries=(ContextEntry("term:semantic-asr", "Semantic ASR", tags=("term",)),),
    )
    with pytest.raises(ValueError, match="does not support hotwords"):
        transcribe(
            audio,
            adapter=NoHotwordAdapter(),
            catalog=catalog,
            context_query="Semantic ASR",
            context_tags=("term",),
        )


def test_transcribe_path_returns_segments_and_provenance(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _write_wav(audio, 3.0)
    messages: list[str] = []
    adapter = FakeAdapter()
    result = transcribe(audio, adapter=adapter, on_progress=messages.append)
    assert result.profile.name == "cpu-ja-v1"
    assert result.observed_text == "昨日学校を行きました"
    assert result.segments and result.segments[0].start_ms == 0
    assert result.segments[0].observed == "昨日学校を行きました"
    assert result.provenance["adapter"] == "fake-whisper"
    assert result.provenance["profileDigest"] == result.profile.digest
    assert messages[0].startswith("loading") and messages[-1] == "done"
    payload = result.as_dict()
    assert json.dumps(payload, ensure_ascii=False)
    assert payload["segments"][0]["status"]
    outputs = result.write(tmp_path / "out")
    assert set(outputs) >= {"json", "observed", "normalized", "srt"}
    assert adapter.requests, "the adapter was used for decoding"


def test_transcribe_accepts_numpy_array_and_cleans_temp_file() -> None:
    np = pytest.importorskip("numpy")
    samples = np.zeros(16_000 * 2, dtype=np.float32)
    result = transcribe(samples, adapter=FakeAdapter())
    assert result.duration_ms >= 1_900
    assert Path(result.source_name).suffix == ".wav"
    leftovers = list(Path(tempfile.gettempdir()).glob("semantic-asr-*.wav"))
    assert not leftovers


def test_transcribe_segments_matches_koemo_contract(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _write_wav(audio, 2.0)
    rows = transcribe_segments(audio, profile="cpu-ja-v1", transcriber=_warm(FakeAdapter()))
    assert rows == [(0.0, 2.0, "昨日学校を行きました")]
    normalized = transcribe_segments(
        audio, profile="cpu-ja-v1", normalized=True, transcriber=_warm(FakeAdapter())
    )
    assert normalized and normalized[0][2]


def _warm(adapter: FakeAdapter):
    from semantic_asr.api import load_transcriber

    return load_transcriber("cpu-ja-v1", adapter=adapter)


def test_run_cli_writes_outputs_with_injected_adapter(tmp_path: Path, capsys) -> None:
    from semantic_asr.run_cli import build_parser, run_transcription

    audio = tmp_path / "clip.wav"
    _write_wav(audio, 1.5)
    args = build_parser().parse_args(
        [str(audio), "--output-dir", str(tmp_path / "out"), "--formats", "json,observed", "--quiet"]
    )
    payload = run_transcription(args, adapter=FakeAdapter())
    assert payload["status"] == "ok"
    assert payload["profile"] == "cpu-ja-v1"
    assert set(payload["outputs"]) == {"json", "observed", "transcript_json"}
    assert (tmp_path / "out").exists()


def test_run_cli_reazon_profile_requires_local_artifact_pair() -> None:
    from semantic_asr.run_cli import build_parser, build_profile_adapters

    args = build_parser().parse_args(["clip.wav", "--profile", "reazon-ja-v1"])
    with pytest.raises(ValueError, match="Reazon profiles require"):
        build_profile_adapters(args)


def test_qwen_profile_requires_explicit_local_artifact_before_model_import():
    from semantic_asr.api import build_adapter
    from semantic_asr.run_cli import build_parser, build_profile_adapters

    with pytest.raises(ValueError, match="verified local artifact"):
        build_adapter(runtime_profile("qwen-ja-cpu-v1"))
    args = build_parser().parse_args(["clip.wav", "--profile", "qwen-ja-cpu-v1"])
    with pytest.raises(ValueError, match="both model"):
        build_profile_adapters(args)
    args = build_parser().parse_args(["clip.wav", "--qwen-model-dir", "local-model"])
    with pytest.raises(ValueError, match="require qwen"):
        build_profile_adapters(args)


def test_qwen_profile_preserves_provisional_unscored_observation(tmp_path):
    from semantic_asr.adapters import CandidateEvidence

    class QwenLike:
        name = "qwen3-asr"
        model_artifact_sha256 = "a" * 64

        def decode(self, request):
            return [
                CandidateEvidence(
                    candidate_id="qwen-0000",
                    text="ええ、今日は今日は晴れです。",
                    source=self.name,
                    metadata={"adapter": self.name, "scoreKind": "unscored-transcript"},
                )
            ]

    audio = tmp_path / "qwen.wav"
    _write_wav(audio, 1.0)
    result = transcribe(audio, profile="qwen-ja-cpu-v1", adapter=QwenLike())
    result.verify()
    assert result.observed_text == "ええ、今日は今日は晴れです。"
    assert all(s.status == "provisional" and s.confidence is None for s in result.segments)


def test_run_cli_rejects_second_ear_on_default_profile() -> None:
    from semantic_asr.run_cli import build_parser, build_profile_adapters

    args = build_parser().parse_args(
        [
            "clip.wav",
            "--reazon-model-dir",
            "model",
            "--reazon-artifact-sha256",
            "a" * 64,
        ]
    )
    with pytest.raises(ValueError, match="require a Reazon profile"):
        build_profile_adapters(args)


def test_root_cli_routes_run_command() -> None:
    from semantic_asr.cli_root import main

    with pytest.raises(SystemExit) as info:
        main(["run", "--help"])
    assert info.value.code == 0


def test_calibrated_confidence_is_monotone_and_optional() -> None:
    from semantic_asr.api import calibrated_confidence

    profile = runtime_profile("cpu-ja-v1")
    low = calibrated_confidence(profile, 0.2)
    high = calibrated_confidence(profile, 0.95)
    assert low is not None and high is not None and 0.0 < low < high < 1.0
    assert calibrated_confidence(profile, None) is None
    assert calibrated_confidence(profile, float("nan")) is None
    disabled = RuntimeProfile(name="x", description="", confidence_calibration=None)
    assert calibrated_confidence(disabled, 0.9) is None


def test_unknown_adapter_does_not_inherit_measured_confidence(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    _write_wav(audio, 2.0)
    result = transcribe(audio, adapter=FakeAdapter())
    assert result.segments[0].confidence is None
    assert result.provenance["confidenceCalibrationApplied"] is False


def test_quality_profile_reaches_the_decode_request(tmp_path: Path) -> None:
    audio = tmp_path / "quality.wav"
    _write_wav(audio, 1.0)
    adapter = FakeAdapter()
    result = transcribe(audio, profile="cpu-ja-quality-v1", adapter=adapter)
    assert adapter.requests[0].beam_size == 12
    assert adapter.requests[0].hypotheses == 12
    assert result.provenance["beamSize"] == 12
    assert result.provenance["hypotheses"] == 12
    assert result.profile.model_revision is not None
    assert result.provenance["modelRevision"] is None
    assert result.provenance["requestedModelRevision"] == result.profile.model_revision


def test_warm_transcriber_profile_mismatch_fails_closed(tmp_path: Path) -> None:
    audio = tmp_path / "mismatch.wav"
    _write_wav(audio, 1.0)
    with pytest.raises(ValueError, match="does not match"):
        transcribe(
            audio,
            profile="cpu-ja-quality-v1",
            transcriber=_warm(FakeAdapter()),
        )


def test_catalog_terms_are_selected_without_leaking_raw_names(tmp_path: Path) -> None:
    audio = tmp_path / "catalog.wav"
    _write_wav(audio, 1.0)
    catalog = ContextCatalog(
        name="meeting",
        revision="agenda-v1",
        entries=(
            ContextEntry(
                "person:moriwaki",
                "森脇翔太",
                aliases=("森脇さん",),
                tags=("person",),
            ),
        ),
    )
    adapter = FakeAdapter()
    result = transcribe(
        audio,
        adapter=adapter,
        catalog=catalog,
        context_query="森脇さんとSemantic ASRを確認",
        context_tags=("person",),
    )
    assert adapter.requests[0].hotwords == ("森脇翔太",)
    receipt = result.provenance["contextCatalog"]
    assert receipt["enabled"] is True
    assert receipt["abstained"] is False
    assert "森脇翔太" not in json.dumps(receipt, ensure_ascii=False)
    assert result.provenance["catalogHotwordCount"] == 1


@pytest.mark.parametrize("shape", [(2, 16_000), (16_000, 2)])
def test_array_audio_accepts_common_channel_orders(shape) -> None:
    np = pytest.importorskip("numpy")
    result = transcribe(np.zeros(shape, dtype=np.float32), adapter=FakeAdapter())
    assert 900 <= result.duration_ms <= 1_100


def test_array_audio_rejects_ambiguous_or_non_finite_shapes() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="channel axis"):
        transcribe(np.zeros((16, 16), dtype=np.float32), adapter=FakeAdapter())
    bad = np.zeros(16_000, dtype=np.float32)
    bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        transcribe(bad, adapter=FakeAdapter())


class InvalidSpanAdapter(FakeAdapter):
    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.requests.append(request)
        return [
            CandidateEvidence(
                "spans",
                "有効な字幕です",
                acoustic=0.9,
                rank=1,
                hypothesis_count=1,
                avg_logprob=-0.05,
                source=self.name,
                metadata={
                    "utteranceSpans": [
                        {"startMs": 800, "endMs": 200, "text": "逆転"},
                        {"startMs": 100, "endMs": 500, "text": "有効"},
                        {"startMs": "bad", "endMs": 700, "text": "不正"},
                    ]
                },
            )
        ]


def test_invalid_timestamp_rows_never_create_negative_srt_ranges(tmp_path: Path) -> None:
    audio = tmp_path / "spans.wav"
    _write_wav(audio, 1.0)
    result = transcribe(audio, adapter=InvalidSpanAdapter())
    assert [utterance.text for utterance in result.utterances] == ["有効な字幕です"]
    assert all(row.end_ms > row.start_ms for row in result.utterances)


def test_run_cli_accepts_frozen_context_catalog(tmp_path: Path) -> None:
    from semantic_asr.run_cli import build_parser, run_transcription

    audio = tmp_path / "clip.wav"
    _write_wav(audio, 1.0)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "name": "meeting",
                "revision": "v1",
                "entries": [
                    {
                        "id": "person:moriwaki",
                        "phrase": "森脇翔太",
                        "aliases": ["森脇さん"],
                        "tags": ["person"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            str(audio),
            "--catalog",
            str(catalog),
            "--context-query",
            "森脇さんとの会議",
            "--context-tag",
            "person",
            "--output-dir",
            str(tmp_path / "out"),
            "--quiet",
        ]
    )
    adapter = FakeAdapter()
    payload = run_transcription(args, adapter=adapter)
    assert payload["status"] == "ok"
    assert adapter.requests[0].hotwords == ("森脇翔太",)


def test_effort_profile_controls_runtime_evidence_budget() -> None:
    from semantic_asr.api import load_transcriber

    light = load_transcriber("cpu-ja-v1", adapter=FakeAdapter())
    assert light.evidence_budget.total_cost_ms == 0
    assert light.evidence_budget.max_actions == 0
    assert light.runtime_profile_digest == runtime_profile("cpu-ja-v1").digest

    quality = load_transcriber("cpu-ja-quality-v1", adapter=FakeAdapter())
    assert quality.evidence_budget.total_cost_ms == 4_000
    assert quality.evidence_budget.max_actions == 4


def test_warm_transcriber_rejects_same_shape_different_profile(tmp_path: Path) -> None:
    from semantic_asr.api import load_transcriber

    audio = tmp_path / "profile-binding.wav"
    _write_wav(audio, 1.0)
    quality = load_transcriber("cpu-ja-quality-v1", adapter=FakeAdapter())
    with pytest.raises(ValueError, match="does not match"):
        transcribe(audio, profile="gpu-ja-v1", transcriber=quality)


def test_unbound_warm_transcriber_fails_closed(tmp_path: Path) -> None:
    from semantic_asr.longform import SemanticASRTranscriber

    audio = tmp_path / "unbound.wav"
    _write_wav(audio, 1.0)
    with pytest.raises(ValueError, match="not bound"):
        transcribe(audio, transcriber=SemanticASRTranscriber(FakeAdapter()))


def test_runtime_profile_rejects_invalid_patience_and_effort_bounds() -> None:
    with pytest.raises(ValueError, match="patience"):
        RuntimeProfile(name="bad", description="", patience=float("nan"))
    with pytest.raises(ValueError, match="at most"):
        RuntimeProfile(
            name="bad",
            description="",
            beam_size=6,
            hypotheses=6,
            effort="ultra-light",
        )


@pytest.mark.parametrize("failure", [RuntimeError, EOFError])
def test_phone_observer_failure_retains_first_pass_and_reports_coverage(tmp_path, failure):
    audio = tmp_path / "phone-failure.wav"
    _write_wav(audio, 1.0)
    baseline = transcribe(audio, adapter=FakeAdapter())

    class FailingObserver:
        def observe(self, path, *, start_ms, end_ms):
            assert Path(path) == audio
            assert (start_ms, end_ms) == (0, 1000)
            raise failure("local phone model unavailable")

    result = transcribe(audio, adapter=FakeAdapter(), phone_observer=FailingObserver())
    result.verify()
    assert result.evidence_sha256 == baseline.evidence_sha256
    assert result.observed_text == baseline.observed_text
    assert result.normalized_text == baseline.normalized_text
    assert "phoneEvidence" not in baseline.diagnostics
    evidence = result.diagnostics["phoneEvidence"]
    assert evidence["completed_windows"] == 0
    assert evidence["total_windows"] == 1
    assert "unavailable" in evidence["windows"][0]["reason"]
    evidence["completed_windows"] = 1
    with pytest.raises(ValueError, match="phone evidence"):
        result.verify()


def test_phone_cli_pair_is_checked_without_loading_models():
    from semantic_asr.run_cli import build_parser, build_phone_observer

    args = build_parser().parse_args(["audio.wav", "--phone-model-dir", "model"])
    with pytest.raises(ValueError, match="supplied together"):
        build_phone_observer(args)


@pytest.mark.parametrize("score_error", [None, FileNotFoundError, EOFError])
def test_successful_phone_observer_survives_api_json_without_changing_transcript(
    tmp_path, score_error
):
    from semantic_asr.audio_phone_runtime import AudioPhoneObservation
    from semantic_asr.phonetic_evidence import PosteriorFrame, PosteriorSequence

    audio = tmp_path / "phone-success.wav"
    _write_wav(audio, 1.0)
    baseline = transcribe(audio, adapter=FakeAdapter())

    class Observer:
        def score_candidates(self, observation, segment):
            if score_error:
                raise score_error("candidate resource unavailable")
            return None

        def observe(self, path, *, start_ms, end_ms):
            assert Path(path) == audio
            assert (start_ms, end_ms) == (0, 1000)
            posterior = PosteriorSequence(
                "phone",
                "PAD",
                ("PAD", "a"),
                tuple(
                    PosteriorFrame.from_mapping(
                        start_ms=i * 20, end_ms=(i + 1) * 20, probabilities={"PAD": 0.1, "a": 0.9}
                    )
                    for i in range(50)
                ),
                "fixture",
                "artifact:" + "a" * 64,
                "labels-v1",
                baseline.source_audio_sha256,
            )
            return AudioPhoneObservation(
                posterior, 0, 16000, 16000, "b" * 64, "c" * 64, "a" * 64, "fixture", 320, 400
            )

    result = transcribe(audio, adapter=FakeAdapter(), phone_observer=Observer())
    payload = result.as_dict()
    assert result.evidence_sha256 == baseline.evidence_sha256
    assert result.observed_text == baseline.observed_text
    assert result.normalized_text == baseline.normalized_text
    phone = payload["diagnostics"]["phoneEvidence"]
    assert phone["completed_windows"] == phone["total_windows"] == 1
    assert phone["windows"][0]["observation"]["phones"][0]["phone"] == "a"
    if score_error:
        assert phone["windows"][0]["candidate_checks"]["execution"] == "unavailable"
