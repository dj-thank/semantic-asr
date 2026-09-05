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
    assert {name for name in PROFILES} >= {"cpu-ja-v1", "cpu-ja-quality-v1", "gpu-ja-v1"}
    with pytest.raises(ValueError):
        runtime_profile("nope")
    with pytest.raises(ValueError):
        RuntimeProfile(name="bad", description="", beam_size=2, hypotheses=5)
    with pytest.raises(ValueError):
        RuntimeProfile(name="bad", description="", window_ms=40_000)


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
