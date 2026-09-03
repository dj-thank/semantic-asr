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
    assert set(payload["outputs"]) == {"json", "observed"}
    assert (tmp_path / "out").exists()


def test_root_cli_routes_run_command() -> None:
    from semantic_asr.cli_root import main

    with pytest.raises(SystemExit) as info:
        main(["run", "--help"])
    assert info.value.code == 0
