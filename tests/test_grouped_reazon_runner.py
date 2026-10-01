"""WAV runner wiring with synthetic audio and model-free backend substitutes."""

import hashlib
import json
import subprocess
import sys
import threading
import wave
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import realtime_reazon as runner
from semantic_asr.realtime_reazon import RealtimeReazonSession
from semantic_asr.realtime_refine_runtime import GroupedRealtimeReazon


def arguments(tmp_path, **overrides):
    vad = tmp_path / "vad.onnx"
    vad.write_bytes(b"synthetic-vad-identity")
    source = tmp_path / "test.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(b"\x01\x00" * (512 * 52))
    fields = dict(
        audio=str(source), model_dir=str(tmp_path), model_sha256="a" * 64,
        vad_model=str(vad), vad_model_sha256=hashlib.sha256(vad.read_bytes()).hexdigest(),
        events_jsonl=str(tmp_path / "events.jsonl"), threads=1, vad_threshold=0.5,
        vad_min_silence=0.35, max_speech=12.0, partial_interval_ms=500,
        preroll_ms=800, refine_idle_ms=2000, max_history_utterances=16,
        realtime=False, allow_local_research=True, grouped_refine=True,
        max_pending_groups=2, refine_shutdown_timeout=5.0,
    )
    return Namespace(**(fields | overrides))


class Samples:
    def __init__(self, raw):
        self.raw = raw
    def astype(self, _):
        return self
    def __truediv__(self, _):
        return self
    def __len__(self):
        return len(self.raw) // 2


class Vad:
    def __init__(self):
        self.index = 0
    def accept_waveform(self, _):
        self.index += 1
    def is_speech_detected(self):
        return self.index <= 20 or 25 <= self.index <= 40
    def empty(self):
        return True
    def flush(self):
        pass


def install_backends(monkeypatch, *, failure=None):
    creations, calls = [], []
    owner = threading.get_ident()
    class Adapter:
        model_artifact_sha256 = "a" * 64
        runtime_revision = "test-runtime"
        def __init__(self, model_dir, *, artifact_sha256, cpu_threads):
            self.owner = threading.get_ident()
            creations.append((self, self.owner, artifact_sha256, cpu_threads))
        def decode(self, request):
            assert self.owner == threading.get_ident()
            with wave.open(request.audio_path, "rb") as handle:
                calls.append((self, handle.readframes(handle.getnframes()), request))
            if self.owner != owner and failure is not None:
                raise failure
            return [SimpleNamespace(text="一次認識" if self.owner == owner else "長い文脈の候補")]
    monkeypatch.setattr(runner, "ReazonSpeechK2Adapter", Adapter)
    monkeypatch.setattr(runner, "build_vad", lambda *args, **kwargs: Vad())
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(
        frombuffer=lambda raw, dtype: Samples(raw), float32="float32"))
    return creations, calls


def test_wav_runner_emits_fast_finals_then_bound_group_candidate(tmp_path, monkeypatch):
    creations, calls = install_backends(monkeypatch)
    args = arguments(tmp_path)
    assert runner.run(args) == 0
    records = [json.loads(line) for line in Path(args.events_jsonl).read_text().splitlines()]
    finals = [r for r in records if r.get("kind") == "final"]
    candidates = [r for r in records if r.get("kind") == "group_refine"]
    assert len(finals) == 2 and len(candidates) == 1
    group = candidates[0]
    assert [p["finalDigest"] for p in group["request"]["parentFinals"]] == [
        f["evidenceDigest"] for f in finals]
    assert all(records.index(f) < records.index(group) for f in finals)
    assert group["candidateText"] == "長い文脈の候補"
    assert group["automaticallyApplied"] is False
    assert group["independentEvidence"] is False
    assert len(creations) == 2 and creations[0][1] != creations[1][1]
    assert all(c[2:] == (args.model_sha256, 1) for c in creations)
    refined = [c for c in calls if c[0] is creations[1][0]]
    assert len(refined) == 1
    with wave.open(args.audio, "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    start, end = group["request"]["startSample"], group["request"]["endSample"]
    assert refined[0][1] == raw[start * 2:end * 2]
    decode_request = refined[0][2]
    assert decode_request.language == "ja" and decode_request.hypotheses == 1
    assert decode_request.initial_prompt is None and decode_request.hotwords == ()
    assert records[-1]["status"] == "completed"
    assert records[-1]["groupedRefine"]["workerStillRunning"] is False


def test_default_runner_does_not_construct_second_model_or_change_output_schema(tmp_path, monkeypatch):
    creations, _ = install_backends(monkeypatch)
    args = arguments(tmp_path, grouped_refine=False)
    assert runner.run(args) == 0
    records = [json.loads(line) for line in Path(args.events_jsonl).read_text().splitlines()]
    assert len(creations) == 1
    assert all(r.get("kind") != "group_refine" for r in records)
    assert "groupedRefine" not in records[-1]
    assert records[-1]["status"] == "completed"


def test_failed_second_pass_reports_partial_nonzero_without_losing_first_pass(tmp_path, monkeypatch):
    install_backends(monkeypatch, failure=RuntimeError("PRIVATE_AUDIO_PATH"))
    args = arguments(tmp_path)
    assert runner.run(args) == 2
    text = Path(args.events_jsonl).read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert len([r for r in records if r.get("kind") == "final"]) == 2
    assert records[-1]["status"] == "partial"
    assert records[-1]["groupedRefine"]["outcomes"] == {"error": 1}
    assert "PRIVATE_AUDIO_PATH" not in text


@pytest.mark.parametrize("field,value", [
    ("refine_shutdown_timeout", -1), ("refine_shutdown_timeout", float("nan")),
    ("max_pending_groups", 0), ("max_pending_groups", True),
    ("max_speech", 25.0), ("max_speech", float("inf")),
    ("refine_idle_ms", 0),
])
def test_invalid_group_controls_fail_before_loading_models(tmp_path, monkeypatch, field, value):
    creations, _ = install_backends(monkeypatch)
    args = arguments(tmp_path, **{field: value})
    with pytest.raises((ValueError, TypeError)):
        runner.run(args)
    assert creations == []


def test_session_factory_is_opt_in_and_second_adapter_is_lazy(tmp_path, monkeypatch):
    creations, _ = install_backends(monkeypatch)
    args = arguments(tmp_path)
    adapter = runner.ReazonSpeechK2Adapter(tmp_path, artifact_sha256="a" * 64, cpu_threads=1)
    rt = runner.make_reazon_session(adapter, args)
    assert isinstance(rt, GroupedRealtimeReazon) and len(creations) == 1
    assert rt.close(timeout_seconds=0) == ()
    args.grouped_refine = False
    assert isinstance(runner.make_reazon_session(adapter, args), RealtimeReazonSession)
    assert len(creations) == 1


def test_import_and_help_have_no_optional_backend_imports_or_model_work():
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
sys.path.insert(0, "src")
from scripts import realtime_reazon
assert not any(x in sys.modules for x in ('numpy', 'sherpa_onnx', 'torch'))
sys.argv = ['realtime_reazon.py', '--help']
realtime_reazon.main()
"""
    result = subprocess.run([sys.executable, "-S", "-c", program], cwd=root,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--grouped-refine" in result.stdout
    assert "--max-pending-groups" in result.stdout
