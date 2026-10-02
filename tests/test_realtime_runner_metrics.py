"""Synthetic WAV checks for opt-in runner measurements; no real performance claim."""

import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from test_grouped_reazon_runner import arguments, install_backends

from scripts import realtime_reazon as runner

DURATION_KEYS = {
    "fastFinalDecodeMs",
    "finalEmittingCallMs",
    "refineQueueWaitMs",
    "refineWorkerMs",
    "refineFactoryMs",
    "refineCompletionAfterSubmitMs",
}


def records(args):
    return [json.loads(line) for line in Path(args.events_jsonl).read_text().splitlines()]


def measured_run(tmp_path, *, name, **overrides):
    args = arguments(
        tmp_path,
        events_jsonl=str(tmp_path / f"{name}.jsonl"),
        measure_runtime=True,
        timing_max_samples=10_000,
        **overrides,
    )
    assert runner.run(args) == 0
    return args, records(args)[-1]


def test_both_arms_bind_the_same_stream_and_config_without_private_paths(tmp_path, monkeypatch):
    install_backends(monkeypatch)
    fast_args, fast_summary = measured_run(tmp_path, name="first-pass", grouped_refine=False)
    _, grouped_summary = measured_run(tmp_path, name="grouped", grouped_refine=True)
    fast_metrics, grouped_metrics = (
        fast_summary["runtimeMetrics"],
        grouped_summary["runtimeMetrics"],
    )
    assert fast_metrics["schema"] == grouped_metrics["schema"] == "semantic-asr-realtime-metrics-v1"
    assert fast_metrics["identity"] == grouped_metrics["identity"]
    assert fast_metrics["comparisonIdentitySha256"] == grouped_metrics["comparisonIdentitySha256"]
    identity = grouped_metrics["identity"]
    assert (
        identity["inputArtifactSha256"]
        == hashlib.sha256(Path(fast_args.audio).read_bytes()).hexdigest()
    )
    with wave.open(fast_args.audio, "rb") as handle:
        pcm = handle.readframes(handle.getnframes())
    assert identity["inputPcmSha256"] == hashlib.sha256(pcm).hexdigest()
    expected_trace = hashlib.sha256(b"semantic-asr-vad-trace-v1\0")
    for index in range(1, 53):
        expected_trace.update((512).to_bytes(8, "big"))
        expected_trace.update(bytes([index <= 20 or 25 <= index <= 40]))
    assert identity["vadTraceSha256"] == expected_trace.hexdigest()
    assert identity["modelArtifactSha256"] == fast_args.model_sha256
    assert identity["vadArtifactSha256"] == fast_args.vad_model_sha256
    assert identity["runtimeRevision"] == "test-runtime"
    assert identity["cpuThreads"] == identity["vadThreads"] == 1
    assert identity["realtime"] is False
    assert "firstPassConfig" in identity and "asrDecode" in identity
    assert "groupConfig" not in identity
    assert fast_metrics["groupConfig"] != grouped_metrics["groupConfig"]
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert (
        grouped_metrics["comparisonIdentitySha256"]
        == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    for summary, metrics in (
        (fast_summary, fast_metrics),
        (grouped_summary, grouped_metrics),
    ):
        assert summary["status"] == "completed"
        assert metrics["complete"] is True
        assert metrics["setupMs"] >= 0
        assert metrics["rtfIncludesRealtimeSleep"] is False
        assert metrics["untimedRefineOutcomes"] == 0
        assert set(metrics["durations"]) == DURATION_KEYS
        assert metrics["durations"]["fastFinalDecodeMs"]["count"] == 2
        assert metrics["durations"]["finalEmittingCallMs"]["count"] == 2
        assert all(item["complete"] for item in metrics["durations"].values())
        assert str(tmp_path) not in json.dumps(summary)
    for key in DURATION_KEYS - {"fastFinalDecodeMs", "finalEmittingCallMs"}:
        assert fast_metrics["durations"][key]["count"] == 0
        assert grouped_metrics["durations"][key]["count"] == 1
        assert grouped_metrics["durations"][key]["minMs"] >= 0


@pytest.mark.parametrize("change", [{"realtime": True}, {"partial_interval_ms": 100}])
def test_realtime_mode_or_first_pass_config_changes_comparison_identity(
    tmp_path, monkeypatch, change
):
    install_backends(monkeypatch)
    # This tests the mode identity, not elapsed sleep or real-time performance.
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    _, original = measured_run(tmp_path, name="original", grouped_refine=False)
    _, changed = measured_run(tmp_path, name="changed", grouped_refine=False, **change)
    first, second = original["runtimeMetrics"], changed["runtimeMetrics"]
    assert first["identity"]["inputArtifactSha256"] == second["identity"]["inputArtifactSha256"]
    assert first["identity"]["inputPcmSha256"] == second["identity"]["inputPcmSha256"]
    assert first["identity"]["vadTraceSha256"] == second["identity"]["vadTraceSha256"]
    assert first["comparisonIdentitySha256"] != second["comparisonIdentitySha256"]
    if "realtime" in change:
        assert second["identity"]["realtime"] is True
        assert second["rtfIncludesRealtimeSleep"] is True
    else:
        assert first["identity"]["firstPassConfig"] != second["identity"]["firstPassConfig"]


def test_timing_cap_disables_quantiles_without_marking_recognition_partial(tmp_path, monkeypatch):
    install_backends(monkeypatch)
    args = arguments(tmp_path, measure_runtime=True, timing_max_samples=1)
    assert runner.run(args) == 0
    summary = records(args)[-1]
    assert summary["status"] == "completed"
    metrics = summary["runtimeMetrics"]
    assert metrics["complete"] is False
    for key in ("fastFinalDecodeMs", "finalEmittingCallMs"):
        timing = metrics["durations"][key]
        assert timing["count"] == 2 and timing["retained"] == 1
        assert timing["complete"] is False
        assert timing["p50Ms"] is timing["p95Ms"] is None
        assert timing["meanMs"] >= 0
    assert all(row.get("kind") != "group_refine_warning" for row in records(args))


@pytest.mark.parametrize("value", [0, True, "2", 1.5, float("nan"), 100_001])
def test_invalid_timing_budget_fails_before_models_are_constructed(tmp_path, monkeypatch, value):
    creations, _ = install_backends(monkeypatch)
    args = arguments(tmp_path, measure_runtime=True, timing_max_samples=value)
    with pytest.raises((TypeError, ValueError)):
        runner.run(args)
    assert creations == []


@pytest.mark.parametrize("explicit", [False, True])
def test_disabled_measurement_retains_existing_summary_schema(tmp_path, monkeypatch, explicit):
    install_backends(monkeypatch)
    args = arguments(tmp_path, grouped_refine=False)
    if explicit:
        args.measure_runtime = False
        args.timing_max_samples = 10_000
    assert runner.run(args) == 0
    summary = records(args)[-1]
    assert summary["status"] == "completed"
    assert "runtimeMetrics" not in summary
    assert "comparisonIdentitySha256" not in summary


def test_source_mutation_during_decode_never_emits_a_completed_summary(tmp_path, monkeypatch):
    install_backends(monkeypatch)
    args = arguments(tmp_path, grouped_refine=False, measure_runtime=True)
    adapter_class = runner.ReazonSpeechK2Adapter
    original_decode = adapter_class.decode
    changed = False

    def decode_and_mutate(self, request):
        nonlocal changed
        result = original_decode(self, request)
        if not changed:
            changed = True
            # Trailing bytes change the source identity without changing its PCM.
            with Path(args.audio).open("ab") as source:
                source.write(b"source-changed-during-inference")
        return result

    monkeypatch.setattr(adapter_class, "decode", decode_and_mutate)
    with pytest.raises(ValueError):
        runner.run(args)
    assert changed
    assert all(row.get("schema") != "semantic-asr-realtime-run-summary-v1" for row in records(args))


def test_identity_includes_audio_preprocessing_and_percentile_implementation(tmp_path, monkeypatch):
    install_backends(monkeypatch)
    _, summary = measured_run(tmp_path, name="implementation")
    implementation = summary["runtimeMetrics"]["identity"]["implementationFiles"]
    root = Path(runner.__file__).resolve().parents[1]
    for name in ("audio", "benchmark"):
        path = root / "src" / "semantic_asr" / f"{name}.py"
        assert (
            implementation[f"semantic_asr/{name}.py"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_changed_implementation_cannot_publish_measured_run_summary(tmp_path, monkeypatch):
    install_backends(monkeypatch)
    args = arguments(tmp_path, grouped_refine=False, measure_runtime=True)
    original_identity = runner._implementation_identity
    checks = 0

    def changed_identity():
        nonlocal checks
        checks += 1
        identity = original_identity()
        if checks > 1:
            identity["semantic_asr/audio.py"] = "b" * 64
        return identity

    monkeypatch.setattr(runner, "_implementation_identity", changed_identity)
    with pytest.raises(ValueError, match="implementation changed"):
        runner.run(args)
    assert checks == 2
    assert all(row.get("schema") != "semantic-asr-realtime-run-summary-v1" for row in records(args))


def test_early_readframes_eof_never_emits_a_completed_summary(tmp_path, monkeypatch):
    install_backends(monkeypatch)
    args = arguments(tmp_path, grouped_refine=False, measure_runtime=True)
    source = Path(args.audio).resolve()
    original_open = wave.open

    class TruncatedReader:
        def __init__(self, handle):
            self.handle = handle
            self.reads = 0

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def readframes(self, count):
            self.reads += 1
            return self.handle.readframes(count) if self.reads == 1 else b""

    def open_truncated(file, mode=None):
        handle = original_open(file, mode)
        if mode == "rb" and Path(file).resolve() == source:
            return TruncatedReader(handle)
        return handle

    monkeypatch.setattr(runner.wave, "open", open_truncated)
    with pytest.raises(ValueError):
        runner.run(args)
    assert all(row.get("schema") != "semantic-asr-realtime-run-summary-v1" for row in records(args))


def test_metrics_help_flags_do_not_require_optional_backends():
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
sys.path.insert(0, 'src')
from scripts import realtime_reazon
assert not any(name in sys.modules for name in ('numpy', 'sherpa_onnx', 'torch'))
sys.argv = ['realtime_reazon.py', '--help']
realtime_reazon.main()
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", program], cwd=root, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    assert "--measure-runtime" in result.stdout
    assert "--timing-max-samples" in result.stdout
