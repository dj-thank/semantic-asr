"""Opt-in realtime-style Reazon runner with pinned local Silero VAD assets.

This is an engineering/evaluation entry point, not a promoted production profile.
It reuses Semantic ASR's existing Reazon adapter and the dependency-free realtime
contract.  The VAD shape follows Hayamimi's MIT-licensed Japanese live path:
Silero via sherpa-onnx, 16 kHz, 512-sample windows, threshold 0.5, 0.25 s minimum
speech, 0.35 s endpointing and 12 s maximum speech.

The first implementation intentionally feeds the existing file-based Reazon adapter
through short temporary WAVs.  That preserves the already-tested adapter contract at
the cost of extra I/O; issue #64/#39 can optimize the measured bottleneck later
without changing evidence semantics.
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from semantic_asr.adapters import DecodeRequest
from semantic_asr.realtime_reazon import (
    RealtimeDecodeInput,
    RealtimeReazonConfig,
    RealtimeReazonSession,
)
from semantic_asr.reazon_adapter import ReazonSpeechK2Adapter
from semantic_asr.revisions import verify_artifact_sha256

SAMPLE_RATE = 16_000
WINDOW_SIZE = 512
VAD_MIN_SPEECH_S = 0.25
VAD_BUFFER_S = 30.0
VAD_NUM_THREADS = 1


def _require_probability(value: float, *, name: str) -> float:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be > 0 and <= 1")
    return value


def build_vad(
    sherpa_onnx,
    vad_model: Path,
    *,
    threshold: float = 0.5,
    min_silence_seconds: float = 0.35,
    max_speech_seconds: float = 12.0,
):
    """Construct the bounded Silero VAD used by this runner."""

    _require_probability(threshold, name="threshold")
    if min_silence_seconds <= 0:
        raise ValueError("min_silence_seconds must be positive")
    if not 0 < max_speech_seconds <= 30:
        raise ValueError("max_speech_seconds must be in (0, 30]")
    if not vad_model.is_file() or not vad_model.stat().st_size:
        raise ValueError("VAD model must be an existing non-empty file")
    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(vad_model),
            threshold=threshold,
            min_silence_duration=min_silence_seconds,
            min_speech_duration=VAD_MIN_SPEECH_S,
            window_size=WINDOW_SIZE,
            max_speech_duration=max_speech_seconds,
        ),
        sample_rate=SAMPLE_RATE,
        num_threads=VAD_NUM_THREADS,
    )
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=VAD_BUFFER_S)


def _drain_vad_queue(vad) -> int:
    """Discard completed VAD-owned segments after our PCM session has consumed them.

    sherpa-onnx retains completed segments until callers pop them.  Semantic ASR owns
    its own exact PCM evidence buffer, so keeping the duplicate VAD segments only
    grows queued state during long sessions.  Hayamimi drains the same queue after
    every input step; this helper mirrors that lifecycle without treating VAD-owned
    audio as transcript evidence.
    """

    drained = 0
    while not vad.empty():
        vad.pop()
        drained += 1
    return drained


def validate_wave(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1:
            raise ValueError("realtime Reazon input must be mono WAV")
        if handle.getsampwidth() != 2:
            raise ValueError("realtime Reazon input must be PCM16 WAV")
        if handle.getframerate() != SAMPLE_RATE:
            raise ValueError("realtime Reazon input must be 16 kHz WAV")
        if handle.getcomptype() != "NONE":
            raise ValueError("compressed WAV is unsupported")
        return handle.getnframes(), handle.getframerate()


def make_reazon_decoder(adapter: ReazonSpeechK2Adapter):
    """Bridge exact session PCM into the existing, tested file-based adapter."""

    def decode(request: RealtimeDecodeInput) -> str:
        with TemporaryDirectory(prefix="semantic-asr-realtime-") as folder:
            path = Path(folder) / "utterance.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setparams((1, 2, request.sample_rate, 0, "NONE", "not compressed"))
                handle.writeframes(request.pcm16le)
            candidates = adapter.decode(
                DecodeRequest(
                    str(path),
                    language="ja",
                    beam_size=4,
                    hypotheses=1,
                    return_timestamps=False,
                )
            )
        return candidates[0].text if candidates else ""

    return decode


def _emit(events, *, output) -> None:
    for event in events:
        line = json.dumps(event.as_dict(), ensure_ascii=False, allow_nan=False)
        print(line)
        if output is not None:
            output.write(line + "\n")
            output.flush()


def run(args: argparse.Namespace) -> int:
    if not args.allow_local_research:
        raise ValueError("realtime execution requires --allow-local-research")
    source = Path(args.audio).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    vad_model = Path(args.vad_model).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"audio does not exist: {source}")
    total_frames, _ = validate_wave(source)
    verify_artifact_sha256(vad_model, args.vad_model_sha256, identifier="VAD model")

    try:
        import numpy as np
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError("install semantic-asr[sherpa] for realtime Reazon execution") from exc

    adapter = ReazonSpeechK2Adapter(
        model_dir,
        artifact_sha256=args.model_sha256,
        cpu_threads=args.threads,
    )
    vad = build_vad(
        sherpa_onnx,
        vad_model,
        threshold=args.vad_threshold,
        min_silence_seconds=args.vad_min_silence,
        max_speech_seconds=args.max_speech,
    )
    decoder = make_reazon_decoder(adapter)
    # Silero already owns endpointing.  Once it transitions to non-speech, one
    # 512-sample chunk is enough for the evidence session to close its utterance;
    # adding another 350 ms here would double-count endpointing latency.
    session = RealtimeReazonSession(
        decoder,
        partial_decoder=decoder,
        config=RealtimeReazonConfig(
            partial_interval_ms=args.partial_interval_ms,
            min_silence_ms=32,
            max_speech_ms=round(args.max_speech * 1000),
            preroll_ms=args.preroll_ms,
            refine_idle_ms=args.refine_idle_ms,
            max_chunk_ms=32,
            max_history_utterances=args.max_history_utterances,
        ),
    )

    output_handle = None
    if args.events_jsonl is not None:
        output_path = Path(args.events_jsonl).expanduser()
        if output_path.exists():
            raise ValueError("events output already exists; choose a new path")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("x", encoding="utf-8")

    started = time.perf_counter()
    accepted_samples = 0
    drained_vad_segments = 0
    try:
        with wave.open(str(source), "rb") as handle:
            while True:
                raw = handle.readframes(WINDOW_SIZE)
                if not raw:
                    break
                if len(raw) % 2:
                    raise ValueError("WAV yielded a partial int16 sample")
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                vad.accept_waveform(samples)
                speech = bool(vad.is_speech_detected())
                _emit(session.feed_pcm16(raw, speech=speech), output=output_handle)
                drained_vad_segments += _drain_vad_queue(vad)
                accepted_samples += len(samples)
                if args.realtime:
                    time.sleep(len(samples) / SAMPLE_RATE)
        vad.flush()
        drained_vad_segments += _drain_vad_queue(vad)
        _emit(session.flush(), output=output_handle)
        elapsed = time.perf_counter() - started
        summary = {
            "schema": "semantic-asr-realtime-run-summary-v1",
            "status": "completed",
            "audio": source.name,
            "inputSamples": total_frames,
            "acceptedSamples": accepted_samples,
            "durationSeconds": total_frames / SAMPLE_RATE,
            "wallSeconds": elapsed,
            "rtf": elapsed / (total_frames / SAMPLE_RATE) if total_frames else None,
            "modelArtifactSha256": adapter.model_artifact_sha256,
            "vadArtifactSha256": args.vad_model_sha256.lower(),
            "drainedVadSegments": drained_vad_segments,
            "vad": {
                "threshold": args.vad_threshold,
                "minSilenceSeconds": args.vad_min_silence,
                "minSpeechSeconds": VAD_MIN_SPEECH_S,
                "maxSpeechSeconds": args.max_speech,
                "windowSamples": WINDOW_SIZE,
            },
            "note": (
                "Engineering measurement only: temporary-WAV adapter bridge is intentionally "
                "not a latency-optimized production path."
            ),
        }
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        if output_handle is not None:
            output_handle.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        if output_handle is not None:
            output_handle.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--vad-model", required=True)
    parser.add_argument("--vad-model-sha256", required=True)
    parser.add_argument("--events-jsonl")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence", type=float, default=0.35)
    parser.add_argument("--max-speech", type=float, default=12.0)
    parser.add_argument("--partial-interval-ms", type=int, default=500)
    parser.add_argument("--preroll-ms", type=int, default=800)
    parser.add_argument("--refine-idle-ms", type=int, default=2000)
    parser.add_argument("--max-history-utterances", type=int, default=16)
    parser.add_argument("--realtime", action="store_true", help="sleep between chunks")
    parser.add_argument("--allow-local-research", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
