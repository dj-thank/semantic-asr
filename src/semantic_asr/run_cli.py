"""``semantic-asr run``: the one-call facade as a CLI vertical slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import ASRAdapter
from .api import PROFILES, transcribe
from .parakeet_adapter import ParakeetJapaneseCtcAdapter
from .reazon_adapter import ReazonSpeechK2Adapter

RUN_COMMANDS = {"run"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-asr run")
    parser.add_argument("audio", help="audio file (any format ffmpeg/PyAV can decode)")
    parser.add_argument("--profile", default="cpu-ja-v1", choices=sorted(PROFILES))
    parser.add_argument("--whisper-model-dir", default=None)
    parser.add_argument("--whisper-artifact-sha256", default=None)
    parser.add_argument("--qwen-model-dir", default=None)
    parser.add_argument("--qwen-artifact-sha256", default=None)
    parser.add_argument("--phone-model-dir", default=None, help="optional local HuBERT phone model")
    parser.add_argument("--phone-artifact-sha256", default=None)
    parser.add_argument(
        "--phone-check-candidates",
        action="store_true",
        help="check whole-window candidate readings against audio phones",
    )
    parser.add_argument("--reazon-model-dir", default=None)
    parser.add_argument("--reazon-artifact-sha256", default=None)
    parser.add_argument("--second-ear-parakeet-model-dir", default=None)
    parser.add_argument("--second-ear-parakeet-artifact-sha256", default=None)
    parser.add_argument("--output-dir", default="transcripts")
    parser.add_argument("--language", default=None, help="override the profile language")
    parser.add_argument("--hotwords", default="", help="comma or 、 separated prompt bias terms")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--catalog",
        default=None,
        help="frozen context catalog JSON; no query match means no catalog bias",
    )
    parser.add_argument(
        "--context-query",
        default=None,
        help="caller-owned meeting/topic context used to retrieve catalog phrases",
    )
    parser.add_argument("--context-limit", type=int, default=8)
    parser.add_argument("--context-min-score", type=float, default=0.55)
    parser.add_argument(
        "--context-tag",
        action="append",
        default=[],
        help="require a catalog tag; repeat to require multiple tags",
    )
    parser.add_argument(
        "--formats", default="all", help="comma list of json,observed,normalized,md,srt,vtt"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="suppress progress lines on stderr")
    return parser


def run_transcription(
    args: argparse.Namespace,
    *,
    adapter: ASRAdapter | None = None,
    second_ear: ASRAdapter | None = None,
    phone_observer: Any | None = None,
) -> dict[str, Any]:
    if phone_observer is None:
        phone_observer = build_phone_observer(args)
    hotwords = tuple(
        value.strip()
        for value in str(args.hotwords or "").replace("、", ",").split(",")
        if value.strip()
    )
    formats = (
        None
        if args.formats == "all"
        else {value.strip() for value in args.formats.split(",") if value.strip()}
    )

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"[semantic-asr] {message}", file=sys.stderr, flush=True)

    result = transcribe(
        args.audio,
        profile=args.profile,
        language=args.language,
        hotwords=hotwords,
        initial_prompt=args.initial_prompt,
        catalog=args.catalog,
        context_query=args.context_query,
        context_limit=args.context_limit,
        context_min_score=args.context_min_score,
        context_tags=tuple(args.context_tag),
        on_progress=progress,
        adapter=adapter,
        second_ear=second_ear,
        phone_observer=phone_observer,
    )
    outputs = result.write(args.output_dir, overwrite=args.overwrite, formats=formats)
    return {
        "status": "ok",
        "profile": result.profile.name,
        "profileDigest": result.profile.digest,
        "durationMs": result.duration_ms,
        "segments": len(result.segments),
        "provisionalSegments": result.provisional_segment_count,
        "evidenceSha256": result.evidence_sha256,
        "provenance": result.provenance,
        "outputs": {name: str(Path(path)) for name, path in outputs.items()},
    }


def build_phone_observer(args: argparse.Namespace):
    directory = getattr(args, "phone_model_dir", None)
    digest = getattr(args, "phone_artifact_sha256", None)
    if directory is None and digest is None:
        if getattr(args, "phone_check_candidates", False):
            raise ValueError("phone candidate checks require a local phone model")
        return None
    if not directory or not digest:
        raise ValueError("phone model directory and artifact SHA-256 must be supplied together")
    from .audio_phone_runtime import LocalHubertPhoneObserver

    return LocalHubertPhoneObserver(
        directory,
        artifact_sha256=digest,
        check_candidates=getattr(args, "phone_check_candidates", False),
    )


def build_profile_adapters(args: argparse.Namespace) -> tuple[ASRAdapter | None, ASRAdapter | None]:
    """Construct explicitly requested local adapters for the named profile."""

    reazon_values = (args.reazon_model_dir, args.reazon_artifact_sha256)
    parakeet_values = (
        args.second_ear_parakeet_model_dir,
        args.second_ear_parakeet_artifact_sha256,
    )
    whisper_values = (args.whisper_model_dir, args.whisper_artifact_sha256)
    qwen_values = (args.qwen_model_dir, args.qwen_artifact_sha256)
    if args.profile == "qwen-ja-cpu-v1" or any(v is not None for v in qwen_values):
        if args.profile != "qwen-ja-cpu-v1":
            raise ValueError("local Qwen artifacts require qwen-ja-cpu-v1")
        if any(v is None or not str(v).strip() for v in qwen_values):
            raise ValueError("Qwen requires both model directory and artifact SHA-256")
        if any(v is not None for v in (*reazon_values, *parakeet_values, *whisper_values)):
            raise ValueError("do not mix Qwen and other primary/second-ear artifact arguments")
        from .adapters import Qwen3ASRAdapter

        return Qwen3ASRAdapter(
            model=args.qwen_model_dir,
            artifact_sha256=args.qwen_artifact_sha256,
            dtype="float32",
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=256,
        ), None
    if any(value is not None for value in whisper_values):
        if args.profile != "whisper-native-cpu-v1":
            raise ValueError("local native Whisper artifacts require whisper-native-cpu-v1")
        if any(value is None or not str(value).strip() for value in whisper_values):
            raise ValueError("native Whisper requires both model directory and artifact SHA-256")
        if any(value is not None for value in (*reazon_values, *parakeet_values)):
            raise ValueError("do not mix native Whisper and Reazon artifact arguments")
        from .native_whisper_adapter import NativeWhisperAdapter

        return NativeWhisperAdapter(
            model=args.whisper_model_dir,
            artifact_sha256=args.whisper_artifact_sha256,
            device="cpu",
            compute_type="int8",
            cpu_threads=2,
        ), None
    if args.profile not in {"reazon-ja-v1", "reazon-ja-research-v1"}:
        if any(value is not None for value in (*reazon_values, *parakeet_values)):
            raise ValueError("local Reazon/Parakeet artifacts require a Reazon profile")
        return None, None
    if any(value is None for value in reazon_values):
        raise ValueError("Reazon profiles require --reazon-model-dir and --reazon-artifact-sha256")
    if any(value is None for value in parakeet_values):
        if any(value is not None for value in parakeet_values):
            raise ValueError(
                "Parakeet second-ear requires both model directory and artifact SHA-256"
            )
        second_ear = None
    else:
        if args.profile != "reazon-ja-research-v1":
            raise ValueError("Parakeet second-ear requires profile reazon-ja-research-v1")
        second_ear = ParakeetJapaneseCtcAdapter(
            args.second_ear_parakeet_model_dir,
            artifact_sha256=args.second_ear_parakeet_artifact_sha256,
        )
    primary = ReazonSpeechK2Adapter(
        args.reazon_model_dir,
        artifact_sha256=args.reazon_artifact_sha256,
    )
    return primary, second_ear


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "run":
        values = values[1:]
    parser = build_parser()
    args = parser.parse_args(values)
    try:
        phone_observer = build_phone_observer(args)
        adapter, second_ear = build_profile_adapters(args)
    except ValueError as exc:
        parser.error(str(exc))
    payload = run_transcription(
        args, adapter=adapter, second_ear=second_ear, phone_observer=phone_observer
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
