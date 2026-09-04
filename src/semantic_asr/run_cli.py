"""``semantic-asr run``: the one-call facade as a CLI vertical slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import ASRAdapter
from .api import PROFILES, transcribe

RUN_COMMANDS = {"run"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-asr run")
    parser.add_argument("audio", help="audio file (any format ffmpeg/PyAV can decode)")
    parser.add_argument("--profile", default="cpu-ja-v1", choices=sorted(PROFILES))
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
    args: argparse.Namespace, *, adapter: ASRAdapter | None = None
) -> dict[str, Any]:
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


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "run":
        values = values[1:]
    args = build_parser().parse_args(values)
    payload = run_transcription(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
