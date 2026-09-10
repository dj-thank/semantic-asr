"""Opt-in, offline ReazonSpeech transcription with separate numeric normalization."""

from __future__ import annotations

import argparse
import json

from run_real_audio_pipeline import ensure_safe_output_dir

from semantic_asr.api import transcribe
from semantic_asr.contracts import sha256_json
from semantic_asr.hayamimi_itn import convert
from semantic_asr.parakeet_adapter import ParakeetJapaneseCtcAdapter
from semantic_asr.reazon_adapter import ReazonSpeechK2Adapter


def validate_second_ear_args(args: argparse.Namespace) -> None:
    if bool(args.second_ear_model_dir) != bool(args.second_ear_model_sha256):
        raise ValueError(
            "--second-ear-model-dir and --second-ear-model-sha256 must be supplied together"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--second-ear-model-dir")
    parser.add_argument("--second-ear-model-sha256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-local-research", action="store_true")
    parser.add_argument("--normalize-numbers", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not args.allow_local_research:
        parser.error("transcript export requires --allow-local-research")
    try:
        validate_second_ear_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    out = ensure_safe_output_dir(args.output_dir)
    if out.exists():
        parser.error("output already exists; choose a new directory")
    adapter = ReazonSpeechK2Adapter(
        args.model_dir, artifact_sha256=args.model_sha256, cpu_threads=args.threads
    )
    second_ear = (
        None
        if args.second_ear_model_dir is None
        else ParakeetJapaneseCtcAdapter(
            args.second_ear_model_dir,
            artifact_sha256=args.second_ear_model_sha256,
            cpu_threads=args.threads,
        )
    )
    profile = "reazon-ja-research-v1" if second_ear is not None else "reazon-ja-v1"
    result = transcribe(
        args.audio,
        profile=profile,
        language="ja",
        adapter=adapter,
        second_ear=second_ear,
    )
    result.verify()
    out.mkdir(parents=True)
    (out / "observed.txt").write_text(result.observed_text + "\n", encoding="utf-8")
    (out / "evidence.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    normalized = (
        convert(result.normalized_text, "ja") if args.normalize_numbers else result.normalized_text
    )
    layer = {
        "schema": "semantic-asr-normalized-number-view-v1",
        "text": normalized,
        "parentEvidenceSha256": result.evidence_sha256,
        "normalizer": "hayamimi-conservative-cjk-itn" if args.normalize_numbers else "existing",
        "observedUnchanged": True,
    }
    layer["sha256"] = sha256_json(layer)
    (out / "normalized.txt").write_text(normalized + "\n", encoding="utf-8")
    (out / "normalized.json").write_text(
        json.dumps(layer, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "profile": result.profile.name,
                "segments": len(result.segments),
                "provisionalSegments": result.provisional_segment_count,
                "secondEar": result.provenance.get("secondEar"),
                "evidence_sha256": result.evidence_sha256,
                "normalized_view_sha256": layer["sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
