"""Probe a Qwen3-ASR second ear on CPU over a small manifest.

Prints one JSON line per clip with wall-clock seconds, the reference and the second-ear
hypothesis so that real-time factor and agreement with the Whisper pool can be inspected.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from semantic_asr.adapters import DecodeRequest, Qwen3ASRAdapter
from semantic_asr.revisions import QWEN_ASR_MODEL_REVISIONS, resolve_hugging_face_revision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--model-revision")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device-map", default="cpu")
    args = parser.parse_args()
    model_revision = resolve_hugging_face_revision(
        args.model,
        args.model_revision,
        QWEN_ASR_MODEL_REVISIONS,
    )
    started = time.perf_counter()
    adapter = Qwen3ASRAdapter(
        model=args.model,
        model_revision=model_revision,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    print(
        json.dumps(
            {
                "loadSeconds": round(time.perf_counter() - started, 1),
                "model": args.model,
                "modelRevision": model_revision,
            }
        ),
        flush=True,
    )
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        started = time.perf_counter()
        output = adapter.decode(DecodeRequest(audio_path=row["audioPath"], language="ja"))
        print(
            json.dumps(
                {
                    "sampleId": row["sampleId"],
                    "model": args.model,
                    "modelRevision": model_revision,
                    "durationSeconds": row.get("durationSeconds"),
                    "seconds": round(time.perf_counter() - started, 2),
                    "reference": row["reference"],
                    "hypotheses": [candidate.text for candidate in output],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
