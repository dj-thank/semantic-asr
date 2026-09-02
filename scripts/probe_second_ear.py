"""Probe a Qwen3-ASR second ear on CPU over a small local manifest.

The default JSONL output is metadata-only: references and decoded hypotheses are
omitted so that a redirected probe log is not a transcript export.  Pass
``--local-research-output`` for an explicitly local, external output destination
when inspecting those fields is necessary.  Never publish that opt-in output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TextIO

from semantic_asr.adapters import DecodeRequest, Qwen3ASRAdapter
from semantic_asr.revisions import QWEN_ASR_MODEL_REVISIONS, resolve_hugging_face_revision

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def ensure_safe_output_path(output: str | Path) -> Path:
    """Resolve a probe log destination and reject paths inside this checkout."""

    resolved = Path(output).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("probe output must not be a filesystem root")
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError(
        "probe output must be outside the repository checkout; "
        "use an external local-research directory"
    )


def _output_handle(path: str | None) -> tuple[TextIO, TextIO | None]:
    if path is None or path == "-":
        return sys.stdout, None
    target = ensure_safe_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("w", encoding="utf-8", newline="\n")
    return handle, handle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--model-revision")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument(
        "--output",
        help="JSONL output path outside the checkout; omit or use '-' for stdout",
    )
    parser.add_argument(
        "--local-research-output",
        "--include-sensitive",
        "--include-raw",
        action="store_true",
        help="explicitly include raw references and hypotheses for local research only",
    )
    args = parser.parse_args()
    model_revision = resolve_hugging_face_revision(
        args.model,
        args.model_revision,
        QWEN_ASR_MODEL_REVISIONS,
    )
    handle, closer = _output_handle(args.output)

    def emit(payload: dict[str, object]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    try:
        started = time.perf_counter()
        adapter = Qwen3ASRAdapter(
            model=args.model,
            model_revision=model_revision,
            dtype=args.dtype,
            device_map=args.device_map,
        )
        emit(
            {
                "loadSeconds": round(time.perf_counter() - started, 1),
                "model": args.model,
                "modelRevision": model_revision,
                "rawFieldsIncluded": args.local_research_output,
            }
        )
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            started = time.perf_counter()
            output = adapter.decode(DecodeRequest(audio_path=row["audioPath"], language="ja"))
            payload: dict[str, object] = {
                "sampleId": row["sampleId"],
                "model": args.model,
                "modelRevision": model_revision,
                "durationSeconds": row.get("durationSeconds"),
                "seconds": round(time.perf_counter() - started, 2),
                "hypothesisCount": len(output),
                "rawFieldsIncluded": args.local_research_output,
            }
            if args.local_research_output:
                payload.update(
                    {
                        "reference": row.get("reference"),
                        "hypotheses": [candidate.text for candidate in output],
                    }
                )
            emit(payload)
    finally:
        if closer is not None:
            closer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
