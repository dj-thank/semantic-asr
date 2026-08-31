from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .longform import LongformResult


def _timecode(milliseconds: int, *, vtt: bool = False) -> str:
    milliseconds = max(0, int(milliseconds))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    separator = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def render_srt(result: LongformResult, *, normalized: bool = True) -> str:
    output: list[str] = []
    for index, segment in enumerate(result.segments, 1):
        text = segment.normalized.text if normalized else segment.observed.text
        output.extend(
            [
                str(index),
                f"{_timecode(segment.window.start_ms)} --> {_timecode(segment.window.end_ms)}",
                text.strip(),
                "",
            ]
        )
    return "\n".join(output).rstrip() + "\n"


def render_vtt(result: LongformResult, *, normalized: bool = True) -> str:
    output = ["WEBVTT", ""]
    for segment in result.segments:
        text = segment.normalized.text if normalized else segment.observed.text
        output.extend(
            [
                f"{_timecode(segment.window.start_ms, vtt=True)} --> "
                f"{_timecode(segment.window.end_ms, vtt=True)}",
                text.strip(),
                "",
            ]
        )
    return "\n".join(output).rstrip() + "\n"


def render_markdown(result: LongformResult) -> str:
    lines = [
        f"# {result.source_name} — Semantic ASR",
        "",
        f"- Audio SHA-256: `{result.source_audio_sha256}`",
        f"- Evidence SHA-256: `{result.evidence_sha256}`",
        f"- Duration: `{result.duration_ms / 1000:.3f}` seconds",
        f"- Provisional windows: `{result.diagnostics['provisionalWindowCount']}`",
        "",
        "## Observed transcript",
        "",
        result.observed_text,
        "",
        "## Normalized transcript",
        "",
        result.normalized_text,
        "",
        "## Timeline",
        "",
    ]
    for segment in result.segments:
        warnings = ", ".join(segment.normalized.semantic_change_warnings)
        warning_text = f" ⚠️ `{warnings}`" if warnings else ""
        lines.append(
            f"- `{_timecode(segment.window.start_ms, vtt=True)}` "
            f"[{segment.observed.decision}] {segment.normalized.text}{warning_text}"
        )
    return "\n".join(lines).rstrip() + "\n"


def atomic_write(path: str | Path, content: str, *, overwrite: bool = False) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, target)
    return target


def write_outputs(
    result: LongformResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    formats: set[str] | None = None,
) -> dict[str, str]:
    formats = formats or {"json", "observed", "normalized", "md", "srt", "vtt"}
    unknown = formats - {"json", "observed", "normalized", "md", "srt", "vtt"}
    if unknown:
        raise ValueError(f"unknown output formats: {sorted(unknown)}")
    root = Path(output_dir)
    stem = Path(result.source_name).stem
    payload: dict[str, Any] = result.as_dict()
    payload["contract"] = {
        "observedImmutable": True,
        "observedEvidenceSha256": result.evidence_sha256,
        "normalizationSeparate": True,
        "sourcePathExported": False,
    }
    renderers = {
        "json": (
            root / f"{stem}.semantic-asr.json",
            lambda: json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        ),
        "observed": (
            root / f"{stem}.observed.txt",
            lambda: result.observed_text.rstrip() + "\n",
        ),
        "normalized": (
            root / f"{stem}.txt",
            lambda: result.normalized_text.rstrip() + "\n",
        ),
        "md": (root / f"{stem}.md", lambda: render_markdown(result)),
        "srt": (root / f"{stem}.srt", lambda: render_srt(result)),
        "vtt": (root / f"{stem}.vtt", lambda: render_vtt(result)),
    }
    outputs: dict[str, str] = {}
    for name in sorted(formats):
        path, renderer = renderers[name]
        atomic_write(path, renderer(), overwrite=overwrite)
        outputs[name] = str(path)
    return outputs
