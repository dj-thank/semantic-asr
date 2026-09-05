from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
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
    """Publish one complete UTF-8 file without racing an existing destination.

    The temporary file is unique even between threads in one process. Exclusive
    publication uses a hard link in the same directory; unsupported filesystems
    fail safely rather than silently falling back to a racy exists/replace pair.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target) and not overwrite:
        raise FileExistsError(f"output already exists: {target}")
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def publish_output_documents(
    documents: Mapping[str, tuple[Path, str]], *, overwrite: bool = False
) -> dict[str, str]:
    """Preflight the whole set, then publish each file atomically.

    This is not a filesystem-wide transaction: a crash or a concurrently-created
    destination can leave a partial set. Existing files are never silently skipped
    or overwritten, and known conflicts/serialization failures precede any writes.
    """
    paths = [path for path, _ in documents.values()]
    if len(set(paths)) != len(paths):
        raise ValueError("output document destinations must be unique")
    for path in paths:
        if path.is_dir() or (os.path.lexists(path) and not overwrite):
            raise FileExistsError(f"output already exists: {path}")
    return {
        name: str(atomic_write(path, content, overwrite=overwrite))
        for name, (path, content) in documents.items()
    }


def render_output_documents(
    result: LongformResult,
    output_dir: str | Path,
    *,
    formats: set[str] | None = None,
) -> dict[str, tuple[Path, str]]:
    formats = {"json", "observed", "normalized", "md", "srt", "vtt"} if formats is None else formats
    unknown = formats - {"json", "observed", "normalized", "md", "srt", "vtt"}
    if unknown:
        raise ValueError(f"unknown output formats: {sorted(unknown)}")
    result.verify()
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
            lambda: json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
    return {name: (renderers[name][0], renderers[name][1]()) for name in sorted(formats)}


def write_outputs(
    result: LongformResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    formats: set[str] | None = None,
) -> dict[str, str]:
    documents = render_output_documents(result, output_dir, formats=formats)
    return publish_output_documents(documents, overwrite=overwrite)
