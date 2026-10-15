from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._file_lock import exclusive_file_lock
from .longform import LongformResult

_BUNDLE_SCHEMA = "semantic-asr-output-bundle-v1"
_BUNDLE_LOCK = ".semantic-asr-export.lock"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _bundle_lock(root: Path):
    return exclusive_file_lock(
        root / _BUNDLE_LOCK,
        conflict_message=f"output directory already has an active writer: {root}",
    )


def _filename_identity(name: str) -> str:
    # Be conservative across case-insensitive Mac/Windows filesystems, including
    # Win32's trailing-dot/space aliases. Portable bundles must not alias a lock.
    return name.rstrip(" .").casefold()


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
    documents: Mapping[str, tuple[Path, str]],
    *,
    overwrite: bool = False,
    manifest_path: Path | None = None,
    evidence_sha256: str | None = None,
) -> dict[str, str]:
    """Publish complete files; optionally mark a verified bundle complete last.

    A bundle is not a filesystem transaction. Interrupted overwrites leave no
    completion manifest. Cooperating writers/readers share the directory lock;
    consumers must verify the manifest and every artifact, not just its presence.
    """
    if not documents:
        return {}
    paths = [path for path, _ in documents.values()]
    if len(
        {(str(path.parent.resolve()).casefold(), _filename_identity(path.name)) for path in paths}
    ) != len(paths):
        raise ValueError("output document destinations must be unique")
    if manifest_path is None:
        _preflight_outputs(paths, overwrite=overwrite)
        return {
            name: str(atomic_write(path, content, overwrite=overwrite))
            for name, (path, content) in documents.items()
        }
    root = manifest_path.parent.resolve()
    if (
        not _filename_identity(manifest_path.name)
        or _filename_identity(manifest_path.name) == _BUNDLE_LOCK
        or manifest_path.name.endswith((".", " "))
    ):
        raise ValueError("invalid output bundle manifest filename")
    if not _is_sha256(evidence_sha256):
        raise ValueError("bundle requires a verified evidence SHA-256")
    if "manifest" in documents or any(
        path.parent.resolve() != root
        or _filename_identity(path.name) in {_filename_identity(manifest_path.name), _BUNDLE_LOCK}
        for path in paths
    ):
        raise ValueError("bundle artifacts must have distinct names in the manifest directory")
    artifacts = {}
    for name, (path, content) in documents.items():
        data = content.encode("utf-8")
        artifacts[name] = {
            "filename": path.name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "sizeBytes": len(data),
        }
    manifest = {
        "schema": _BUNDLE_SCHEMA,
        "status": "complete",
        "evidenceSha256": evidence_sha256,
        "artifacts": artifacts,
    }
    _verify_bundle_artifacts(manifest, root, manifest_path.name, verify_files=False)
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    all_paths = [*paths, manifest_path]
    _preflight_outputs(all_paths, overwrite=overwrite)
    with _bundle_lock(root):
        # Recheck after acquiring the lock: another writer may have finished
        # between the first preflight and this point.
        _preflight_outputs(all_paths, overwrite=overwrite)
        manifest_path.unlink(missing_ok=True)
        published = {
            name: str(atomic_write(path, content, overwrite=overwrite))
            for name, (path, content) in documents.items()
        }
        _verify_bundle_artifacts(manifest, root, manifest_path.name)
        published["manifest"] = str(atomic_write(manifest_path, serialized))
        return published


def _preflight_outputs(paths: list[Path], *, overwrite: bool) -> None:
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"output must not be a symlink: {path}")
        if path.is_dir() or (os.path.lexists(path) and not overwrite):
            raise FileExistsError(f"output already exists: {path}")


def _verify_bundle_artifacts(
    manifest: Any, root: Path, manifest_name: str, *, verify_files: bool = True
) -> None:
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "status", "evidenceSha256", "artifacts"}
        or manifest["schema"] != _BUNDLE_SCHEMA
        or manifest["status"] != "complete"
        or not _is_sha256(manifest["evidenceSha256"])
        or not isinstance(manifest["artifacts"], dict)
        or not manifest["artifacts"]
    ):
        raise ValueError("invalid output bundle manifest")
    seen = set()
    for name, artifact in manifest["artifacts"].items():
        if (
            not isinstance(name, str)
            or not name
            or name == "manifest"
            or not isinstance(artifact, dict)
            or set(artifact) != {"filename", "sha256", "sizeBytes"}
        ):
            raise ValueError("invalid output bundle artifact")
        filename = artifact["filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or any(char in filename for char in ("/", "\\", ":", "\0"))
            or not _filename_identity(filename)
            or filename.endswith((".", " "))
            or _filename_identity(filename) in {_filename_identity(manifest_name), _BUNDLE_LOCK}
            or _filename_identity(filename) in seen
            or not _is_sha256(artifact["sha256"])
            or type(artifact["sizeBytes"]) is not int
            or artifact["sizeBytes"] < 0
        ):
            raise ValueError("invalid output bundle artifact identity")
        seen.add(_filename_identity(filename))
        if not verify_files:
            continue
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"bundle artifact is missing or not a regular file: {filename}")
        with path.open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if size != artifact["sizeBytes"] or digest != artifact["sha256"]:
            raise ValueError(f"bundle artifact does not match manifest: {filename}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in output bundle manifest")
        result[key] = value
    return result


def verify_output_bundle(
    manifest_path: str | Path, *, expected_evidence_sha256: str | None = None
) -> dict[str, Any]:
    """Verify an export under its writer lock; this is integrity, not authentication.

    The lock ends on return. Callers needing a lasting snapshot must use their own
    copy of the verified files; a later explicit overwrite can replace the bundle.
    Legacy exports without a completion manifest are not accepted as bundles.
    """
    path = Path(manifest_path)
    if not path.parent.is_dir():
        raise FileNotFoundError(path)
    with _bundle_lock(path.parent.resolve()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("output bundle manifest is missing or not a regular file")
        manifest = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
        _verify_bundle_artifacts(manifest, path.parent.resolve(), path.name)
        if (
            expected_evidence_sha256 is not None
            and manifest["evidenceSha256"] != expected_evidence_sha256
        ):
            raise ValueError("output bundle evidence SHA-256 does not match expected evidence")
        return manifest


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
    return publish_output_documents(
        documents,
        overwrite=overwrite,
        manifest_path=Path(output_dir)
        / f"{Path(result.source_name).stem}.semantic-asr.bundle.json",
        evidence_sha256=result.evidence_sha256,
    )
