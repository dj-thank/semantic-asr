from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_SHA256 = re.compile(r"[0-9a-f]{64}")

FASTER_WHISPER_MODEL_REVISIONS: dict[str, str] = {
    "large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": ("0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
}

QWEN_ASR_MODEL_REVISIONS: dict[str, str] = {
    "Qwen/Qwen3-ASR-0.6B": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
    "Qwen/Qwen3-ASR-1.7B": "7278e1e70fe206f11671096ffdd38061171dd6e5",
}

QWEN_FORCED_ALIGNER_REVISIONS: dict[str, str] = {
    "Qwen/Qwen3-ForcedAligner-0.6B": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
}

PUBLIC_DATASET_REVISIONS: dict[str, str] = {
    "japanese-asr/ja_asr.reazonspeech_test": ("dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f"),
}


def resolve_hugging_face_revision(
    identifier: str,
    explicit_revision: str | None,
    known_revisions: Mapping[str, str],
) -> str:
    """Return an immutable Hub commit for a known or explicitly pinned object."""

    revision = explicit_revision or known_revisions.get(identifier)
    if revision is None:
        raise ValueError(f"an exact 40-character revision is required for {identifier!r}")
    revision = revision.lower()
    if _COMMIT_SHA.fullmatch(revision) is None:
        raise ValueError(f"revision for {identifier!r} must be an exact 40-character commit SHA")
    return revision


def validate_artifact_sha256(value: str, *, identifier: str = "artifact") -> str:
    """Validate and normalize a separately tracked local-artifact digest.

    A local model directory has no Hub commit identity.  Its bytes therefore use
    this independent SHA-256 contract instead of being passed through
    :func:`resolve_hugging_face_revision`.
    """

    digest = str(value).strip().lower()
    if _ARTIFACT_SHA256.fullmatch(digest) is None:
        raise ValueError(f"{identifier} must be an exact 64-character SHA-256 digest")
    return digest


def sha256_artifact(path: str | Path) -> str:
    """Hash one local model artifact file or directory deterministically.

    Files are hashed as raw bytes.  For directories, the digest binds the
    normalized relative POSIX path and bytes of every regular file in sorted
    order, with an explicit contract marker so a directory cannot collide with
    a single-file artifact containing the same bytes.  Symlinks are read through
    their target, matching the bytes consumed by a local loader.
    """

    source = Path(path).expanduser()
    if source.is_file():
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not source.is_dir():
        raise FileNotFoundError(source)

    digest = hashlib.sha256(b"semantic-asr-local-directory-artifact-v1\0")
    files = sorted(
        (entry for entry in source.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(source).as_posix(),
    )
    for entry in files:
        relative = entry.relative_to(source).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_artifact_sha256(
    path: str | Path,
    expected: str,
    *,
    identifier: str = "model artifact",
) -> str:
    """Verify a local artifact and return its normalized digest."""

    expected_digest = validate_artifact_sha256(expected, identifier=identifier)
    actual_digest = sha256_artifact(path)
    if actual_digest != expected_digest:
        raise ValueError(
            f"{identifier} SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )
    return actual_digest
