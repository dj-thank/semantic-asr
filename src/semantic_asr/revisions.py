from __future__ import annotations

import re
from collections.abc import Mapping

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

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
