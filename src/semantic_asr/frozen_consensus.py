"""Opt-in full-window selection; agreement is never a correctness probability."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .contracts import CandidateEvidence
from .evaluation import normalize_characters_lenient

ROLES = ("whisper", "qwen", "reazon", "parakeet")
POLICY_ID = "qwen-reazon-parakeet-unanimous-else-whisper-v1"


def select_frozen_consensus(
    candidates: Mapping[str, CandidateEvidence],
) -> CandidateEvidence:
    """Return an original candidate, without ranking, rewriting, or calibration.

    Four captured top-one hypotheses must describe the same native 16kHz PCM
    window. Metadata identities are caller evidence, not proof of live inference.
    The fixed rule compares NFKC text without whitespace, punctuation or symbols.
    It does not compare arbitrary confidence/likelihood values across engines.
    """
    if set(candidates) != set(ROLES):
        raise ValueError("exactly whisper, qwen, reazon and parakeet are required")
    windows = []
    models = []
    ids = []
    for role in ROLES:
        candidate = candidates[role]
        if not isinstance(candidate, CandidateEvidence):
            raise TypeError("inputs must be CandidateEvidence")
        metadata = candidate.metadata
        digest = metadata.get("audioWindowSha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("audioWindowSha256 must identify the decoded PCM window")
        start, count = metadata.get("startMs"), metadata.get("sampleCount")
        if type(start) is not int or start < 0 or type(count) is not int or not 0 < count <= 480000:
            raise ValueError("integer startMs and 1..480000 sampleCount are required")
        if (
            type(metadata.get("sampleRate")) is not int
            or metadata["sampleRate"] != 16000
            or metadata.get("language") != "ja"
        ):
            raise ValueError("only Japanese native 16kHz windows are supported")
        model = metadata.get("model")
        artifact = metadata.get("modelArtifactSha256")
        revision = artifact if artifact is not None else metadata.get("modelRevision")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model identity is required")
        pattern = r"[0-9a-f]{64}" if artifact is not None else r"[0-9a-f]{40}|[0-9a-f]{64}"
        if not isinstance(revision, str) or not re.fullmatch(pattern, revision):
            raise ValueError("immutable model revision or artifact SHA-256 is required")
        if candidate.rank != 1:
            raise ValueError("the fixed rule requires native top-one hypotheses")
        windows.append((digest, start, count))
        models.append(revision)
        ids.append(candidate.candidate_id)
    if len(set(windows)) != 1:
        raise ValueError("all candidates must cover the identical audio window")
    if len(set(models)) != 4 or len(set(ids)) != 4:
        raise ValueError("four distinct model identities and candidate IDs are required")
    support = [normalize_characters_lenient(candidates[role].text) for role in ROLES[1:]]
    if support[0] and support[0] == support[1] == support[2]:
        return candidates["qwen"]
    return candidates["whisper"]
