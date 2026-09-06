"""Context receipt types for document deliberation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..audio import require_integer
from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..deliberation_lattice import DocumentContext

ContextArm = Literal[
    "none",
    "declared-only",
    "left-only",
    "bidirectional-offline",
    "shuffled-context",
]


@dataclass(frozen=True, slots=True)
class FrozenWindowContext:
    """One leakage-auditable context arm for a target first-pass window."""

    target_window_index: int
    arm: ContextArm
    context: DocumentContext
    source_window_indices: tuple[int, ...]
    first_pass_evidence_sha256: str
    shuffle_seed_digest: str | None = None

    def __post_init__(self) -> None:
        require_integer(self.target_window_index, name="target_window_index")
        if self.arm not in {
            "none",
            "declared-only",
            "left-only",
            "bidirectional-offline",
            "shuffled-context",
        }:
            raise ValueError("unknown document context arm")
        if not _is_sha256(self.first_pass_evidence_sha256):
            raise ValueError("first_pass_evidence_sha256 must be a SHA-256 value")
        for index in self.source_window_indices:
            require_integer(index, name="source_window_index")
        if self.target_window_index in self.source_window_indices:
            raise ValueError("target window must not be injected into its own context")
        if len(set(self.source_window_indices)) != len(self.source_window_indices):
            raise ValueError("context source window indices must be unique")
        metadata = self.context.metadata
        if metadata.get("targetWindowIndex") != self.target_window_index:
            raise ValueError("context metadata is not bound to the target window")
        if tuple(metadata.get("sourceWindowIndices", ())) != self.source_window_indices:
            raise ValueError("context metadata is not bound to its source windows")
        if metadata.get("firstPassEvidenceSha256") != self.first_pass_evidence_sha256:
            raise ValueError("context metadata is not bound to first-pass evidence")
        if metadata.get("contextArm") != self.arm:
            raise ValueError("context metadata is not bound to the declared arm")
        window_count = metadata.get("windowCount")
        require_integer(window_count, name="context windowCount", minimum=1)
        if self.target_window_index >= window_count or any(
            index >= window_count for index in self.source_window_indices
        ):
            raise ValueError("context receipt references a window outside the recording")
        if self.arm in {"none", "declared-only"} and self.source_window_indices:
            raise ValueError(f"{self.arm} cannot contain first-pass source windows")
        if self.arm == "left-only" and any(
            index >= self.target_window_index for index in self.source_window_indices
        ):
            raise ValueError("left-only context cannot contain current or future windows")
        if self.arm == "shuffled-context":
            if self.shuffle_seed_digest is None or not _is_sha256(self.shuffle_seed_digest):
                raise ValueError("shuffled context requires a frozen seed digest")
        elif self.shuffle_seed_digest is not None:
            raise ValueError("shuffle_seed_digest is only valid for shuffled context")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "targetWindowIndex": self.target_window_index,
                "arm": self.arm,
                "contextDigest": self.context.digest,
                "sourceWindowIndices": self.source_window_indices,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "shuffleSeedDigest": self.shuffle_seed_digest,
            }
        )
