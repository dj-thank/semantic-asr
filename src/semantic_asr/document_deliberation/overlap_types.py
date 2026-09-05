"""Overlap compatibility receipts for document deliberation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..audio import require_integer
from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float


@dataclass(frozen=True, slots=True)
class OverlapCompatibility:
    left_window_index: int
    right_window_index: int
    overlap_start_ms: int
    overlap_end_ms: int
    left_text_sha256: str
    right_text_sha256: str
    similarity: float | None
    retained_similarity: float | None
    similarity_delta: float
    compared_characters: int
    compatible: bool

    def __post_init__(self) -> None:
        require_integer(self.left_window_index, name="left_window_index")
        require_integer(self.right_window_index, name="right_window_index")
        require_integer(self.overlap_start_ms, name="overlap_start_ms")
        require_integer(self.overlap_end_ms, name="overlap_end_ms")
        if self.overlap_end_ms <= self.overlap_start_ms:
            raise ValueError("overlap receipt requires a positive time intersection")
        if not _is_sha256(self.left_text_sha256) or not _is_sha256(self.right_text_sha256):
            raise ValueError("overlap text identities must be SHA-256 values")
        for name in ("similarity", "retained_similarity"):
            value = getattr(self, name)
            if value is not None:
                numeric = _strict_float(value, name=name)
                if not 0.0 <= numeric <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
                object.__setattr__(self, name, numeric)
        delta = _strict_float(self.similarity_delta, name="similarity_delta")
        if not -1.0 <= delta <= 1.0:
            raise ValueError("similarity_delta must be in [-1, 1]")
        require_integer(self.compared_characters, name="compared_characters")
        if not isinstance(self.compatible, bool):
            raise TypeError("compatible must be a boolean")
        object.__setattr__(self, "similarity_delta", delta)

    @property
    def duration_ms(self) -> int:
        return self.overlap_end_ms - self.overlap_start_ms

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))
