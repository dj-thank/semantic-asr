"""Whole-document path hypothesis types."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from ..deliberation_lattice import LatticeArc
from ..japanese import join_timed_fragments
from .overlap_types import OverlapCompatibility
from .window_types import WindowPathOption


@dataclass(frozen=True, slots=True)
class DocumentPathHypothesis:
    options: tuple[WindowPathOption, ...]
    overlap_receipts: tuple[OverlapCompatibility, ...]
    base_score: float
    mean_audio_support: float
    context_score: float = 0.0
    final_score: float = 0.0
    scorer_source: str | None = None
    scorer_profile_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("document path requires at least one window option")
        if tuple(option.window_index for option in self.options) != tuple(range(len(self.options))):
            raise ValueError("document path window indices must be contiguous from zero")
        for name in ("base_score", "mean_audio_support", "context_score", "final_score"):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if not -1.0 <= self.mean_audio_support <= 1.0:
            raise ValueError("mean_audio_support must be in [-1, 1]")
        if not -1.0 <= self.context_score <= 1.0:
            raise ValueError("context_score must be in [-1, 1]")
        if (self.scorer_source is None) != (self.scorer_profile_digest is None):
            raise ValueError("scorer source and profile digest must be supplied together")
        if self.scorer_profile_digest is not None and not _is_sha256(self.scorer_profile_digest):
            raise ValueError("scorer_profile_digest must be a SHA-256 value")
        expected_pairs = {
            (left.window_index, right.window_index)
            for left_index, left in enumerate(self.options)
            for right in self.options[left_index + 1 :]
            if right.start_ms < left.end_ms
        }
        actual_pairs = {
            (receipt.left_window_index, receipt.right_window_index)
            for receipt in self.overlap_receipts
        }
        if actual_pairs != expected_pairs:
            raise ValueError("document path overlap receipts do not match overlapping windows")

    @property
    def changed_window_count(self) -> int:
        return sum(option.changed for option in self.options)

    @property
    def generated_window_count(self) -> int:
        return sum(option.generated for option in self.options)

    @property
    def text(self) -> str:
        return join_timed_fragments(
            (option.start_ms, option.end_ms, option.text) for option in self.options
        )

    @property
    def digest(self) -> str:
        """Stable path identity independent of later score attachment."""

        return sha256_json(
            {
                "optionDigests": [option.digest for option in self.options],
                "overlapReceiptDigests": [receipt.digest for receipt in self.overlap_receipts],
                "text": self.text,
            }
        )

    @property
    def score_digest(self) -> str:
        return sha256_json(
            {
                "pathDigest": self.digest,
                "baseScore": self.base_score,
                "meanAudioSupport": self.mean_audio_support,
                "contextScore": self.context_score,
                "finalScore": self.final_score,
                "scorerSource": self.scorer_source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )

    def scorer_arc(self, *, source_audio_sha256: str, config_digest: str) -> LatticeArc:
        return LatticeArc(
            arc_id=f"document-path:{self.digest[:20]}",
            span_id=f"document:{source_audio_sha256[:20]}",
            text=self.text,
            origin="first-pass",
            utilities=(),
            observed_eligible=True,
            source_audio_sha256=source_audio_sha256,
            metadata={
                "documentPathDigest": self.digest,
                "windowOptionDigests": tuple(option.digest for option in self.options),
                "overlapReceiptDigests": tuple(receipt.digest for receipt in self.overlap_receipts),
                "configDigest": config_digest,
            },
        )
