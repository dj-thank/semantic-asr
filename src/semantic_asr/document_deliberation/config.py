"""Bounded search configuration for joint document deliberation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..audio import require_integer
from ..contracts import sha256_json
from ..deliberation_evidence import _strict_float
from .context_types import ContextArm


@dataclass(frozen=True, slots=True)
class DocumentBeamConfig:
    """Bounded document search and application policy."""

    enabled: bool = True
    local_paths_per_window: int = 6
    beam_size: int = 96
    global_rescore_paths: int = 32
    overlap_consistency_weight: float = 0.65
    maximum_overlap_similarity_regression: float = 0.35
    minimum_overlap_characters: int = 2
    change_penalty: float = 0.02
    generated_penalty: float = 0.08
    global_context_weight: float = 1.0
    maximum_document_audio_regression: float = 0.10
    maximum_changed_windows: int | None = None
    maximum_generated_windows: int = 2
    minimum_final_margin: float = 0.02
    proposal_context_arm: ContextArm = "bidirectional-offline"
    maximum_left_windows: int = 4
    maximum_right_windows: int = 4
    maximum_context_characters: int = 12_000
    require_sequence_scorer: bool = True
    apply_provisional: bool = False
    fail_closed_to_first_pass: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "local_paths_per_window",
            "beam_size",
            "global_rescore_paths",
            "minimum_overlap_characters",
            "maximum_generated_windows",
            "maximum_left_windows",
            "maximum_right_windows",
            "maximum_context_characters",
        ):
            require_integer(getattr(self, name), name=name)
        if self.local_paths_per_window < 1 or self.beam_size < 1 or self.global_rescore_paths < 1:
            raise ValueError("document beam sizes must be positive")
        if self.minimum_overlap_characters < 1:
            raise ValueError("minimum_overlap_characters must be positive")
        if self.maximum_changed_windows is not None:
            require_integer(self.maximum_changed_windows, name="maximum_changed_windows")
        for name in (
            "overlap_consistency_weight",
            "maximum_overlap_similarity_regression",
            "change_penalty",
            "generated_penalty",
            "global_context_weight",
            "maximum_document_audio_regression",
            "minimum_final_margin",
        ):
            value = _strict_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "enabled",
            "require_sequence_scorer",
            "apply_provisional",
            "fail_closed_to_first_pass",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.proposal_context_arm not in {
            "none",
            "declared-only",
            "left-only",
            "bidirectional-offline",
            "shuffled-context",
        }:
            raise ValueError("unknown proposal_context_arm")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))
