"""Joint document-level decoding over per-window Semantic ASR lattices."""

from .config import DocumentBeamConfig
from .context import build_frozen_window_contexts
from .context_types import ContextArm, FrozenWindowContext
from .decision_types import DocumentDeliberationDecision, DocumentDeliberationPlan
from .local import _coverage_attribution
from .overlap_types import OverlapCompatibility
from .path_types import DocumentPathHypothesis
from .planning import plan_document_deliberation
from .runtime import (
    DocumentDeliberatingTranscriber,
    apply_document_deliberation,
    with_document_deliberation,
)
from .window_types import WindowPathOption, WindowPathSet

__all__ = [
    "ContextArm",
    "DocumentBeamConfig",
    "DocumentDeliberatingTranscriber",
    "DocumentDeliberationDecision",
    "DocumentDeliberationPlan",
    "DocumentPathHypothesis",
    "FrozenWindowContext",
    "OverlapCompatibility",
    "WindowPathOption",
    "WindowPathSet",
    "_coverage_attribution",
    "apply_document_deliberation",
    "build_frozen_window_contexts",
    "plan_document_deliberation",
    "with_document_deliberation",
]
