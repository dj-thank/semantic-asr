"""Semantic ASR: evidence-preserving Japanese speech recognition."""

from .contracts import (
    CandidateEvidence,
    GateDecision,
    MoraUnit,
    NormalizedTranscript,
    ObservedTranscript,
    RankedCandidate,
)
from .fusion import FusionConfig, fuse_candidates
from .japanese import mora_sequence, split_mora, to_katakana

__all__ = [
    "CandidateEvidence",
    "FusionConfig",
    "GateDecision",
    "MoraUnit",
    "NormalizedTranscript",
    "ObservedTranscript",
    "RankedCandidate",
    "fuse_candidates",
    "mora_sequence",
    "split_mora",
    "to_katakana",
]

__version__ = "0.1.0"
