"""Semantic ASR: evidence-preserving Japanese speech recognition."""

from .adaptive import AdaptiveKConfig, AdaptiveKDecision, select_adaptive_k
from .api import (
    PROFILES,
    RuntimeProfile,
    TranscriptResult,
    TranscriptSegment,
    load_transcriber,
    transcribe,
    transcribe_segments,
)
from .candidate_pool import aggregate_surface_candidates, merge_candidate_pools
from .cascade import CascadeConfig, CascadeDecision, run_candidate_cascade
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
from .mbr import MBRDecision, SemanticMBRConfig, semantic_minimum_bayes_risk
from .score_semantics import EvidenceScore, ScoreKind

__all__ = [
    "PROFILES",
    "RuntimeProfile",
    "TranscriptResult",
    "TranscriptSegment",
    "load_transcriber",
    "transcribe",
    "transcribe_segments",
    "AdaptiveKConfig",
    "AdaptiveKDecision",
    "CandidateEvidence",
    "CascadeConfig",
    "CascadeDecision",
    "EvidenceScore",
    "FusionConfig",
    "GateDecision",
    "MBRDecision",
    "MoraUnit",
    "NormalizedTranscript",
    "ObservedTranscript",
    "RankedCandidate",
    "ScoreKind",
    "SemanticMBRConfig",
    "aggregate_surface_candidates",
    "fuse_candidates",
    "merge_candidate_pools",
    "mora_sequence",
    "run_candidate_cascade",
    "select_adaptive_k",
    "semantic_minimum_bayes_risk",
    "split_mora",
    "to_katakana",
]

__version__ = "0.2.0"
