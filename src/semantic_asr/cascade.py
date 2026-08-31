from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .adaptive import AdaptiveKConfig, AdaptiveKDecision, select_adaptive_k
from .candidate_pool import aggregate_surface_candidates
from .contracts import CandidateEvidence, RankedCandidate
from .fusion import FusionConfig, fuse_candidates
from .mbr import MBRDecision, SemanticMBRConfig, semantic_minimum_bayes_risk


@dataclass(frozen=True, slots=True)
class CascadeConfig:
    selection_policy: str = "fusion"
    maximum_fusion_margin_for_mbr_tiebreak: float = 0.12
    minimum_mbr_risk_margin: float = 0.025
    disagreement_requires_evidence: bool = True

    def __post_init__(self) -> None:
        if self.selection_policy not in {"fusion", "mbr-tiebreak"}:
            raise ValueError("selection_policy must be fusion or mbr-tiebreak")
        if self.maximum_fusion_margin_for_mbr_tiebreak < 0:
            raise ValueError("fusion margin threshold must be non-negative")
        if self.minimum_mbr_risk_margin < 0:
            raise ValueError("MBR margin threshold must be non-negative")


@dataclass(frozen=True, slots=True)
class CascadeDecision:
    selected_candidate_id: str
    selected_text: str
    ranked: tuple[RankedCandidate, ...]
    mbr: MBRDecision
    adaptive_k: AdaptiveKDecision
    fusion_mbr_agree: bool
    requires_additional_evidence: bool
    reasons: tuple[str, ...]
    path_aggregated_candidate_count: int


def run_candidate_cascade(
    candidates: Sequence[CandidateEvidence],
    *,
    fusion_config: FusionConfig | None = None,
    mbr_config: SemanticMBRConfig | None = None,
    adaptive_config: AdaptiveKConfig | None = None,
    cascade_config: CascadeConfig | None = None,
    semantic_criticality: float = 0.0,
) -> CascadeDecision:
    cascade_config = cascade_config or CascadeConfig()
    pooled = aggregate_surface_candidates(candidates, id_prefix="cascade")
    if not pooled:
        raise ValueError("at least one candidate is required")
    ranked = fuse_candidates(pooled, fusion_config)
    gate = ranked[0].gate
    mbr = semantic_minimum_bayes_risk(
        pooled,
        posterior=gate.posterior,
        config=mbr_config,
    )
    adaptive = select_adaptive_k(
        pooled,
        gate.posterior,
        selective_risk=gate.selective_risk,
        semantic_criticality=semantic_criticality,
        config=adaptive_config,
    )
    fusion_id = ranked[0].candidate.candidate_id
    agree = fusion_id == mbr.selected_candidate_id
    selected_id = fusion_id
    reasons: list[str] = []
    if not agree:
        reasons.append("fusion-mbr-disagreement")
        posterior_order = sorted(gate.posterior.values(), reverse=True)
        fusion_margin = posterior_order[0] - posterior_order[1] if len(posterior_order) > 1 else 1.0
        if (
            cascade_config.selection_policy == "mbr-tiebreak"
            and fusion_margin <= cascade_config.maximum_fusion_margin_for_mbr_tiebreak
            and mbr.risk_margin >= cascade_config.minimum_mbr_risk_margin
        ):
            selected_id = mbr.selected_candidate_id
            reasons.append("semantic-mbr-tiebreak")
    if gate.needs_relisten:
        reasons.extend(gate.reasons)
    requires = bool(
        gate.needs_relisten
        or (
            cascade_config.disagreement_requires_evidence and not agree and selected_id == fusion_id
        )
    )
    selected = next(candidate for candidate in pooled if candidate.candidate_id == selected_id)
    return CascadeDecision(
        selected_candidate_id=selected_id,
        selected_text=selected.text,
        ranked=tuple(ranked),
        mbr=mbr,
        adaptive_k=adaptive,
        fusion_mbr_agree=agree,
        requires_additional_evidence=requires,
        reasons=tuple(dict.fromkeys(reasons)),
        path_aggregated_candidate_count=len(pooled),
    )
