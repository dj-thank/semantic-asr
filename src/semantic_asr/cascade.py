from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from .adaptive import AdaptiveKConfig, AdaptiveKDecision, select_adaptive_k
from .candidate_pool import CandidatePool, aggregate_surface_candidates
from .contracts import CandidateEvidence, RankedCandidate
from .fusion import FusionConfig, fuse_candidates
from .mbr import MBRDecision, SemanticMBRConfig, semantic_minimum_bayes_risk
from .score_types import EvidenceScore


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


# Immutable, budgeted sparse-stage controller for research and deployment traces.

DecisionStatus = Literal[
    "unresolved",
    "accepted",
    "provisional",
    "rejected",
]
StageKind = Literal[
    "generator",
    "scorer",
    "reranker",
    "calibrator",
    "verifier",
    "second-ear",
    "proposal",
    "normalizer",
]


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    maximum_cost_ms: float
    maximum_stages: int = 16
    maximum_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.maximum_cost_ms < 0 or not math.isfinite(self.maximum_cost_ms):
            raise ValueError("maximum_cost_ms must be finite and non-negative")
        if self.maximum_stages < 1:
            raise ValueError("maximum_stages must be positive")
        if self.maximum_memory_mb is not None and (
            self.maximum_memory_mb < 0 or not math.isfinite(self.maximum_memory_mb)
        ):
            raise ValueError("maximum_memory_mb must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StageTrace:
    stage_id: str
    stage_kind: StageKind
    invoked: bool
    estimated_cost_ms: float
    measured_cost_ms: float
    reason: str
    input_digest: str
    output_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineState:
    candidate_pool: CandidatePool
    scores: dict[str, dict[str, EvidenceScore]] = field(default_factory=dict)
    posterior: dict[str, float] = field(default_factory=dict)
    selected_candidate_id: str | None = None
    generated_candidate_ids: tuple[str, ...] = ()
    verified_candidate_ids: tuple[str, ...] = ()
    entropy: float = 1.0
    disagreement: float = 1.0
    evidence_coverage: float = 0.0
    semantic_criticality: float = 0.0
    selective_risk: float = 1.0
    decision_status: DecisionStatus = "unresolved"
    metadata: dict[str, Any] = field(default_factory=dict)
    traces: tuple[StageTrace, ...] = ()

    def __post_init__(self) -> None:
        candidate_ids = {candidate.candidate_id for candidate in self.candidate_pool.candidates}
        if (
            self.selected_candidate_id is not None
            and self.selected_candidate_id not in candidate_ids
        ):
            raise ValueError("selected candidate is outside candidate pool")
        if set(self.scores) - candidate_ids:
            raise ValueError("scores reference candidates outside candidate pool")
        if set(self.posterior) - candidate_ids:
            raise ValueError("posterior references candidates outside candidate pool")
        for name, value in (
            ("entropy", self.entropy),
            ("disagreement", self.disagreement),
            ("evidence_coverage", self.evidence_coverage),
            ("semantic_criticality", self.semantic_criticality),
            ("selective_risk", self.selective_risk),
        ):
            if not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.decision_status == "accepted":
            if self.selected_candidate_id is None:
                raise ValueError("accepted state requires a selected candidate")
            if (
                self.selected_candidate_id in self.generated_candidate_ids
                and self.selected_candidate_id not in self.verified_candidate_ids
            ):
                raise ValueError("generated candidates cannot be accepted before verification")

    @property
    def digest(self) -> str:
        payload = {
            "candidatePool": [
                {
                    "candidateId": candidate.candidate_id,
                    "text": candidate.text,
                    "aggregateLogLikelihood": candidate.aggregate_log_likelihood,
                    "pathDigests": [path.provenance_digest for path in candidate.paths],
                }
                for candidate in self.candidate_pool.candidates
            ],
            "scores": {
                candidate_id: {
                    stream: {
                        "value": score.value,
                        "semantics": score.semantics.value,
                        "provenance": score.provenance.digest,
                        "calibrated": score.calibrated,
                    }
                    for stream, score in sorted(streams.items())
                }
                for candidate_id, streams in sorted(self.scores.items())
            },
            "posterior": self.posterior,
            "selectedCandidateId": self.selected_candidate_id,
            "generatedCandidateIds": self.generated_candidate_ids,
            "verifiedCandidateIds": self.verified_candidate_ids,
            "entropy": self.entropy,
            "disagreement": self.disagreement,
            "evidenceCoverage": self.evidence_coverage,
            "semanticCriticality": self.semantic_criticality,
            "selectiveRisk": self.selective_risk,
            "decisionStatus": self.decision_status,
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def with_score(
        self,
        candidate_id: str,
        stream: str,
        score: EvidenceScore,
    ) -> PipelineState:
        candidate_ids = {candidate.candidate_id for candidate in self.candidate_pool.candidates}
        if candidate_id not in candidate_ids:
            raise ValueError("score candidate is outside candidate pool")
        scores = {key: dict(value) for key, value in self.scores.items()}
        scores.setdefault(candidate_id, {})[stream] = score
        return replace(self, scores=scores)

    def with_trace(self, trace: StageTrace) -> PipelineState:
        return replace(self, traces=(*self.traces, trace))

    def accept(self, candidate_id: str) -> PipelineState:
        generated = candidate_id in self.generated_candidate_ids
        verified = candidate_id in self.verified_candidate_ids
        if generated and not verified:
            raise ValueError("generated candidate must be acoustically verified")
        return replace(
            self,
            selected_candidate_id=candidate_id,
            decision_status="accepted",
        )


@dataclass(frozen=True, slots=True)
class StageEstimate:
    cost_ms: float
    memory_mb: float | None = None
    expected_loss_reduction: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_ms < 0 or not math.isfinite(self.cost_ms):
            raise ValueError("stage cost must be finite and non-negative")
        if self.memory_mb is not None and (self.memory_mb < 0 or not math.isfinite(self.memory_mb)):
            raise ValueError("stage memory must be finite and non-negative")
        if not math.isfinite(self.expected_loss_reduction):
            raise ValueError("expected_loss_reduction must be finite")

    @property
    def utility(self) -> float:
        return self.expected_loss_reduction / max(1.0, self.cost_ms)


@dataclass(frozen=True, slots=True)
class StageExecution:
    state: PipelineState
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    stage_id: str
    stage_kind: StageKind

    def estimate(self, state: PipelineState) -> StageEstimate: ...

    def should_run(self, state: PipelineState) -> tuple[bool, str]: ...

    def run(self, state: PipelineState) -> StageExecution: ...


@dataclass(frozen=True, slots=True)
class FunctionStage:
    """Functional adapter for independently testable pipeline stages."""

    stage_id: str
    stage_kind: StageKind
    runner: Callable[[PipelineState], StageExecution]
    estimator: Callable[[PipelineState], StageEstimate]
    predicate: Callable[[PipelineState], tuple[bool, str]] = lambda _state: (
        True,
        "enabled",
    )

    def estimate(self, state: PipelineState) -> StageEstimate:
        return self.estimator(state)

    def should_run(self, state: PipelineState) -> tuple[bool, str]:
        return self.predicate(state)

    def run(self, state: PipelineState) -> StageExecution:
        return self.runner(state)


@dataclass(frozen=True, slots=True)
class SparseRoute:
    stage_id: str
    minimum_risk: float = 0.0
    minimum_entropy: float = 0.0
    minimum_disagreement: float = 0.0
    minimum_criticality: float = 0.0
    maximum_coverage: float = 1.0

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id is required")
        for name, value in (
            ("minimum_risk", self.minimum_risk),
            ("minimum_entropy", self.minimum_entropy),
            ("minimum_disagreement", self.minimum_disagreement),
            ("minimum_criticality", self.minimum_criticality),
            ("maximum_coverage", self.maximum_coverage),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def matches(self, state: PipelineState) -> bool:
        return (
            state.selective_risk >= self.minimum_risk
            and state.entropy >= self.minimum_entropy
            and state.disagreement >= self.minimum_disagreement
            and state.semantic_criticality >= self.minimum_criticality
            and state.evidence_coverage <= self.maximum_coverage
        )


@dataclass(frozen=True, slots=True)
class CascadeResult:
    state: PipelineState
    budget_ms: float
    estimated_used_ms: float
    measured_used_ms: float
    stopping_reason: str


class CascadeController:
    """Budgeted sparse stage executor with immutable state and complete traces."""

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        *,
        routes: Mapping[str, SparseRoute] | None = None,
        minimum_utility: float = 0.0,
    ) -> None:
        if len({stage.stage_id for stage in stages}) != len(stages):
            raise ValueError("stage IDs must be unique")
        if minimum_utility < 0 or not math.isfinite(minimum_utility):
            raise ValueError("minimum_utility must be finite and non-negative")
        self.stages = tuple(stages)
        self.routes = dict(routes or {})
        unknown = set(self.routes) - {stage.stage_id for stage in stages}
        if unknown:
            raise ValueError(f"routes reference unknown stages: {unknown}")
        self.minimum_utility = minimum_utility

    def execute(
        self,
        initial: PipelineState,
        *,
        budget: ExecutionBudget,
    ) -> CascadeResult:
        state = initial
        estimated_used = 0.0
        measured_used = 0.0
        invoked = 0
        stopping_reason = "pipeline-complete"

        for stage in self.stages:
            if invoked >= budget.maximum_stages:
                stopping_reason = "maximum-stage-count"
                break
            input_digest = state.digest
            should_run, reason = stage.should_run(state)
            route = self.routes.get(stage.stage_id)
            if route is not None and not route.matches(state):
                should_run = False
                reason = "sparse-route-not-matched"
            estimate = stage.estimate(state)
            if estimate.utility < self.minimum_utility:
                should_run = False
                reason = "below-utility-frontier"
            if estimated_used + estimate.cost_ms > budget.maximum_cost_ms:
                should_run = False
                reason = "estimated-cost-budget"
            if (
                budget.maximum_memory_mb is not None
                and estimate.memory_mb is not None
                and estimate.memory_mb > budget.maximum_memory_mb
            ):
                should_run = False
                reason = "memory-budget"

            if not should_run:
                state = state.with_trace(
                    StageTrace(
                        stage_id=stage.stage_id,
                        stage_kind=stage.stage_kind,
                        invoked=False,
                        estimated_cost_ms=estimate.cost_ms,
                        measured_cost_ms=0.0,
                        reason=reason,
                        input_digest=input_digest,
                    )
                )
                continue

            started = time.perf_counter()
            execution = stage.run(state)
            measured = (time.perf_counter() - started) * 1000.0
            if execution.state.traces != state.traces:
                raise ValueError("stages must not mutate traces; the controller owns the audit log")
            estimated_used += estimate.cost_ms
            measured_used += measured
            invoked += 1
            state = execution.state.with_trace(
                StageTrace(
                    stage_id=stage.stage_id,
                    stage_kind=stage.stage_kind,
                    invoked=True,
                    estimated_cost_ms=estimate.cost_ms,
                    measured_cost_ms=measured,
                    reason=reason,
                    input_digest=input_digest,
                    output_digest=execution.state.digest,
                    metadata=execution.metadata,
                )
            )
            if state.decision_status == "accepted":
                stopping_reason = "accepted"
                break
            if measured_used > budget.maximum_cost_ms:
                stopping_reason = "measured-cost-budget"
                break

        if state.decision_status == "unresolved":
            state = replace(state, decision_status="provisional")
            if stopping_reason == "pipeline-complete":
                stopping_reason = "unresolved-after-pipeline"
        return CascadeResult(
            state=state,
            budget_ms=budget.maximum_cost_ms,
            estimated_used_ms=estimated_used,
            measured_used_ms=measured_used,
            stopping_reason=stopping_reason,
        )


@dataclass(frozen=True, slots=True)
class DraftVerifyPolicy:
    """Guard for cheap-draft / acoustic-verifier execution."""

    maximum_draft_risk: float = 0.22
    mandatory_verify_generated: bool = True
    mandatory_verify_criticality: float = 0.75
    acceptance_verifier_probability: float = 0.80

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_draft_risk", self.maximum_draft_risk),
            ("mandatory_verify_criticality", self.mandatory_verify_criticality),
            ("acceptance_verifier_probability", self.acceptance_verifier_probability),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    def requires_verification(self, state: PipelineState, candidate_id: str) -> bool:
        return (
            (self.mandatory_verify_generated and candidate_id in state.generated_candidate_ids)
            or state.semantic_criticality >= self.mandatory_verify_criticality
            or state.selective_risk > self.maximum_draft_risk
        )

    def can_accept(
        self,
        state: PipelineState,
        candidate_id: str,
        *,
        verifier_probability: float | None,
    ) -> bool:
        if not self.requires_verification(state, candidate_id):
            return state.selective_risk <= self.maximum_draft_risk
        if candidate_id not in state.verified_candidate_ids:
            return False
        return (
            verifier_probability is not None
            and verifier_probability >= self.acceptance_verifier_probability
        )
