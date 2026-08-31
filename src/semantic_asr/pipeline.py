from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from .contracts import canonical_json

CostTier = Literal["free", "cheap", "moderate", "expensive"]
EffortName = Literal["ultra-light", "cpu-quality", "edge-gpu", "research"]


@dataclass(frozen=True, slots=True)
class ArtifactEnvelope:
    kind: str
    schema_version: str
    payload: Any
    sha256: str
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        payload: Any,
        schema_version: str = "1",
        provenance: Mapping[str, Any] | None = None,
    ) -> ArtifactEnvelope:
        if not kind or not schema_version:
            raise ValueError("artifact kind and schema version are required")
        canonical = {
            "kind": kind,
            "schemaVersion": schema_version,
            "payload": payload,
            "provenance": dict(provenance or {}),
        }
        digest = hashlib.sha256(canonical_json(canonical).encode("utf-8")).hexdigest()
        return cls(
            kind=kind,
            schema_version=schema_version,
            payload=payload,
            sha256=digest,
            provenance=dict(provenance or {}),
        )

    def verify(self) -> None:
        rebuilt = ArtifactEnvelope.create(
            kind=self.kind,
            schema_version=self.schema_version,
            payload=self.payload,
            provenance=self.provenance,
        )
        if rebuilt.sha256 != self.sha256:
            raise ValueError("pipeline artifact digest mismatch")


@dataclass(frozen=True, slots=True)
class StageSpec:
    name: str
    input_kind: str
    output_kind: str
    estimated_cost_ms: int = 0
    cost_tier: CostTier = "cheap"
    deterministic: bool = True
    optional: bool = False
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.input_kind or not self.output_kind:
            raise ValueError("stage name and artifact kinds are required")
        if self.estimated_cost_ms < 0:
            raise ValueError("estimated stage cost must be non-negative")
        if self.cost_tier not in {"free", "cheap", "moderate", "expensive"}:
            raise ValueError("unknown stage cost tier")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PipelineContext:
    run_id: str
    effort: EffortName
    total_budget_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("pipeline run_id is required")
        if self.effort not in {"ultra-light", "cpu-quality", "edge-gpu", "research"}:
            raise ValueError("unknown pipeline effort profile")
        if self.total_budget_ms < 0:
            raise ValueError("pipeline budget must be non-negative")


@dataclass(frozen=True, slots=True)
class StageExecution:
    spec: StageSpec
    input_sha256: str
    output: ArtifactEnvelope
    actual_cost_ms: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.input_sha256) != 64:
            raise ValueError("stage input digest must be SHA-256 hex")
        if self.actual_cost_ms < 0:
            raise ValueError("actual stage cost must be non-negative")
        self.output.verify()


@dataclass(frozen=True, slots=True)
class PipelineState:
    artifact: ArtifactEnvelope
    history: tuple[StageExecution, ...] = ()
    used_budget_ms: int = 0
    stopped: bool = False
    stop_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.used_budget_ms < 0:
            raise ValueError("used pipeline budget must be non-negative")
        self.artifact.verify()


class FunctionalStage(Protocol):
    spec: StageSpec

    def __call__(
        self, artifact: ArtifactEnvelope, context: PipelineContext
    ) -> StageExecution: ...


StageFunction = Callable[[ArtifactEnvelope, PipelineContext], tuple[Any, int, Mapping[str, Any]]]


class FunctionStage:
    def __init__(self, spec: StageSpec, function: StageFunction) -> None:
        self.spec = spec
        self.function = function

    def __call__(
        self, artifact: ArtifactEnvelope, context: PipelineContext
    ) -> StageExecution:
        artifact.verify()
        if artifact.kind != self.spec.input_kind:
            raise ValueError(
                f"stage {self.spec.name} expects {self.spec.input_kind}, got {artifact.kind}"
            )
        payload, actual_cost_ms, diagnostics = self.function(artifact, context)
        if (
            actual_cost_ms < 0
            or not math.isfinite(float(actual_cost_ms))
            or int(actual_cost_ms) != actual_cost_ms
        ):
            raise ValueError("stage function returned an invalid integer millisecond cost")
        output = ArtifactEnvelope.create(
            kind=self.spec.output_kind,
            payload=payload,
            provenance={
                "stage": self.spec.name,
                "stageVersion": self.spec.version,
                "stageDigest": self.spec.digest,
                "inputSha256": artifact.sha256,
                "runId": context.run_id,
            },
        )
        return StageExecution(
            spec=self.spec,
            input_sha256=artifact.sha256,
            output=output,
            actual_cost_ms=int(actual_cost_ms),
            diagnostics=dict(diagnostics),
        )


@dataclass(frozen=True, slots=True)
class ComputeEffortProfile:
    name: EffortName
    maximum_candidates: int
    evidence_budget_ms: int
    maximum_evidence_actions: int
    enable_neural_reranker: bool
    enable_acoustic_verifier: bool
    enable_second_ear: bool
    enable_offline_teacher: bool

    def __post_init__(self) -> None:
        if self.maximum_candidates < 1:
            raise ValueError("effort profile requires at least one candidate")
        if self.evidence_budget_ms < 0 or self.maximum_evidence_actions < 0:
            raise ValueError("effort profile budgets must be non-negative")


EFFORT_PROFILES: dict[EffortName, ComputeEffortProfile] = {
    "ultra-light": ComputeEffortProfile(
        name="ultra-light",
        maximum_candidates=5,
        evidence_budget_ms=0,
        maximum_evidence_actions=0,
        enable_neural_reranker=False,
        enable_acoustic_verifier=False,
        enable_second_ear=False,
        enable_offline_teacher=False,
    ),
    "cpu-quality": ComputeEffortProfile(
        name="cpu-quality",
        maximum_candidates=12,
        evidence_budget_ms=4_000,
        maximum_evidence_actions=4,
        enable_neural_reranker=True,
        enable_acoustic_verifier=False,
        enable_second_ear=False,
        enable_offline_teacher=True,
    ),
    "edge-gpu": ComputeEffortProfile(
        name="edge-gpu",
        maximum_candidates=16,
        evidence_budget_ms=12_000,
        maximum_evidence_actions=8,
        enable_neural_reranker=True,
        enable_acoustic_verifier=True,
        enable_second_ear=True,
        enable_offline_teacher=True,
    ),
    "research": ComputeEffortProfile(
        name="research",
        maximum_candidates=50,
        evidence_budget_ms=120_000,
        maximum_evidence_actions=32,
        enable_neural_reranker=True,
        enable_acoustic_verifier=True,
        enable_second_ear=True,
        enable_offline_teacher=True,
    ),
}


def effort_profile(name: str) -> ComputeEffortProfile:
    try:
        return EFFORT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown effort profile: {name}") from exc


class FunctionalPipeline:
    def __init__(self, stages: Sequence[FunctionalStage]) -> None:
        self.stages = tuple(stages)
        names = [stage.spec.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("pipeline stage names must be unique")
        for left, right in zip(self.stages, self.stages[1:]):
            if left.spec.output_kind != right.spec.input_kind:
                raise ValueError(
                    f"pipeline contract mismatch: {left.spec.name} emits "
                    f"{left.spec.output_kind}, {right.spec.name} expects "
                    f"{right.spec.input_kind}"
                )

    def run(
        self,
        initial: ArtifactEnvelope,
        *,
        context: PipelineContext,
    ) -> PipelineState:
        state = PipelineState(artifact=initial)
        for stage in self.stages:
            estimated_total = state.used_budget_ms + stage.spec.estimated_cost_ms
            if estimated_total > context.total_budget_ms:
                if stage.spec.optional and stage.spec.input_kind == stage.spec.output_kind:
                    state = PipelineState(
                        artifact=state.artifact,
                        history=state.history,
                        used_budget_ms=state.used_budget_ms,
                        stopped=False,
                        stop_reasons=state.stop_reasons
                        + (f"skipped-optional-stage-budget:{stage.spec.name}",),
                    )
                    continue
                reason = (
                    "optional-stage-contract-budget"
                    if stage.spec.optional
                    else "required-stage-budget"
                )
                return PipelineState(
                    artifact=state.artifact,
                    history=state.history,
                    used_budget_ms=state.used_budget_ms,
                    stopped=True,
                    stop_reasons=state.stop_reasons + (f"{reason}:{stage.spec.name}",),
                )
            execution = stage(state.artifact, context)
            used = state.used_budget_ms + execution.actual_cost_ms
            if used > context.total_budget_ms:
                return PipelineState(
                    artifact=state.artifact,
                    history=state.history,
                    used_budget_ms=state.used_budget_ms,
                    stopped=True,
                    stop_reasons=state.stop_reasons
                    + (f"actual-stage-budget:{stage.spec.name}",),
                )
            state = PipelineState(
                artifact=execution.output,
                history=state.history + (execution,),
                used_budget_ms=used,
                stopped=state.stopped,
                stop_reasons=state.stop_reasons,
            )
        return state
