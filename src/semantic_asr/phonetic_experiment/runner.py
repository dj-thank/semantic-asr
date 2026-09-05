"""Two-phase reference-separated execution for phonetic proposal ablations."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .metrics import (
    PairedErrorDelta,
    PhoneticArmAggregate,
    PhoneticCaseArmMetrics,
    aggregate_arm,
    evaluate_case_arm,
    paired_bootstrap_error_delta,
)
from .planner import (
    FrozenPhoneticCandidatePlanner,
    FrozenPhoneticCandidatePool,
    PlanningCaseView,
)
from .protocol import PhoneticAblationManifest, PhoneticAblationProtocol
from .selection import PhoneticAblationDecision, select_phonetic_arm


@dataclass(frozen=True, slots=True)
class PreparedPhoneticAblation:
    manifest_digest: str
    protocol_digest: str
    planner_profile_digest: str
    pools: tuple[FrozenPhoneticCandidatePool, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for value in (
            self.manifest_digest,
            self.protocol_digest,
            self.planner_profile_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("prepared ablation digests must be SHA-256 values")
        if not self.pools:
            raise ValueError("prepared ablation requires candidate pools")
        if len({pool.case_id for pool in self.pools}) != len(self.pools):
            raise ValueError("prepared ablation case IDs must be unique")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "manifestDigest": self.manifest_digest,
                "protocolDigest": self.protocol_digest,
                "plannerProfileDigest": self.planner_profile_digest,
                "poolDigests": [pool.digest for pool in self.pools],
            }
        )

    def pool(self, case_id: str) -> FrozenPhoneticCandidatePool:
        for pool in self.pools:
            if pool.case_id == case_id:
                return pool
        raise KeyError(case_id)


@dataclass(frozen=True, slots=True)
class PhoneticAblationCaseResult:
    case_id: str
    planning_digest: str
    pool_digest: str
    reference_digest: str
    decisions: tuple[PhoneticAblationDecision, ...]
    metrics: tuple[PhoneticCaseArmMetrics, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.decisions or not self.metrics:
            raise ValueError("case result requires case ID, decisions, and metrics")
        if len({row.arm_name for row in self.decisions}) != len(self.decisions):
            raise ValueError("case result decision arm names must be unique")
        if {row.arm_name for row in self.decisions} != {row.arm_name for row in self.metrics}:
            raise ValueError("case result decision and metric arms differ")
        for value in (self.planning_digest, self.pool_digest, self.reference_digest):
            if not _is_sha256(value):
                raise ValueError("case result digests must be SHA-256 values")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "caseId": self.case_id,
                "planningDigest": self.planning_digest,
                "poolDigest": self.pool_digest,
                "referenceDigest": self.reference_digest,
                "decisionDigests": [row.digest for row in self.decisions],
                "metricDigests": [row.digest for row in self.metrics],
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticAblationReport:
    manifest_digest: str
    protocol_digest: str
    prepared_digest: str
    planner_profile_digest: str
    case_results: tuple[PhoneticAblationCaseResult, ...]
    aggregates: tuple[PhoneticArmAggregate, ...]
    paired_deltas: tuple[PairedErrorDelta, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for value in (
            self.manifest_digest,
            self.protocol_digest,
            self.prepared_digest,
            self.planner_profile_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("ablation report digests must be SHA-256 values")
        if not self.case_results or not self.aggregates:
            raise ValueError("ablation report requires case results and aggregates")
        case_ids = [row.case_id for row in self.case_results]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("ablation report case IDs must be unique")
        arm_names = [row.arm_name for row in self.aggregates]
        if len(arm_names) != len(set(arm_names)):
            raise ValueError("ablation report aggregate arm names must be unique")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "manifestDigest": self.manifest_digest,
                "protocolDigest": self.protocol_digest,
                "preparedDigest": self.prepared_digest,
                "plannerProfileDigest": self.planner_profile_digest,
                "caseResultDigests": [row.digest for row in self.case_results],
                "aggregateDigests": [row.digest for row in self.aggregates],
                "pairedDeltaDigests": [row.digest for row in self.paired_deltas],
            }
        )

    def as_dict(self, *, include_ranked_candidate_ids: bool = False) -> dict[str, object]:
        cases = []
        for result in self.case_results:
            decisions = []
            metrics_by_arm = {row.arm_name: row for row in result.metrics}
            for decision in result.decisions:
                metric = metrics_by_arm[decision.arm_name]
                decisions.append(
                    {
                        "armName": decision.arm_name,
                        "armDigest": decision.arm_digest,
                        "decisionDigest": decision.digest,
                        "status": decision.status,
                        "reason": decision.reason,
                        "margin": decision.margin,
                        "changedProposal": decision.changed_proposal,
                        "changedEffective": decision.changed_effective,
                        "proposedCandidateId": decision.proposed_candidate_id,
                        "effectiveCandidateId": decision.effective_candidate_id,
                        "firstPassSelectedCandidateId": (decision.first_pass_selected_candidate_id),
                        "referenceTextSha256": metric.reference_text_sha256,
                        "firstPassTextSha256": metric.first_pass_text_sha256,
                        "proposedTextSha256": metric.proposed_text_sha256,
                        "effectiveTextSha256": metric.effective_text_sha256,
                        "effectiveExact": metric.effective_exact,
                        "proposedExact": metric.proposed_exact,
                        "poolOracle": metric.pool_oracle,
                        "referenceOutsideFirstPass": metric.reference_outside_first_pass,
                        "recoveredOutsideFirstPass": (metric.recovered_outside_first_pass),
                        "falseCorrection": metric.false_correction,
                        "correctedFirstPass": metric.corrected_first_pass,
                        "introducedErrorCharacters": (metric.introduced_error_characters),
                        "correctedErrorCharacters": metric.corrected_error_characters,
                        "accepted": metric.accepted,
                        "critical": metric.critical,
                        "generationLatencyMs": metric.generation_latency_ms,
                        "selectionLatencyMs": metric.selection_latency_ms,
                        **(
                            {"rankedCandidateIds": [row.candidate_id for row in decision.ranked]}
                            if include_ranked_candidate_ids
                            else {}
                        ),
                    }
                )
            cases.append(
                {
                    "caseId": result.case_id,
                    "planningDigest": result.planning_digest,
                    "poolDigest": result.pool_digest,
                    "referenceDigest": result.reference_digest,
                    "resultDigest": result.digest,
                    "arms": decisions,
                }
            )
        return {
            "schemaVersion": self.schema_version,
            "manifestDigest": self.manifest_digest,
            "protocolDigest": self.protocol_digest,
            "preparedDigest": self.prepared_digest,
            "plannerProfileDigest": self.planner_profile_digest,
            "cases": cases,
            "aggregates": [
                {
                    **asdict(row),
                    "exactAccuracy": row.exact_accuracy,
                    "proposedExactAccuracy": row.proposed_exact_accuracy,
                    "oracleCoverage": row.oracle_coverage,
                    "outsideFirstPassRecoveryRate": row.outside_first_pass_recovery_rate,
                    "falseCorrectionRate": row.false_correction_rate,
                    "criticalExactAccuracy": row.critical_exact_accuracy,
                    "acceptedCoverage": row.accepted_coverage,
                    "characterErrorRate": row.character_error_rate,
                    "firstPassCharacterErrorRate": row.first_pass_character_error_rate,
                    "aggregateDigest": row.digest,
                }
                for row in self.aggregates
            ],
            "pairedDeltas": [
                {**asdict(row), "deltaDigest": row.digest} for row in self.paired_deltas
            ],
            "reportDigest": self.digest,
            "rawReferenceTextIncluded": False,
            "rawCandidateTextIncluded": False,
        }

    def write(
        self,
        path: str | Path,
        *,
        include_ranked_candidate_ids: bool = False,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    self.as_dict(include_ranked_candidate_ids=include_ranked_candidate_ids),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return destination


def _bootstrap_group_id(case, protocol: PhoneticAblationProtocol) -> str:
    if protocol.bootstrap_group == "speaker":
        return case.speaker_id
    if protocol.bootstrap_group == "session":
        return case.session_id
    if protocol.bootstrap_group == "source":
        return case.source_id
    raise ValueError("unknown bootstrap group")


def prepare_phonetic_ablation(
    manifest: PhoneticAblationManifest,
    protocol: PhoneticAblationProtocol,
    planner: FrozenPhoneticCandidatePlanner,
) -> PreparedPhoneticAblation:
    if planner.runtime.profile_digest != manifest.runtime_profile_digest:
        raise ValueError("planner runtime profile differs from the registered manifest")
    if planner.utility_artifact.digest != manifest.utility_artifact_digest:
        raise ValueError("planner utility artifact differs from the registered manifest")
    pools = tuple(
        planner.plan(PlanningCaseView.from_case(case), protocol=protocol) for case in manifest.cases
    )
    return PreparedPhoneticAblation(
        manifest_digest=manifest.digest,
        protocol_digest=protocol.digest,
        planner_profile_digest=planner.profile_digest,
        pools=pools,
    )


def evaluate_prepared_phonetic_ablation(
    manifest: PhoneticAblationManifest,
    protocol: PhoneticAblationProtocol,
    prepared: PreparedPhoneticAblation,
) -> PhoneticAblationReport:
    if prepared.manifest_digest != manifest.digest:
        raise ValueError("prepared pools belong to a different manifest")
    if prepared.protocol_digest != protocol.digest:
        raise ValueError("prepared pools belong to a different protocol")
    expected_case_ids = {case.case_id for case in manifest.cases}
    if {pool.case_id for pool in prepared.pools} != expected_case_ids:
        raise ValueError("prepared pool case IDs differ from the manifest")
    case_results: list[PhoneticAblationCaseResult] = []
    metrics_by_arm: dict[str, list[PhoneticCaseArmMetrics]] = {
        arm.name: [] for arm in protocol.arms
    }
    for case in manifest.cases:
        pool = prepared.pool(case.case_id)
        if pool.planning_digest != case.planning_digest:
            raise ValueError("prepared pool planning digest differs from the case")
        decisions = tuple(select_phonetic_arm(pool, arm) for arm in protocol.arms)
        metrics = tuple(
            evaluate_case_arm(
                pool,
                decision,
                case.reference,
                group_id=_bootstrap_group_id(case, protocol),
            )
            for decision in decisions
        )
        for row in metrics:
            metrics_by_arm[row.arm_name].append(row)
        case_results.append(
            PhoneticAblationCaseResult(
                case_id=case.case_id,
                planning_digest=case.planning_digest,
                pool_digest=pool.digest,
                reference_digest=case.reference.digest,
                decisions=decisions,
                metrics=metrics,
            )
        )
    aggregates = tuple(aggregate_arm(tuple(metrics_by_arm[arm.name])) for arm in protocol.arms)
    baseline = tuple(metrics_by_arm[protocol.baseline_arm])
    deltas = tuple(
        paired_bootstrap_error_delta(
            tuple(metrics_by_arm[arm.name]),
            baseline,
            resamples=protocol.bootstrap_resamples,
            seed=f"{protocol.bootstrap_seed}:{arm.name}:{protocol.baseline_arm}",
        )
        for arm in protocol.arms
        if arm.name != protocol.baseline_arm
    )
    return PhoneticAblationReport(
        manifest_digest=manifest.digest,
        protocol_digest=protocol.digest,
        prepared_digest=prepared.digest,
        planner_profile_digest=prepared.planner_profile_digest,
        case_results=tuple(case_results),
        aggregates=aggregates,
        paired_deltas=deltas,
    )


def run_phonetic_ablation(
    manifest: PhoneticAblationManifest,
    protocol: PhoneticAblationProtocol,
    planner: FrozenPhoneticCandidatePlanner,
) -> PhoneticAblationReport:
    prepared = prepare_phonetic_ablation(manifest, protocol, planner)
    return evaluate_prepared_phonetic_ablation(manifest, protocol, prepared)
