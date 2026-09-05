"""Two-phase execution and reporting for context × phonetic factorial experiments."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_experiment.planner import FrozenPhoneticCandidatePlanner
from .context_scorer import CandidateContextScorer
from .metrics import (
    ContextPhoneticArmAggregate,
    ContextPhoneticCaseMetrics,
    FactorialInteractionContrast,
    GroupedPairedContrast,
    aggregate_factorial_arm,
    evaluate_factorial_case_arm,
    grouped_factorial_interaction,
    grouped_paired_contrast,
)
from .planner import (
    PreparedContextPhoneticExperiment,
    prepare_context_phonetic_experiment,
)
from .protocol import (
    ContextPhoneticArm,
    ContextPhoneticManifest,
    ContextPhoneticProtocol,
)
from .selection import ContextPhoneticDecision, select_context_phonetic_arm


@dataclass(frozen=True, slots=True)
class ContextPhoneticCaseResult:
    case_id: str
    case_digest: str
    prepared_case_digest: str
    decisions: tuple[ContextPhoneticDecision, ...]
    metrics: tuple[ContextPhoneticCaseMetrics, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.decisions or not self.metrics:
            raise ValueError("factorial case result requires ID, decisions, and metrics")
        for value in (self.case_digest, self.prepared_case_digest):
            if not _is_sha256(value):
                raise ValueError("factorial case result digests must be SHA-256 values")
        if {row.arm_name for row in self.decisions} != {
            row.arm_name for row in self.metrics
        }:
            raise ValueError("factorial case decisions and metrics contain different arms")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "caseId": self.case_id,
                "caseDigest": self.case_digest,
                "preparedCaseDigest": self.prepared_case_digest,
                "decisionDigests": [row.digest for row in self.decisions],
                "metricDigests": [row.digest for row in self.metrics],
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPhoneticFactorialReport:
    manifest_digest: str
    manifest_planning_digest: str
    protocol_digest: str
    prepared_digest: str
    phonetic_planner_profile_digest: str
    context_scorer_source: str
    context_scorer_profile_digest: str
    case_results: tuple[ContextPhoneticCaseResult, ...]
    aggregates: tuple[ContextPhoneticArmAggregate, ...]
    contrasts: tuple[GroupedPairedContrast, ...]
    interaction: FactorialInteractionContrast
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for value in (
            self.manifest_digest,
            self.manifest_planning_digest,
            self.protocol_digest,
            self.prepared_digest,
            self.phonetic_planner_profile_digest,
            self.context_scorer_profile_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("factorial report digests must be SHA-256 values")
        if not self.context_scorer_source or not self.case_results or not self.aggregates:
            raise ValueError("factorial report requires scorer, case results, and aggregates")
        if len({row.case_id for row in self.case_results}) != len(self.case_results):
            raise ValueError("factorial report case IDs must be unique")
        if len({row.arm_name for row in self.aggregates}) != len(self.aggregates):
            raise ValueError("factorial aggregate arm names must be unique")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "manifestDigest": self.manifest_digest,
                "manifestPlanningDigest": self.manifest_planning_digest,
                "protocolDigest": self.protocol_digest,
                "preparedDigest": self.prepared_digest,
                "phoneticPlannerProfileDigest": self.phonetic_planner_profile_digest,
                "contextScorerSource": self.context_scorer_source,
                "contextScorerProfileDigest": self.context_scorer_profile_digest,
                "caseResultDigests": [row.digest for row in self.case_results],
                "aggregateDigests": [row.digest for row in self.aggregates],
                "contrastDigests": [row.digest for row in self.contrasts],
                "interactionDigest": self.interaction.digest,
            }
        )

    def as_dict(self, *, include_ranked_candidate_ids: bool = False) -> dict[str, object]:
        case_rows = []
        for result in self.case_results:
            metrics_by_arm = {row.arm_name: row for row in result.metrics}
            case_rows.append(
                {
                    "caseId": result.case_id,
                    "caseDigest": result.case_digest,
                    "preparedCaseDigest": result.prepared_case_digest,
                    "resultDigest": result.digest,
                    "arms": [
                        {
                            "armName": decision.arm_name,
                            "armDigest": decision.arm_digest,
                            "decisionDigest": decision.digest,
                            "status": decision.status,
                            "reason": decision.reason,
                            "margin": decision.margin,
                            "proposedCandidateId": decision.proposed_candidate_id,
                            "effectiveCandidateId": decision.effective_candidate_id,
                            "firstPassSelectedCandidateId": (
                                decision.first_pass_selected_candidate_id
                            ),
                            "changedProposal": decision.changed_proposal,
                            "changedEffective": decision.changed_effective,
                            "referenceTextSha256": metrics_by_arm[
                                decision.arm_name
                            ].reference_text_sha256,
                            "proposedTextSha256": metrics_by_arm[
                                decision.arm_name
                            ].proposed_text_sha256,
                            "effectiveTextSha256": metrics_by_arm[
                                decision.arm_name
                            ].effective_text_sha256,
                            "effectiveExact": metrics_by_arm[
                                decision.arm_name
                            ].effective_exact,
                            "referenceOutsideFirstPass": metrics_by_arm[
                                decision.arm_name
                            ].reference_outside_first_pass,
                            "recoveredOutsideFirstPass": metrics_by_arm[
                                decision.arm_name
                            ].recovered_outside_first_pass,
                            "falseCorrection": metrics_by_arm[
                                decision.arm_name
                            ].false_correction,
                            "critical": metrics_by_arm[decision.arm_name].critical,
                            "contextCondition": metrics_by_arm[
                                decision.arm_name
                            ].context_condition,
                            "contextDonorCaseId": metrics_by_arm[
                                decision.arm_name
                            ].context_donor_case_id,
                            **(
                                {
                                    "rankedCandidateIds": [
                                        row.candidate_id for row in decision.ranked
                                    ]
                                }
                                if include_ranked_candidate_ids
                                else {}
                            ),
                        }
                        for decision in result.decisions
                    ],
                }
            )
        return {
            "schemaVersion": self.schema_version,
            "manifestDigest": self.manifest_digest,
            "manifestPlanningDigest": self.manifest_planning_digest,
            "protocolDigest": self.protocol_digest,
            "preparedDigest": self.prepared_digest,
            "phoneticPlannerProfileDigest": self.phonetic_planner_profile_digest,
            "contextScorerSource": self.context_scorer_source,
            "contextScorerProfileDigest": self.context_scorer_profile_digest,
            "cases": case_rows,
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
                    "totalRuntimeMs": row.total_runtime_ms,
                    "aggregateDigest": row.digest,
                }
                for row in self.aggregates
            ],
            "contrasts": [
                {**asdict(row), "contrastDigest": row.digest} for row in self.contrasts
            ],
            "interaction": {
                **asdict(self.interaction),
                "interactionDigest": self.interaction.digest,
            },
            "reportDigest": self.digest,
            "rawCandidateTextIncluded": False,
            "rawReferenceTextIncluded": False,
            "rawContextIncluded": False,
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
                    self.as_dict(
                        include_ranked_candidate_ids=include_ranked_candidate_ids
                    ),
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


def _find_arm(
    protocol: ContextPhoneticProtocol,
    *,
    phonetic_arm_name: str,
    context_condition: str,
) -> ContextPhoneticArm:
    rows = [
        arm
        for arm in protocol.arms
        if arm.phonetic_arm_name == phonetic_arm_name
        and arm.context_condition == context_condition
    ]
    if len(rows) != 1:
        raise ValueError(
            "factorial protocol requires exactly one arm for "
            f"phonetic={phonetic_arm_name!r}, context={context_condition!r}"
        )
    return rows[0]


def evaluate_prepared_context_phonetic_experiment(
    manifest: ContextPhoneticManifest,
    protocol: ContextPhoneticProtocol,
    prepared: PreparedContextPhoneticExperiment,
) -> ContextPhoneticFactorialReport:
    if prepared.manifest_planning_digest != manifest.planning_digest:
        raise ValueError("prepared factorial pools belong to a different planning manifest")
    if prepared.protocol_digest != protocol.digest:
        raise ValueError("prepared factorial pools belong to a different protocol")
    if {case.case_id for case in prepared.cases} != {
        case.case_id for case in manifest.cases
    }:
        raise ValueError("prepared factorial case IDs differ from the manifest")

    metrics_by_arm: dict[str, list[ContextPhoneticCaseMetrics]] = {
        arm.name: [] for arm in protocol.arms
    }
    case_results: list[ContextPhoneticCaseResult] = []
    for case in manifest.cases:
        prepared_case = prepared.case(case.case_id)
        if prepared_case.case_planning_digest != case.planning_digest:
            raise ValueError("prepared factorial case planning digest differs")
        decisions = tuple(
            select_context_phonetic_arm(prepared_case, arm, protocol)
            for arm in protocol.arms
        )
        metrics = tuple(
            evaluate_factorial_case_arm(
                prepared_case,
                decision,
                case,
                protocol.arm(decision.arm_name),
                protocol,
            )
            for decision in decisions
        )
        for row in metrics:
            metrics_by_arm[row.arm_name].append(row)
        case_results.append(
            ContextPhoneticCaseResult(
                case_id=case.case_id,
                case_digest=case.digest,
                prepared_case_digest=prepared_case.digest,
                decisions=decisions,
                metrics=metrics,
            )
        )

    aggregates = tuple(
        aggregate_factorial_arm(tuple(metrics_by_arm[arm.name]))
        for arm in protocol.arms
    )
    target = protocol.arm(protocol.target_arm)
    baseline = protocol.arm(protocol.baseline_arm)
    shuffled = protocol.arm(protocol.shuffled_control_arm)
    target_none = _find_arm(
        protocol,
        phonetic_arm_name=target.phonetic_arm_name,
        context_condition="none",
    )
    baseline_ordered = _find_arm(
        protocol,
        phonetic_arm_name=baseline.phonetic_arm_name,
        context_condition="ordered",
    )
    contrast_pairs = (
        ("combined-vs-baseline", target.name, baseline.name),
        ("ordered-vs-shuffled", target.name, shuffled.name),
        ("phonetic-main-effect", target_none.name, baseline.name),
        ("context-main-effect", baseline_ordered.name, baseline.name),
    )
    contrasts = tuple(
        grouped_paired_contrast(
            name,
            tuple(metrics_by_arm[target_name]),
            tuple(metrics_by_arm[baseline_name]),
            resamples=protocol.bootstrap_resamples,
            seed=f"{protocol.shuffle_seed}:{name}:{target_name}:{baseline_name}",
        )
        for name, target_name, baseline_name in contrast_pairs
    )
    interaction = grouped_factorial_interaction(
        "context-by-phonetic-error-interaction",
        tuple(metrics_by_arm[target.name]),
        tuple(metrics_by_arm[target_none.name]),
        tuple(metrics_by_arm[baseline_ordered.name]),
        tuple(metrics_by_arm[baseline.name]),
        resamples=protocol.bootstrap_resamples,
        seed=f"{protocol.shuffle_seed}:factorial-interaction",
    )
    return ContextPhoneticFactorialReport(
        manifest_digest=manifest.digest,
        manifest_planning_digest=manifest.planning_digest,
        protocol_digest=protocol.digest,
        prepared_digest=prepared.digest,
        phonetic_planner_profile_digest=prepared.phonetic_planner_profile_digest,
        context_scorer_source=prepared.context_scorer_source,
        context_scorer_profile_digest=prepared.context_scorer_profile_digest,
        case_results=tuple(case_results),
        aggregates=aggregates,
        contrasts=contrasts,
        interaction=interaction,
    )


def run_context_phonetic_experiment(
    manifest: ContextPhoneticManifest,
    protocol: ContextPhoneticProtocol,
    phonetic_planner: FrozenPhoneticCandidatePlanner,
    context_scorer: CandidateContextScorer,
) -> ContextPhoneticFactorialReport:
    prepared = prepare_context_phonetic_experiment(
        manifest,
        protocol,
        phonetic_planner,
        context_scorer,
    )
    return evaluate_prepared_context_phonetic_experiment(
        manifest,
        protocol,
        prepared,
    )
