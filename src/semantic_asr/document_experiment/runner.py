"""Two-phase, equal-candidate-budget document-context experiment runner."""

from __future__ import annotations

import json
import os
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from ..deliberation_lattice import DocumentContext
from ..document_deliberation import DocumentPathHypothesis
from ..longform import LongformResult
from .metrics import (
    ArmAggregateMetrics,
    CaseArmMetrics,
    PairedBootstrapInterval,
    aggregate_arm_metrics,
    paired_bootstrap_cer_delta,
    text_error_metrics,
    window_revision_metrics,
)
from .ngram_scorer import DocumentLanguageScore
from .protocol import (
    DocumentExperimentArm,
    DocumentExperimentCase,
    DocumentExperimentManifest,
    DocumentExperimentProtocol,
)


class DocumentArmScorer(Protocol):
    @property
    def profile_digest(self) -> str: ...

    def score_path(
        self,
        path: DocumentPathHypothesis,
        arm: DocumentExperimentArm,
        *,
        case_id: str,
        left_context: str,
        right_context: str,
        maximum_characters: int,
    ) -> DocumentLanguageScore: ...


@dataclass(frozen=True, slots=True)
class PlanningCaseView:
    """Reference-free view supplied to the candidate planner."""

    case_id: str
    first_pass: LongformResult = field(repr=False)
    source_audio_sha256: str
    planning_digest: str
    external_contexts: tuple[tuple[str, DocumentContext], ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not _is_sha256(self.source_audio_sha256):
            raise ValueError("planning view requires case ID and source-audio SHA-256")
        if not _is_sha256(self.planning_digest):
            raise ValueError("planning_digest must be a SHA-256 value")
        if self.first_pass.source_audio_sha256 != self.source_audio_sha256:
            raise ValueError("planning view first pass belongs to different audio")
        names = [name for name, _ in self.external_contexts]
        if len(names) != len(set(names)):
            raise ValueError("planning-view context names must be unique")

    def context(self, name: str | None) -> DocumentContext:
        if name is None:
            return DocumentContext()
        for candidate, context in self.external_contexts:
            if candidate == name:
                return context
        raise KeyError(name)


CandidatePlanner = Callable[[PlanningCaseView], object]


@dataclass(frozen=True, slots=True)
class FrozenDocumentCandidates:
    case_id: str
    first_pass_evidence_sha256: str
    planning_digest: str
    planner_output_digest: str
    paths: tuple[DocumentPathHypothesis, ...] = field(repr=False)
    retained_path_digest: str = ""
    planning_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.case_id or not self.paths:
            raise ValueError("frozen document candidates require case ID and paths")
        for digest in (
            self.first_pass_evidence_sha256,
            self.planning_digest,
            self.planner_output_digest,
            self.retained_path_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("frozen candidate digests must be SHA-256 values")
        latency = _strict_float(self.planning_latency_ms, name="planning_latency_ms")
        if latency < 0.0:
            raise ValueError("planning_latency_ms must be non-negative")
        if len({path.digest for path in self.paths}) != len(self.paths):
            raise ValueError("frozen candidate paths must be unique")
        if self.retained_path_digest not in {path.digest for path in self.paths}:
            raise ValueError("retained path is absent from frozen candidates")
        option_counts = {len(path.options) for path in self.paths}
        if len(option_counts) != 1:
            raise ValueError("all document paths must select one option per same window set")
        object.__setattr__(self, "planning_latency_ms", latency)

    @property
    def candidate_set_digest(self) -> str:
        return sha256_json(
            {
                "caseId": self.case_id,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "planningDigest": self.planning_digest,
                "plannerOutputDigest": self.planner_output_digest,
                "retainedPathDigest": self.retained_path_digest,
                "paths": [
                    {
                        "pathDigest": path.digest,
                        "baseScore": path.base_score,
                        "meanAudioSupport": path.mean_audio_support,
                        "optionDigests": [option.digest for option in path.options],
                    }
                    for path in self.paths
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedDocumentCase:
    case: DocumentExperimentCase = field(repr=False)
    candidates: FrozenDocumentCandidates = field(repr=False)

    def __post_init__(self) -> None:
        if self.case.case_id != self.candidates.case_id:
            raise ValueError("prepared case and frozen candidates have different case IDs")
        if self.case.first_pass.evidence_sha256 != self.candidates.first_pass_evidence_sha256:
            raise ValueError("frozen candidates belong to different first-pass evidence")
        if self.case.planning_digest != self.candidates.planning_digest:
            raise ValueError("frozen candidates belong to a different planning view")


@dataclass(frozen=True, slots=True)
class PathArmScore:
    path_digest: str
    base_score: float
    language_score: DocumentLanguageScore
    final_score: float

    def __post_init__(self) -> None:
        if not _is_sha256(self.path_digest):
            raise ValueError("path_digest must be a SHA-256 value")
        base = _strict_float(self.base_score, name="base_score")
        final = _strict_float(self.final_score, name="final_score")
        if self.language_score.path_digest != self.path_digest:
            raise ValueError("language score is bound to a different path")
        object.__setattr__(self, "base_score", base)
        object.__setattr__(self, "final_score", final)


@dataclass(frozen=True, slots=True)
class CaseArmResult:
    case_id: str
    arm_name: str
    arm_digest: str
    candidate_set_digest: str
    selected_path_digest: str
    retained_path_digest: str
    selected_text: str
    selected_window_texts: tuple[str, ...]
    accepted: bool
    margin: float
    path_scores: tuple[PathArmScore, ...]
    metrics: CaseArmMetrics
    planning_latency_ms: float
    error: str | None = None

    def __post_init__(self) -> None:
        for digest in (
            self.arm_digest,
            self.candidate_set_digest,
            self.selected_path_digest,
            self.retained_path_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("case-arm result contains an invalid SHA-256 value")
        margin = _strict_float(self.margin, name="margin")
        planning_latency = _strict_float(
            self.planning_latency_ms,
            name="planning_latency_ms",
        )
        if margin < 0.0 or planning_latency < 0.0:
            raise ValueError("case-arm margin and planning latency must be non-negative")
        if not self.selected_text or not self.selected_window_texts:
            raise ValueError("case-arm result requires selected document and window text")
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "planning_latency_ms", planning_latency)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "caseId": self.case_id,
                "armName": self.arm_name,
                "armDigest": self.arm_digest,
                "candidateSetDigest": self.candidate_set_digest,
                "selectedPathDigest": self.selected_path_digest,
                "retainedPathDigest": self.retained_path_digest,
                "selectedTextSha256": sha256_json({"text": self.selected_text}),
                "selectedWindowSha256": [
                    sha256_json({"text": row}) for row in self.selected_window_texts
                ],
                "accepted": self.accepted,
                "margin": self.margin,
                "pathScores": [
                    {
                        "pathDigest": row.path_digest,
                        "baseScore": row.base_score,
                        "languageScore": asdict(row.language_score),
                        "finalScore": row.final_score,
                    }
                    for row in self.path_scores
                ],
                "metrics": asdict(self.metrics),
                "planningLatencyMs": self.planning_latency_ms,
                "error": self.error,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentContextExperimentReport:
    protocol_digest: str
    manifest_digest: str
    case_results: tuple[CaseArmResult, ...]
    aggregates: tuple[ArmAggregateMetrics, ...]
    paired_intervals: tuple[PairedBootstrapInterval, ...]
    failures: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not _is_sha256(self.protocol_digest) or not _is_sha256(self.manifest_digest):
            raise ValueError("report protocol and manifest digests must be SHA-256 values")
        if not self.case_results:
            raise ValueError("experiment report requires case results")
        if len({(row.case_id, row.arm_name) for row in self.case_results}) != len(
            self.case_results
        ):
            raise ValueError("duplicate case/arm result")

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict(include_digest=False))

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schemaVersion": self.schema_version,
            "protocolDigest": self.protocol_digest,
            "manifestDigest": self.manifest_digest,
            "caseResults": [
                {
                    "caseId": row.case_id,
                    "armName": row.arm_name,
                    "armDigest": row.arm_digest,
                    "candidateSetDigest": row.candidate_set_digest,
                    "selectedPathDigest": row.selected_path_digest,
                    "retainedPathDigest": row.retained_path_digest,
                    "selectedText": row.selected_text,
                    "selectedWindowTexts": row.selected_window_texts,
                    "accepted": row.accepted,
                    "margin": row.margin,
                    "pathScores": [
                        {
                            "pathDigest": score.path_digest,
                            "baseScore": score.base_score,
                            "languageScore": asdict(score.language_score),
                            "finalScore": score.final_score,
                        }
                        for score in row.path_scores
                    ],
                    "metrics": {
                        "text": asdict(row.metrics.text),
                        "windows": asdict(row.metrics.windows),
                        "accepted": row.metrics.accepted,
                        "latencyMs": row.metrics.latency_ms,
                        "pythonPeakBytes": row.metrics.python_peak_bytes,
                        "scoredCharacters": row.metrics.scored_characters,
                        "scorerCalls": row.metrics.scorer_calls,
                    },
                    "planningLatencyMs": row.planning_latency_ms,
                    "error": row.error,
                    "digest": row.digest,
                }
                for row in self.case_results
            ],
            "aggregates": [
                {
                    **asdict(row),
                    "strictCer": row.strict_cer,
                    "lenientCer": row.lenient_cer,
                    "coverage": row.coverage,
                    "acceptedStrictCer": row.accepted_strict_cer,
                    "revisionRate": row.revision_rate,
                }
                for row in self.aggregates
            ],
            "pairedIntervals": [asdict(row) for row in self.paired_intervals],
            "failures": self.failures,
        }
        if include_digest:
            payload["reportDigest"] = self.digest
        return payload

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(self.as_dict(), ensure_ascii=False, indent=2))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return destination


def _planning_view(case: DocumentExperimentCase) -> PlanningCaseView:
    return PlanningCaseView(
        case_id=case.case_id,
        first_pass=case.first_pass,
        source_audio_sha256=case.first_pass.source_audio_sha256,
        planning_digest=case.planning_digest,
        external_contexts=tuple((row.name, row.context) for row in case.external_contexts),
    )


def _plan_digest(plan: object) -> str:
    digest = getattr(plan, "digest", None)
    if isinstance(digest, str) and _is_sha256(digest):
        return digest
    as_dict = getattr(plan, "as_dict", None)
    if callable(as_dict):
        return sha256_json(as_dict())
    raise TypeError("candidate planner output requires a stable digest or as_dict()")


def freeze_planner_output(
    case: DocumentExperimentCase,
    plan: object,
    *,
    planning_latency_ms: float,
    maximum_candidate_documents: int,
) -> FrozenDocumentCandidates:
    decision = getattr(plan, "decision", None)
    if decision is None:
        raise TypeError("candidate planner output must expose decision")
    alternatives = tuple(getattr(decision, "alternatives", ()))
    retained = getattr(decision, "retained", None)
    if retained is None or not alternatives:
        raise ValueError("candidate planner output requires retained and alternative paths")
    by_digest = {path.digest: path for path in alternatives}
    by_digest[retained.digest] = retained
    ordered = sorted(
        by_digest.values(),
        key=lambda path: (
            path.digest != retained.digest,
            -path.base_score,
            path.digest,
        ),
    )
    if len(ordered) > maximum_candidate_documents:
        retained_row = next(path for path in ordered if path.digest == retained.digest)
        others = [path for path in ordered if path.digest != retained.digest]
        ordered = [retained_row, *others[: maximum_candidate_documents - 1]]
    return FrozenDocumentCandidates(
        case_id=case.case_id,
        first_pass_evidence_sha256=case.first_pass.evidence_sha256,
        planning_digest=case.planning_digest,
        planner_output_digest=_plan_digest(plan),
        paths=tuple(ordered),
        retained_path_digest=retained.digest,
        planning_latency_ms=planning_latency_ms,
    )


def prepare_document_experiment(
    manifest: DocumentExperimentManifest,
    protocol: DocumentExperimentProtocol,
    planner: CandidatePlanner,
) -> tuple[PreparedDocumentCase, ...]:
    """Run reference-free candidate planning before any reference metric is computed."""

    prepared: list[PreparedDocumentCase] = []
    for case in manifest.cases:
        view = _planning_view(case)
        started = time.perf_counter_ns()
        plan = planner(view)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        prepared.append(
            PreparedDocumentCase(
                case=case,
                candidates=freeze_planner_output(
                    case,
                    plan,
                    planning_latency_ms=elapsed,
                    maximum_candidate_documents=protocol.maximum_candidate_documents,
                ),
            )
        )
    return tuple(prepared)


def _path_scores(
    prepared: PreparedDocumentCase,
    arm: DocumentExperimentArm,
    scorers: Mapping[str, DocumentArmScorer],
    protocol: DocumentExperimentProtocol,
) -> tuple[tuple[PathArmScore, ...], float, int, int, int]:
    paths = prepared.candidates.paths
    direction_multiplier = 2 if arm.direction == "bidirectional" else 1
    per_path_budget = max(
        1,
        protocol.maximum_scored_characters // (len(paths) * direction_multiplier),
    )
    context = prepared.case.context(arm.external_context_name)
    started = time.perf_counter_ns()
    tracemalloc.start()
    try:
        rows: list[PathArmScore] = []
        scored_characters = 0
        scorer_calls = 0
        for path in paths:
            if arm.scorer_key is None:
                language = DocumentLanguageScore.neutral(
                    path_digest=path.digest,
                    arm_digest=arm.digest,
                )
            else:
                try:
                    scorer = scorers[arm.scorer_key]
                except KeyError as exc:
                    raise ValueError(
                        f"arm {arm.name!r} references unknown scorer {arm.scorer_key!r}"
                    ) from exc
                language = scorer.score_path(
                    path,
                    arm,
                    case_id=prepared.case.case_id,
                    left_context=context.left_context,
                    right_context=context.right_context,
                    maximum_characters=per_path_budget,
                )
            if language.arm_digest != arm.digest:
                raise ValueError("document scorer returned a result for a different arm")
            scored_characters += language.scored_characters
            scorer_calls += language.scorer_calls
            rows.append(
                PathArmScore(
                    path_digest=path.digest,
                    base_score=path.base_score,
                    language_score=language,
                    final_score=path.base_score + arm.linguistic_weight * language.value,
                )
            )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
    if scored_characters > protocol.maximum_scored_characters:
        raise ValueError("arm exceeded the preregistered scored-character budget")
    rows.sort(key=lambda row: (-row.final_score, row.path_digest))
    return tuple(rows), elapsed, peak, scored_characters, scorer_calls


def _find_path(
    prepared: PreparedDocumentCase,
    digest: str,
) -> DocumentPathHypothesis:
    for path in prepared.candidates.paths:
        if path.digest == digest:
            return path
    raise KeyError(digest)


def _run_case_arm(
    prepared: PreparedDocumentCase,
    arm: DocumentExperimentArm,
    scorers: Mapping[str, DocumentArmScorer],
    protocol: DocumentExperimentProtocol,
) -> CaseArmResult:
    path_scores, latency_ms, peak, scored_characters, scorer_calls = _path_scores(
        prepared,
        arm,
        scorers,
        protocol,
    )
    selected_score = path_scores[0]
    selected = _find_path(prepared, selected_score.path_digest)
    margin = (
        selected_score.final_score - path_scores[1].final_score if len(path_scores) > 1 else 0.0
    )
    generated = int(getattr(selected, "generated_window_count", 0)) > 0
    accepted = not generated and (len(path_scores) == 1 or margin >= arm.minimum_margin)
    reference = prepared.case.reference
    metrics = CaseArmMetrics(
        text=text_error_metrics(selected.text, reference),
        windows=window_revision_metrics(
            tuple(segment.observed.text for segment in prepared.case.first_pass.segments),
            selected,
            reference.window_texts,
        ),
        accepted=accepted,
        latency_ms=latency_ms,
        python_peak_bytes=peak,
        scored_characters=scored_characters,
        scorer_calls=scorer_calls,
    )
    return CaseArmResult(
        case_id=prepared.case.case_id,
        arm_name=arm.name,
        arm_digest=arm.digest,
        candidate_set_digest=prepared.candidates.candidate_set_digest,
        selected_path_digest=selected.digest,
        retained_path_digest=prepared.candidates.retained_path_digest,
        selected_text=selected.text,
        selected_window_texts=tuple(option.text for option in selected.options),
        accepted=accepted,
        margin=margin,
        path_scores=path_scores,
        metrics=metrics,
        planning_latency_ms=prepared.candidates.planning_latency_ms,
    )


def run_document_context_experiment(
    prepared_cases: Sequence[PreparedDocumentCase],
    manifest: DocumentExperimentManifest,
    protocol: DocumentExperimentProtocol,
    *,
    scorers: Mapping[str, DocumentArmScorer],
) -> DocumentContextExperimentReport:
    """Score the exact same frozen candidate documents under every registered arm."""

    if not prepared_cases:
        raise ValueError("experiment requires prepared cases")
    by_case = {row.case.case_id: row for row in prepared_cases}
    expected = {case.case_id for case in manifest.cases}
    if set(by_case) != expected:
        raise ValueError("prepared cases do not match the frozen experiment manifest")
    results: list[CaseArmResult] = []
    failures: list[tuple[str, str, str]] = []
    for case in manifest.cases:
        prepared = by_case[case.case_id]
        for arm in protocol.arms:
            try:
                results.append(_run_case_arm(prepared, arm, scorers, protocol))
            except Exception as exc:
                if protocol.fail_on_case_error:
                    raise
                failures.append((case.case_id, arm.name, f"{type(exc).__name__}:{exc}"))
    grouped = {
        arm.name: tuple(row for row in results if row.arm_name == arm.name) for arm in protocol.arms
    }
    if any(not rows for rows in grouped.values()):
        raise ValueError("every experiment arm must produce at least one case result")
    aggregates = tuple(aggregate_arm_metrics(arm.name, grouped[arm.name]) for arm in protocol.arms)
    baseline = grouped[protocol.baseline_arm]
    intervals = tuple(
        paired_bootstrap_cer_delta(
            arm.name,
            protocol.baseline_arm,
            grouped[arm.name],
            baseline,
            resamples=protocol.bootstrap_resamples,
            seed=protocol.bootstrap_seed,
        )
        for arm in protocol.arms
        if arm.name != protocol.baseline_arm and len(grouped[arm.name]) == len(baseline)
    )
    candidate_set_by_case: dict[str, set[str]] = {}
    for row in results:
        candidate_set_by_case.setdefault(row.case_id, set()).add(row.candidate_set_digest)
    if any(len(values) != 1 for values in candidate_set_by_case.values()):
        raise ValueError("experiment arms did not share one frozen candidate set per case")
    return DocumentContextExperimentReport(
        protocol_digest=protocol.digest,
        manifest_digest=manifest.digest,
        case_results=tuple(results),
        aggregates=aggregates,
        paired_intervals=intervals,
        failures=tuple(failures),
    )
