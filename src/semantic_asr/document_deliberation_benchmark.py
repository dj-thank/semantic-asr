"""Paired evaluation for document-level Semantic ASR deliberation.

The benchmark separates useful context corrections from false corrections and overlap cleanup. CER
alone can hide a system that improves long sentences while corrupting already-correct numbers,
negation, names, or disfluencies. Promotion gates therefore operate on paired per-document outcomes
and explicitly fail when coverage or sample size is insufficient.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .contracts import sha256_json


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    return _edit_distance(reference, hypothesis) / max(1, len(reference))


def _token_error(reference: str, hypothesis: str, token: str) -> bool:
    return reference.count(token) != hypothesis.count(token)


def _boundary_duplicate_count(segments: Sequence[str], *, minimum: int = 4) -> int:
    count = 0
    for left, right in zip(segments, segments[1:], strict=False):
        limit = min(len(left), len(right), 80)
        for width in range(limit, minimum - 1, -1):
            if left[-width:] == right[:width]:
                count += 1
                break
    return count


@dataclass(frozen=True, slots=True)
class DocumentEvaluationCase:
    case_id: str
    reference: str
    first_pass: str
    final: str
    final_status: str
    first_pass_segments: tuple[str, ...] = ()
    final_segments: tuple[str, ...] = ()
    critical_tokens: tuple[str, ...] = ()
    changed_window_count: int = 0
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.reference:
            raise ValueError("evaluation case requires case_id and reference")
        if self.final_status not in {"accepted", "provisional", "first-pass"}:
            raise ValueError("final_status must be accepted, provisional, or first-pass")
        if self.changed_window_count < 0:
            raise ValueError("changed_window_count must be non-negative")
        if any(not token for token in self.critical_tokens):
            raise ValueError("critical tokens must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def first_cer(self) -> float:
        return character_error_rate(self.reference, self.first_pass)

    @property
    def final_cer(self) -> float:
        return character_error_rate(self.reference, self.final)

    @property
    def improved(self) -> bool:
        return self.final_cer < self.first_cer

    @property
    def regressed(self) -> bool:
        return self.final_cer > self.first_cer

    @property
    def changed(self) -> bool:
        return self.final != self.first_pass

    @property
    def first_exact(self) -> bool:
        return self.first_pass == self.reference

    @property
    def false_correction(self) -> bool:
        return self.first_exact and self.final != self.reference

    @property
    def critical_first_errors(self) -> int:
        return sum(
            _token_error(self.reference, self.first_pass, token)
            for token in self.critical_tokens
        )

    @property
    def critical_final_errors(self) -> int:
        return sum(
            _token_error(self.reference, self.final, token)
            for token in self.critical_tokens
        )

    @property
    def first_boundary_duplicates(self) -> int:
        return _boundary_duplicate_count(self.first_pass_segments)

    @property
    def final_boundary_duplicates(self) -> int:
        return _boundary_duplicate_count(self.final_segments)


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int

    def __post_init__(self) -> None:
        for value in (self.estimate, self.lower, self.upper, self.confidence):
            if not math.isfinite(float(value)):
                raise ValueError("bootstrap interval values must be finite")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("bootstrap confidence must be in (0, 1)")
        if self.samples < 1:
            raise ValueError("bootstrap samples must be positive")


@dataclass(frozen=True, slots=True)
class DocumentBenchmarkReport:
    case_count: int
    reference_character_count: int
    first_corpus_cer: float
    final_corpus_cer: float
    corpus_cer_delta: float
    mean_case_cer_delta: float
    improved_case_rate: float
    regressed_case_rate: float
    changed_case_rate: float
    accepted_coverage: float
    accepted_error_rate: float
    false_correction_rate_on_first_exact: float | None
    first_exact_case_count: int
    critical_token_count: int
    critical_first_error_rate: float | None
    critical_final_error_rate: float | None
    critical_error_delta: float | None
    first_boundary_duplicates: int
    final_boundary_duplicates: int
    overlap_duplicate_reduction: int
    mean_changed_windows: float
    cer_delta_interval: BootstrapInterval
    case_digests: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentPromotionGate:
    minimum_cases: int = 100
    minimum_reference_characters: int = 10_000
    minimum_accepted_coverage: float = 0.20
    maximum_false_correction_rate: float = 0.005
    maximum_regressed_case_rate: float = 0.10
    maximum_critical_error_delta: float = 0.0
    require_cer_delta_upper_below: float = 0.0
    require_overlap_non_regression: bool = True

    def __post_init__(self) -> None:
        if self.minimum_cases < 1 or self.minimum_reference_characters < 1:
            raise ValueError("promotion minimum sample sizes must be positive")
        for name in (
            "minimum_accepted_coverage",
            "maximum_false_correction_rate",
            "maximum_regressed_case_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(float(self.maximum_critical_error_delta)):
            raise ValueError("maximum_critical_error_delta must be finite")
        if not math.isfinite(float(self.require_cer_delta_upper_below)):
            raise ValueError("require_cer_delta_upper_below must be finite")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    report_digest: str
    gate_digest: str

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


def _corpus_cer(cases: Sequence[DocumentEvaluationCase], *, final: bool) -> float:
    errors = sum(
        _edit_distance(case.reference, case.final if final else case.first_pass)
        for case in cases
    )
    characters = sum(len(case.reference) for case in cases)
    return errors / max(1, characters)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_cer_delta(
    cases: Sequence[DocumentEvaluationCase],
    *,
    samples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    if not cases:
        raise ValueError("paired bootstrap requires at least one case")
    if samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap configuration")
    randomizer = random.Random(seed)
    values = []
    for _ in range(samples):
        resampled = [cases[randomizer.randrange(len(cases))] for _ in range(len(cases))]
        values.append(_corpus_cer(resampled, final=True) - _corpus_cer(resampled, final=False))
    tail = (1.0 - confidence) / 2.0
    estimate = _corpus_cer(cases, final=True) - _corpus_cer(cases, final=False)
    return BootstrapInterval(
        estimate=estimate,
        lower=_quantile(values, tail),
        upper=_quantile(values, 1.0 - tail),
        confidence=confidence,
        samples=samples,
        seed=seed,
    )


def evaluate_document_deliberation(
    cases: Sequence[DocumentEvaluationCase],
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> DocumentBenchmarkReport:
    if not cases:
        raise ValueError("document benchmark requires at least one case")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("document evaluation case IDs must be unique")
    first_cer = _corpus_cer(cases, final=False)
    final_cer = _corpus_cer(cases, final=True)
    accepted = [case for case in cases if case.final_status == "accepted"]
    first_exact = [case for case in cases if case.first_exact]
    critical_count = sum(len(case.critical_tokens) for case in cases)
    critical_first = sum(case.critical_first_errors for case in cases)
    critical_final = sum(case.critical_final_errors for case in cases)
    interval = paired_bootstrap_cer_delta(
        cases,
        samples=bootstrap_samples,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    first_duplicates = sum(case.first_boundary_duplicates for case in cases)
    final_duplicates = sum(case.final_boundary_duplicates for case in cases)
    return DocumentBenchmarkReport(
        case_count=len(cases),
        reference_character_count=sum(len(case.reference) for case in cases),
        first_corpus_cer=first_cer,
        final_corpus_cer=final_cer,
        corpus_cer_delta=final_cer - first_cer,
        mean_case_cer_delta=sum(case.final_cer - case.first_cer for case in cases) / len(cases),
        improved_case_rate=sum(case.improved for case in cases) / len(cases),
        regressed_case_rate=sum(case.regressed for case in cases) / len(cases),
        changed_case_rate=sum(case.changed for case in cases) / len(cases),
        accepted_coverage=len(accepted) / len(cases),
        accepted_error_rate=(
            sum(case.final != case.reference for case in accepted) / len(accepted)
            if accepted
            else 0.0
        ),
        false_correction_rate_on_first_exact=(
            sum(case.false_correction for case in first_exact) / len(first_exact)
            if first_exact
            else None
        ),
        first_exact_case_count=len(first_exact),
        critical_token_count=critical_count,
        critical_first_error_rate=(critical_first / critical_count if critical_count else None),
        critical_final_error_rate=(critical_final / critical_count if critical_count else None),
        critical_error_delta=(
            (critical_final - critical_first) / critical_count if critical_count else None
        ),
        first_boundary_duplicates=first_duplicates,
        final_boundary_duplicates=final_duplicates,
        overlap_duplicate_reduction=first_duplicates - final_duplicates,
        mean_changed_windows=(
            sum(case.changed_window_count for case in cases) / len(cases)
        ),
        cer_delta_interval=interval,
        case_digests=tuple(
            sha256_json(
                {
                    "caseId": case.case_id,
                    "reference": case.reference,
                    "firstPass": case.first_pass,
                    "final": case.final,
                    "finalStatus": case.final_status,
                    "criticalTokens": case.critical_tokens,
                    "changedWindowCount": case.changed_window_count,
                    "metadata": case.metadata,
                }
            )
            for case in cases
        ),
    )


def apply_document_promotion_gate(
    report: DocumentBenchmarkReport,
    gate: DocumentPromotionGate,
) -> PromotionDecision:
    reasons = []
    if report.case_count < gate.minimum_cases:
        reasons.append("insufficient-case-count")
    if report.reference_character_count < gate.minimum_reference_characters:
        reasons.append("insufficient-reference-characters")
    if report.accepted_coverage < gate.minimum_accepted_coverage:
        reasons.append("insufficient-accepted-coverage")
    if report.first_exact_case_count == 0:
        reasons.append("no-first-pass-exact-arm")
    elif (
        report.false_correction_rate_on_first_exact is not None
        and report.false_correction_rate_on_first_exact > gate.maximum_false_correction_rate
    ):
        reasons.append("false-correction-rate-regression")
    if report.regressed_case_rate > gate.maximum_regressed_case_rate:
        reasons.append("case-regression-rate-too-high")
    if report.critical_token_count == 0:
        reasons.append("no-critical-token-arm")
    elif (
        report.critical_error_delta is not None
        and report.critical_error_delta > gate.maximum_critical_error_delta
    ):
        reasons.append("critical-token-regression")
    if report.cer_delta_interval.upper >= gate.require_cer_delta_upper_below:
        reasons.append("paired-cer-improvement-not-demonstrated")
    if gate.require_overlap_non_regression and report.overlap_duplicate_reduction < 0:
        reasons.append("overlap-duplication-regression")
    return PromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        report_digest=report.digest,
        gate_digest=gate.digest,
    )
