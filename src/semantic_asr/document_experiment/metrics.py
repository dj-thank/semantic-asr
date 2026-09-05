"""Reference-side metrics for preregistered document-context experiments."""

from __future__ import annotations

import math
import random
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from ..deliberation_evidence import _strict_float
from ..document_deliberation import DocumentPathHypothesis
from .protocol import CriticalReferenceToken, FrozenReference


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def lenient_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S", "Z"))
    )


def _token_count(text: str, token: CriticalReferenceToken) -> int:
    values = (token.text, *token.aliases)
    return max(text.count(value) for value in values)


def critical_token_errors(
    hypothesis: str,
    tokens: Sequence[CriticalReferenceToken],
) -> tuple[int, tuple[tuple[str, str, int, int], ...]]:
    rows: list[tuple[str, str, int, int]] = []
    total = 0
    for token in tokens:
        observed = _token_count(hypothesis, token)
        error = abs(token.count - observed)
        total += error
        rows.append((token.kind, token.text, token.count, observed))
    return total, tuple(rows)


@dataclass(frozen=True, slots=True)
class TextErrorMetrics:
    reference_characters: int
    strict_edits: int
    lenient_reference_characters: int
    lenient_edits: int
    critical_token_errors: int
    critical_token_counts: tuple[tuple[str, str, int, int], ...]

    def __post_init__(self) -> None:
        for name in (
            "reference_characters",
            "strict_edits",
            "lenient_reference_characters",
            "lenient_edits",
            "critical_token_errors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.reference_characters < 1 or self.lenient_reference_characters < 1:
            raise ValueError("reference surfaces must contain at least one scored character")

    @property
    def strict_cer(self) -> float:
        return self.strict_edits / self.reference_characters

    @property
    def lenient_cer(self) -> float:
        return self.lenient_edits / self.lenient_reference_characters


def text_error_metrics(
    hypothesis: str,
    reference: FrozenReference,
) -> TextErrorMetrics:
    strict_reference = tuple(reference.text)
    strict_hypothesis = tuple(hypothesis)
    lenient_reference = tuple(lenient_surface(reference.text))
    lenient_hypothesis = tuple(lenient_surface(hypothesis))
    if not lenient_reference:
        raise ValueError("lenient reference surface is empty")
    critical_errors, counts = critical_token_errors(hypothesis, reference.critical_tokens)
    return TextErrorMetrics(
        reference_characters=len(strict_reference),
        strict_edits=edit_distance(strict_reference, strict_hypothesis),
        lenient_reference_characters=len(lenient_reference),
        lenient_edits=edit_distance(lenient_reference, lenient_hypothesis),
        critical_token_errors=critical_errors,
        critical_token_counts=counts,
    )


@dataclass(frozen=True, slots=True)
class WindowRevisionMetrics:
    window_count: int
    changed_windows: int
    improved_windows: int
    worsened_windows: int
    unchanged_error_windows: int
    false_correction_windows: int
    corrected_characters: int
    introduced_error_characters: int

    def __post_init__(self) -> None:
        for name in asdict(self):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.changed_windows > self.window_count:
            raise ValueError("changed_windows cannot exceed window_count")

    @property
    def revision_rate(self) -> float:
        return self.changed_windows / self.window_count if self.window_count else 0.0


def window_revision_metrics(
    first_pass_windows: Sequence[str],
    selected_path: DocumentPathHypothesis,
    reference_windows: Sequence[str],
) -> WindowRevisionMetrics:
    selected_windows = tuple(option.text for option in selected_path.options)
    if not (len(first_pass_windows) == len(selected_windows) == len(reference_windows)):
        raise ValueError("first-pass, selected, and reference window counts must match")
    changed = 0
    improved = 0
    worsened = 0
    unchanged_error = 0
    false_corrections = 0
    corrected_characters = 0
    introduced_characters = 0
    for baseline, selected, reference in zip(
        first_pass_windows,
        selected_windows,
        reference_windows,
        strict=True,
    ):
        baseline_edits = edit_distance(tuple(reference), tuple(baseline))
        selected_edits = edit_distance(tuple(reference), tuple(selected))
        if baseline != selected:
            changed += 1
        if selected_edits < baseline_edits:
            improved += 1
            corrected_characters += baseline_edits - selected_edits
        elif selected_edits > baseline_edits:
            worsened += 1
            introduced_characters += selected_edits - baseline_edits
        else:
            unchanged_error += int(baseline_edits > 0)
        if baseline_edits == 0 and selected_edits > 0:
            false_corrections += 1
    return WindowRevisionMetrics(
        window_count=len(reference_windows),
        changed_windows=changed,
        improved_windows=improved,
        worsened_windows=worsened,
        unchanged_error_windows=unchanged_error,
        false_correction_windows=false_corrections,
        corrected_characters=corrected_characters,
        introduced_error_characters=introduced_characters,
    )


@dataclass(frozen=True, slots=True)
class CaseArmMetrics:
    text: TextErrorMetrics
    windows: WindowRevisionMetrics
    accepted: bool
    latency_ms: float
    python_peak_bytes: int
    scored_characters: int
    scorer_calls: int

    def __post_init__(self) -> None:
        latency = _strict_float(self.latency_ms, name="latency_ms")
        if latency < 0.0:
            raise ValueError("latency_ms must be non-negative")
        for name in ("python_peak_bytes", "scored_characters", "scorer_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean")
        object.__setattr__(self, "latency_ms", latency)


@dataclass(frozen=True, slots=True)
class ArmAggregateMetrics:
    arm_name: str
    case_count: int
    accepted_case_count: int
    strict_reference_characters: int
    strict_edits: int
    lenient_reference_characters: int
    lenient_edits: int
    accepted_reference_characters: int
    accepted_strict_edits: int
    critical_token_errors: int
    changed_windows: int
    total_windows: int
    improved_windows: int
    worsened_windows: int
    false_correction_windows: int
    corrected_characters: int
    introduced_error_characters: int
    total_latency_ms: float
    maximum_case_latency_ms: float
    maximum_python_peak_bytes: int
    scored_characters: int
    scorer_calls: int

    @property
    def strict_cer(self) -> float:
        return self.strict_edits / self.strict_reference_characters

    @property
    def lenient_cer(self) -> float:
        return self.lenient_edits / self.lenient_reference_characters

    @property
    def coverage(self) -> float:
        return self.accepted_case_count / self.case_count

    @property
    def accepted_strict_cer(self) -> float | None:
        if self.accepted_reference_characters == 0:
            return None
        return self.accepted_strict_edits / self.accepted_reference_characters

    @property
    def revision_rate(self) -> float:
        return self.changed_windows / self.total_windows if self.total_windows else 0.0


def aggregate_arm_metrics(
    arm_name: str,
    rows: Sequence[CaseArmMetrics],
) -> ArmAggregateMetrics:
    if not rows:
        raise ValueError("cannot aggregate an empty arm result")
    accepted = [row for row in rows if row.accepted]
    return ArmAggregateMetrics(
        arm_name=arm_name,
        case_count=len(rows),
        accepted_case_count=len(accepted),
        strict_reference_characters=sum(row.text.reference_characters for row in rows),
        strict_edits=sum(row.text.strict_edits for row in rows),
        lenient_reference_characters=sum(row.text.lenient_reference_characters for row in rows),
        lenient_edits=sum(row.text.lenient_edits for row in rows),
        accepted_reference_characters=sum(row.text.reference_characters for row in accepted),
        accepted_strict_edits=sum(row.text.strict_edits for row in accepted),
        critical_token_errors=sum(row.text.critical_token_errors for row in rows),
        changed_windows=sum(row.windows.changed_windows for row in rows),
        total_windows=sum(row.windows.window_count for row in rows),
        improved_windows=sum(row.windows.improved_windows for row in rows),
        worsened_windows=sum(row.windows.worsened_windows for row in rows),
        false_correction_windows=sum(row.windows.false_correction_windows for row in rows),
        corrected_characters=sum(row.windows.corrected_characters for row in rows),
        introduced_error_characters=sum(row.windows.introduced_error_characters for row in rows),
        total_latency_ms=sum(row.latency_ms for row in rows),
        maximum_case_latency_ms=max(row.latency_ms for row in rows),
        maximum_python_peak_bytes=max(row.python_peak_bytes for row in rows),
        scored_characters=sum(row.scored_characters for row in rows),
        scorer_calls=sum(row.scorer_calls for row in rows),
    )


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    metric: str
    arm_name: str
    baseline_arm: str
    case_count: int
    point_delta: float
    lower: float
    upper: float
    resamples: int
    seed: int

    def __post_init__(self) -> None:
        for name in ("point_delta", "lower", "upper"):
            object.__setattr__(self, name, _strict_float(getattr(self, name), name=name))
        if self.lower > self.upper:
            raise ValueError("bootstrap lower bound exceeds upper bound")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_cer_delta(
    arm_name: str,
    baseline_arm: str,
    arm_rows: Sequence[CaseArmMetrics],
    baseline_rows: Sequence[CaseArmMetrics],
    *,
    resamples: int,
    seed: int,
) -> PairedBootstrapInterval:
    if len(arm_rows) != len(baseline_rows) or not arm_rows:
        raise ValueError("paired bootstrap requires equal non-empty case sequences")
    if isinstance(resamples, bool) or resamples < 200:
        raise ValueError("paired bootstrap requires at least 200 resamples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("bootstrap seed must be an integer")
    deltas = [
        arm.text.strict_cer - baseline.text.strict_cer
        for arm, baseline in zip(arm_rows, baseline_rows, strict=True)
    ]
    point = sum(deltas) / len(deltas)
    random_source = random.Random(seed)
    samples = []
    for _ in range(resamples):
        sampled = [deltas[random_source.randrange(len(deltas))] for _ in deltas]
        samples.append(sum(sampled) / len(sampled))
    return PairedBootstrapInterval(
        metric="mean-case-strict-cer-delta",
        arm_name=arm_name,
        baseline_arm=baseline_arm,
        case_count=len(deltas),
        point_delta=point,
        lower=_quantile(samples, 0.025),
        upper=_quantile(samples, 0.975),
        resamples=resamples,
        seed=seed,
    )


def critical_error_counter(
    rows: Sequence[CaseArmMetrics],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for kind, _text, expected, observed in row.text.critical_token_counts:
            counter[kind] += abs(expected - observed)
    return counter
