"""Finite-sample risk control for adaptive ASR hypothesis-set sizing.

The v0.2 adaptive selector estimates how many candidates are useful from the
current candidate distribution. This module adds a conservative calibration
layer that chooses a K only when a group-level Hoeffding upper confidence bound
meets the declared risk target. It is a dependency-free Learn-Then-Test
baseline, not a replacement for richer conformal or CRC experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

CalibrationSplit = Literal["calibration"]


@dataclass(frozen=True, slots=True)
class RiskObservation:
    """One bounded-loss observation for a candidate-set size."""

    sample_id: str
    group_id: str
    k: int
    loss: float
    cost_ms: float
    split: CalibrationSplit = "calibration"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise ValueError("risk observation sample and group IDs are required")
        if self.k < 1:
            raise ValueError("risk observation K must be positive")
        if not math.isfinite(float(self.loss)) or not 0.0 <= float(self.loss) <= 1.0:
            raise ValueError("risk observation loss must be finite and in [0, 1]")
        if not math.isfinite(float(self.cost_ms)) or float(self.cost_ms) < 0:
            raise ValueError("risk observation cost must be finite and non-negative")
        if self.split != "calibration":
            raise ValueError("risk-control fitting may consume only the calibration split")


@dataclass(frozen=True, slots=True)
class LearnThenTestConfig:
    target_risk: float = 0.10
    delta: float = 0.05
    minimum_groups: int = 30

    def __post_init__(self) -> None:
        if not 0.0 < self.target_risk < 1.0:
            raise ValueError("target_risk must be in (0, 1)")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.minimum_groups < 1:
            raise ValueError("minimum_groups must be positive")


@dataclass(frozen=True, slots=True)
class KRiskEstimate:
    k: int
    sample_count: int
    group_count: int
    empirical_risk: float
    upper_confidence_bound: float
    mean_cost_ms: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class LearnThenTestResult:
    selected_k: int | None
    estimates: tuple[KRiskEstimate, ...]
    target_risk: float
    delta: float
    confidence_statement: str
    reason: str


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    name: str
    quality: float
    latency_ms: float
    memory_mb: float = 0.0
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        values = (self.quality, self.latency_ms, self.memory_mb)
        if not self.name or any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Pareto points require a name and finite values")
        if self.latency_ms < 0 or self.memory_mb < 0:
            raise ValueError("latency and memory must be non-negative")


def _group_losses(rows: Sequence[RiskObservation]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.group_id].append(float(row.loss))
    return [fmean(values) for _, values in sorted(grouped.items())]


def learn_then_test_k(
    observations: Sequence[RiskObservation],
    *,
    config: LearnThenTestConfig | None = None,
) -> LearnThenTestResult:
    """Choose the least-cost K whose finite-sample risk bound is acceptable.

    Repeated segments from the same ``group_id`` are averaged before computing
    the bound so they do not artificially increase the independent sample count.
    A Bonferroni correction controls the family-wise failure probability across
    every K evaluated in the same call.
    """

    config = config or LearnThenTestConfig()
    if not observations:
        return LearnThenTestResult(
            selected_k=None,
            estimates=(),
            target_risk=config.target_risk,
            delta=config.delta,
            confidence_statement="No finite-sample statement: no observations.",
            reason="no-observations",
        )
    if any(row.split != "calibration" for row in observations):
        raise ValueError("training and test observations are forbidden during risk fitting")

    by_k: dict[int, list[RiskObservation]] = defaultdict(list)
    for observation in observations:
        by_k[observation.k].append(observation)
    per_test_delta = config.delta / max(1, len(by_k))
    estimates: list[KRiskEstimate] = []
    for k in sorted(by_k):
        rows = by_k[k]
        losses = _group_losses(rows)
        empirical = fmean(losses)
        if len(losses) >= config.minimum_groups:
            radius = math.sqrt(math.log(1.0 / per_test_delta) / (2.0 * len(losses)))
            upper = min(1.0, empirical + radius)
        else:
            upper = 1.0
        estimates.append(
            KRiskEstimate(
                k=k,
                sample_count=len(rows),
                group_count=len(losses),
                empirical_risk=empirical,
                upper_confidence_bound=upper,
                mean_cost_ms=fmean(float(row.cost_ms) for row in rows),
                accepted=(
                    len(losses) >= config.minimum_groups
                    and upper <= config.target_risk
                ),
            )
        )

    accepted = [estimate for estimate in estimates if estimate.accepted]
    selected = min(accepted, key=lambda row: (row.mean_cost_ms, row.k)) if accepted else None
    statement = (
        "Under bounded-loss, independent-group, and fixed-candidate-family assumptions, "
        f"the family-wise failure probability is at most {config.delta:.6g}."
    )
    return LearnThenTestResult(
        selected_k=None if selected is None else selected.k,
        estimates=tuple(estimates),
        target_risk=config.target_risk,
        delta=config.delta,
        confidence_statement=statement,
        reason=(
            "finite-sample-risk-bound"
            if selected is not None
            else "no-k-meets-finite-sample-risk-bound"
        ),
    )


def pareto_frontier(points: Sequence[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    """Return quality-maximizing, latency/memory-minimizing non-dominated points."""

    frontier: list[ParetoPoint] = []
    for point in points:
        dominated = any(
            other is not point
            and other.quality >= point.quality
            and other.latency_ms <= point.latency_ms
            and other.memory_mb <= point.memory_mb
            and (
                other.quality > point.quality
                or other.latency_ms < point.latency_ms
                or other.memory_mb < point.memory_mb
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return tuple(
        sorted(
            frontier,
            key=lambda row: (row.latency_ms, row.memory_mb, -row.quality, row.name),
        )
    )


def risk_observation_from_row(
    row: Mapping[str, Any], *, line_number: int
) -> RiskObservation:
    split = str(row.get("split") or "calibration")
    if split != "calibration":
        raise ValueError(
            f"risk-control row {line_number} belongs to forbidden split {split!r}"
        )
    return RiskObservation(
        sample_id=str(row.get("sampleId") or row.get("sample_id") or line_number),
        group_id=str(row.get("groupId") or row.get("group_id") or ""),
        k=int(row["k"]),
        loss=float(row["loss"]),
        cost_ms=float(row.get("costMs") or row.get("cost_ms") or 0.0),
    )


def load_risk_observations(path: str | Path) -> list[RiskObservation]:
    output: list[RiskObservation] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"risk-control row {line_number} must be an object")
        output.append(risk_observation_from_row(payload, line_number=line_number))
    if not output:
        raise ValueError("risk-control dataset is empty")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output")
    parser.add_argument("--target-risk", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--minimum-groups", type=int, default=30)
    args = parser.parse_args(argv)
    result = learn_then_test_k(
        load_risk_observations(args.input),
        config=LearnThenTestConfig(
            target_risk=args.target_risk,
            delta=args.delta,
            minimum_groups=args.minimum_groups,
        ),
    )
    payload = asdict(result)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
