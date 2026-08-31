from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Self


@dataclass(frozen=True, slots=True)
class PlannerFeatureSchema:
    names: tuple[str, ...] = (
        "entropy",
        "disagreement",
        "missing_evidence",
        "posterior_margin_inverse",
        "semantic_criticality",
        "span_duration_seconds",
        "candidate_count_log",
        "cache_miss",
        "cpu_tier",
        "gpu_available",
    )
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("planner feature names must be unique and non-empty")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {"names": self.names, "version": self.version},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class PlannerFeatures:
    values: dict[str, float]
    schema_digest: str

    @classmethod
    def create(
        cls,
        schema: PlannerFeatureSchema,
        values: Mapping[str, float],
    ) -> Self:
        if set(values) != set(schema.names):
            missing = set(schema.names) - set(values)
            extra = set(values) - set(schema.names)
            raise ValueError(f"planner feature mismatch; missing={missing}, extra={extra}")
        converted = {name: float(values[name]) for name in schema.names}
        if any(not math.isfinite(value) for value in converted.values()):
            raise ValueError("planner features must be finite")
        return cls(values=converted, schema_digest=schema.digest)


@dataclass(frozen=True, slots=True)
class ActionObservation:
    action_kind: str
    features: PlannerFeatures
    loss_before: float
    loss_after: float
    measured_cost_ms: float
    sample_id: str
    platform_id: str

    def __post_init__(self) -> None:
        if not self.action_kind or not self.sample_id or not self.platform_id:
            raise ValueError("action_kind, sample_id and platform_id are required")
        for name, value in (
            ("loss_before", self.loss_before),
            ("loss_after", self.loss_after),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.measured_cost_ms) or self.measured_cost_ms < 0:
            raise ValueError("measured_cost_ms must be finite and non-negative")

    @property
    def realized_gain(self) -> float:
        return self.loss_before - self.loss_after


@dataclass(frozen=True, slots=True)
class LinearRegressor:
    names: tuple[str, ...]
    means: dict[str, float]
    scales: dict[str, float]
    weights: dict[str, float]
    intercept: float
    target_name: str

    def predict(self, features: PlannerFeatures) -> float:
        if set(features.values) != set(self.names):
            raise ValueError("regressor feature mismatch")
        return self.intercept + sum(
            self.weights[name] * ((features.values[name] - self.means[name]) / self.scales[name])
            for name in self.names
        )

    @classmethod
    def fit(
        cls,
        rows: Sequence[PlannerFeatures],
        targets: Sequence[float],
        *,
        schema: PlannerFeatureSchema,
        target_name: str,
        ridge: float = 1e-3,
    ) -> Self:
        if len(rows) != len(targets) or len(rows) < 2:
            raise ValueError("regressor requires matching rows and at least two samples")
        if ridge < 0 or not math.isfinite(ridge):
            raise ValueError("ridge must be finite and non-negative")
        if any(row.schema_digest != schema.digest for row in rows):
            raise ValueError("planner rows do not match schema")
        if any(not math.isfinite(float(target)) for target in targets):
            raise ValueError("regression targets must be finite")
        means = {name: fmean(row.values[name] for row in rows) for name in schema.names}
        scales: dict[str, float] = {}
        for name in schema.names:
            variance = fmean((row.values[name] - means[name]) ** 2 for row in rows)
            scales[name] = max(math.sqrt(variance), 1e-6)
        design = [
            [1.0] + [(row.values[name] - means[name]) / scales[name] for name in schema.names]
            for row in rows
        ]
        dimension = len(schema.names) + 1
        gram = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
        response = [0.0 for _ in range(dimension)]
        for vector, target in zip(design, targets, strict=True):
            for left in range(dimension):
                response[left] += vector[left] * float(target)
                for right in range(dimension):
                    gram[left][right] += vector[left] * vector[right]
        for index in range(1, dimension):
            gram[index][index] += ridge
        solution = _solve_linear_system(gram, response)
        return cls(
            names=schema.names,
            means=means,
            scales=scales,
            weights={name: solution[index + 1] for index, name in enumerate(schema.names)},
            intercept=solution[0],
            target_name=target_name,
        )


@dataclass(frozen=True, slots=True)
class ActionModel:
    action_kind: str
    gain_model: LinearRegressor | None
    log_cost_model: LinearRegressor | None
    fallback_gain: float
    fallback_cost_ms: float
    samples: int

    def predict(self, features: PlannerFeatures) -> tuple[float, float]:
        gain = (
            self.gain_model.predict(features) if self.gain_model is not None else self.fallback_gain
        )
        log_cost = (
            self.log_cost_model.predict(features)
            if self.log_cost_model is not None
            else math.log1p(self.fallback_cost_ms)
        )
        return min(1.0, max(-1.0, gain)), max(0.0, math.expm1(log_cost))


@dataclass(frozen=True, slots=True)
class LearnedPlannerModel:
    schema: PlannerFeatureSchema
    action_models: dict[str, ActionModel]
    training_digest: str
    platform_id: str

    def predict(self, action_kind: str, features: PlannerFeatures) -> tuple[float, float]:
        try:
            model = self.action_models[action_kind]
        except KeyError as exc:
            raise KeyError(f"unknown learned action: {action_kind}") from exc
        return model.predict(features)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    action_id: str
    action_kind: str
    features: PlannerFeatures
    dependencies: tuple[str, ...] = ()
    exclusive_group: str | None = None
    mandatory: bool = False
    affects_observed_decision: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id or not self.action_kind:
            raise ValueError("action_id and action_kind are required")
        if self.action_id in self.dependencies:
            raise ValueError("action cannot depend on itself")


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    action: ActionSpec
    expected_gain: float
    expected_cost_ms: float
    utility: float


@dataclass(frozen=True, slots=True)
class LearnedEvidencePlan:
    selected: tuple[ActionPrediction, ...]
    rejected: tuple[ActionPrediction, ...]
    used_cost_ms: float
    expected_gain: float
    stopping_reason: str
    model_digest: str


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if len(matrix) != len(vector) or any(len(row) != len(vector) for row in matrix):
        raise ValueError("linear system must be square")
    size = len(vector)
    augmented = [list(row) + [float(value)] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            # Ridge should normally avoid singularity. Add a tiny deterministic
            # diagonal fallback rather than emitting unstable coefficients.
            augmented[column][column] += 1e-8
            pivot = column
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [augmented[row][-1] for row in range(size)]


def _observation_digest(observations: Sequence[ActionObservation]) -> str:
    payload = [
        {
            "actionKind": row.action_kind,
            "features": row.features.values,
            "lossBefore": row.loss_before,
            "lossAfter": row.loss_after,
            "costMs": row.measured_cost_ms,
            "sampleId": row.sample_id,
            "platformId": row.platform_id,
        }
        for row in sorted(
            observations,
            key=lambda value: (value.sample_id, value.action_kind),
        )
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def fit_learned_planner(
    observations: Sequence[ActionObservation],
    *,
    schema: PlannerFeatureSchema | None = None,
    platform_id: str,
    minimum_samples_per_action: int = 12,
    ridge: float = 1e-2,
) -> LearnedPlannerModel:
    schema = schema or PlannerFeatureSchema()
    if not observations:
        raise ValueError("planner observations are required")
    if minimum_samples_per_action < 2:
        raise ValueError("minimum_samples_per_action must be at least two")
    filtered = [row for row in observations if row.platform_id == platform_id]
    if not filtered:
        raise ValueError("no observations match platform_id")
    if any(row.features.schema_digest != schema.digest for row in filtered):
        raise ValueError("observation features do not match schema")
    grouped: dict[str, list[ActionObservation]] = {}
    for row in filtered:
        grouped.setdefault(row.action_kind, []).append(row)
    action_models: dict[str, ActionModel] = {}
    for action_kind, rows in grouped.items():
        fallback_gain = fmean(row.realized_gain for row in rows)
        fallback_cost = fmean(row.measured_cost_ms for row in rows)
        gain_model = None
        log_cost_model = None
        if len(rows) >= minimum_samples_per_action:
            gain_model = LinearRegressor.fit(
                [row.features for row in rows],
                [row.realized_gain for row in rows],
                schema=schema,
                target_name="loss-reduction",
                ridge=ridge,
            )
            log_cost_model = LinearRegressor.fit(
                [row.features for row in rows],
                [math.log1p(row.measured_cost_ms) for row in rows],
                schema=schema,
                target_name="log1p-cost-ms",
                ridge=ridge,
            )
        action_models[action_kind] = ActionModel(
            action_kind=action_kind,
            gain_model=gain_model,
            log_cost_model=log_cost_model,
            fallback_gain=fallback_gain,
            fallback_cost_ms=fallback_cost,
            samples=len(rows),
        )
    return LearnedPlannerModel(
        schema=schema,
        action_models=action_models,
        training_digest=_observation_digest(filtered),
        platform_id=platform_id,
    )


def plan_learned_evidence(
    actions: Sequence[ActionSpec],
    model: LearnedPlannerModel,
    *,
    cost_budget_ms: float,
    maximum_actions: int = 8,
    minimum_expected_gain: float = 0.0,
) -> LearnedEvidencePlan:
    if cost_budget_ms < 0 or not math.isfinite(cost_budget_ms):
        raise ValueError("cost_budget_ms must be finite and non-negative")
    if maximum_actions < 1:
        raise ValueError("maximum_actions must be positive")
    if len({action.action_id for action in actions}) != len(actions):
        raise ValueError("action IDs must be unique")
    identifiers = {action.action_id for action in actions}
    if any(set(action.dependencies) - identifiers for action in actions):
        raise ValueError("action dependencies reference unknown IDs")

    predictions: list[ActionPrediction] = []
    for action in actions:
        gain, cost = model.predict(action.action_kind, action.features)
        predictions.append(
            ActionPrediction(
                action=action,
                expected_gain=gain,
                expected_cost_ms=cost,
                utility=gain / max(1.0, cost),
            )
        )
    predictions.sort(
        key=lambda row: (
            not row.action.mandatory,
            -row.utility,
            -row.expected_gain,
            row.expected_cost_ms,
            row.action.action_id,
        )
    )

    selected: list[ActionPrediction] = []
    rejected: list[ActionPrediction] = []
    selected_ids: set[str] = set()
    exclusive_groups: set[str] = set()
    used = 0.0
    pending = list(predictions)
    made_progress = True
    while pending and made_progress and len(selected) < maximum_actions:
        made_progress = False
        next_pending: list[ActionPrediction] = []
        for prediction in pending:
            action = prediction.action
            if not set(action.dependencies).issubset(selected_ids):
                next_pending.append(prediction)
                continue
            if action.exclusive_group and action.exclusive_group in exclusive_groups:
                rejected.append(prediction)
                continue
            if prediction.expected_gain < minimum_expected_gain and not action.mandatory:
                rejected.append(prediction)
                continue
            if used + prediction.expected_cost_ms > cost_budget_ms:
                rejected.append(prediction)
                continue
            selected.append(prediction)
            selected_ids.add(action.action_id)
            if action.exclusive_group:
                exclusive_groups.add(action.exclusive_group)
            used += prediction.expected_cost_ms
            made_progress = True
            if len(selected) >= maximum_actions:
                next_pending.extend(
                    row for row in pending if row.action.action_id not in selected_ids
                )
                break
        pending = next_pending
    rejected.extend(pending)
    if selected and used >= cost_budget_ms:
        reason = "budget-exhausted"
    elif selected:
        reason = "learned-utility-frontier"
    else:
        reason = "no-action-passed-gain-and-budget"
    model_digest = hashlib.sha256(
        json.dumps(
            {
                "schema": model.schema.digest,
                "training": model.training_digest,
                "platform": model.platform_id,
                "actions": sorted(model.action_models),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return LearnedEvidencePlan(
        selected=tuple(selected),
        rejected=tuple(rejected),
        used_cost_ms=used,
        expected_gain=sum(row.expected_gain for row in selected),
        stopping_reason=reason,
        model_digest=model_digest,
    )
