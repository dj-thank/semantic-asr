from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, Self

from .contracts import sha256_json
from .discrete_units import (
    CollapsedUnitSequence,
    DiscreteTokenLanguageModel,
    DiscreteUnitSequence,
    DiscreteUnitSpace,
    SurprisalProfile,
    ensure_same_unit_space,
    validate_sha256,
)

ProjectionMode = Literal["mean", "maximum"]


def _population_std(values: tuple[float, ...]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc


@dataclass(frozen=True, slots=True)
class CentroidDistanceTable:
    """Pairwise Euclidean distances between frozen codebook centroids."""

    space: DiscreteUnitSpace
    distances: tuple[tuple[float, ...], ...]
    metric: str = "euclidean-l2"
    matrix_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        size = self.space.codebook_size
        try:
            normalized = tuple(
                tuple(_strict_float(value, name="centroid distance") for value in row)
                for row in self.distances
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("centroid distances must be a numeric square matrix") from exc
        if len(normalized) != size or any(len(row) != size for row in normalized):
            raise ValueError("distance table dimensions must equal codebook_size")
        for index, row in enumerate(normalized):
            for other, value in enumerate(row):
                if not math.isfinite(value) or value < 0:
                    raise ValueError("centroid distances must be finite and non-negative")
                if index == other and not math.isclose(value, 0.0, abs_tol=1e-9):
                    raise ValueError("centroid distance diagonal must be zero")
                if not math.isclose(
                    value,
                    normalized[other][index],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError("centroid distance table must be symmetric")
        if not isinstance(self.metric, str) or self.metric.strip() != "euclidean-l2":
            raise ValueError("only Euclidean L2 centroid distance is supported")
        object.__setattr__(self, "distances", normalized)
        object.__setattr__(self, "metric", "euclidean-l2")
        object.__setattr__(
            self,
            "matrix_sha256",
            sha256_json({"metric": self.metric, "distances": normalized}),
        )

    @classmethod
    def from_centroids(
        cls,
        space: DiscreteUnitSpace,
        centroids: Sequence[Sequence[float]],
    ) -> Self:
        if len(centroids) != space.codebook_size:
            raise ValueError("centroid count must equal codebook_size")
        dimensions = {len(row) for row in centroids}
        if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) < 1:
            raise ValueError("centroids must share a positive dimensionality")
        normalized: list[tuple[float, ...]] = []
        for row in centroids:
            values = tuple(_strict_float(value, name="centroid coordinate") for value in row)
            if any(not math.isfinite(value) for value in values):
                raise ValueError("centroid coordinates must be finite")
            normalized.append(values)
        distances = tuple(
            tuple(math.dist(left, right) for right in normalized) for left in normalized
        )
        return cls(space=space, distances=distances)

    def distance(self, left: int, right: int) -> float:
        if isinstance(left, bool) or not isinstance(left, Integral):
            raise TypeError("left centroid token ID must be an integer")
        if isinstance(right, bool) or not isinstance(right, Integral):
            raise TypeError("right centroid token ID must be an integer")
        left_index = int(left)
        right_index = int(right)
        if not (
            0 <= left_index < self.space.codebook_size
            and 0 <= right_index < self.space.codebook_size
        ):
            raise ValueError("centroid token ID is outside the codebook")
        return self.distances[left_index][right_index]

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": "centroid-distance-table-v1",
            "unitSpace": self.space.as_dict(),
            "metric": self.metric,
            "distances": self.distances,
            "matrixSha256": self.matrix_sha256,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Self:
        if row.get("schemaVersion") != "centroid-distance-table-v1":
            raise ValueError("unsupported centroid distance table schema")
        table = cls(
            space=DiscreteUnitSpace.from_dict(dict(row["unitSpace"])),
            distances=tuple(tuple(values) for values in row["distances"]),
            metric=row["metric"],
        )
        expected_matrix = validate_sha256(
            row.get("matrixSha256"),
            name="matrixSha256",
        )
        if table.matrix_sha256 != expected_matrix:
            raise ValueError("centroid distance matrix digest mismatch")
        return table

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"artifact": self.as_dict(), "artifactSha256": self.digest}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("artifact"), dict):
            raise ValueError("centroid distance artifact has an invalid envelope")
        table = cls.from_dict(payload["artifact"])
        expected = validate_sha256(
            payload.get("artifactSha256"),
            name="artifactSha256",
        )
        if table.digest != expected:
            raise ValueError("centroid distance artifact digest mismatch")
        return table

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "unitSpaceDigest": self.space.digest,
                "metric": self.metric,
                "matrixSha256": self.matrix_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class DTWConfig:
    max_cells: int = 2_000_000
    projection: ProjectionMode = "mean"
    schema_version: str = "centroid-dtw-v1"

    def __post_init__(self) -> None:
        if isinstance(self.max_cells, bool) or not isinstance(self.max_cells, int):
            raise TypeError("max_cells must be an integer")
        if self.max_cells < 1:
            raise ValueError("max_cells must be positive")
        if self.projection not in {"mean", "maximum"}:
            raise ValueError("projection must be mean or maximum")
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version is required")
        object.__setattr__(self, "schema_version", self.schema_version.strip())

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "maxCells": self.max_cells,
                "projection": self.projection,
                "schemaVersion": self.schema_version,
                "optimization": ("minimum-total-cost", "minimum-path-length"),
                "tieBreak": ("diagonal", "canonical-step", "observed-step"),
            }
        )


@dataclass(frozen=True, slots=True)
class DTWAlignment:
    path: tuple[tuple[int, int], ...]
    local_costs: tuple[float, ...]
    mismatches: tuple[bool, ...]
    total_cost: float
    normalized_cost: float
    canonical_length: int
    observed_length: int
    unit_space_digest: str
    distance_table_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        normalized_path: list[tuple[int, int]] = []
        for pair in self.path:
            if len(pair) != 2:
                raise ValueError("each DTW path entry must contain two indexes")
            left, right = pair
            if (
                isinstance(left, bool)
                or not isinstance(left, Integral)
                or isinstance(right, bool)
                or not isinstance(right, Integral)
            ):
                raise TypeError("DTW path indexes must be integers")
            normalized_path.append((int(left), int(right)))
        local_costs = tuple(
            _strict_float(value, name="DTW local cost") for value in self.local_costs
        )
        if any(not isinstance(value, bool) for value in self.mismatches):
            raise TypeError("DTW mismatch flags must be booleans")
        mismatches = tuple(self.mismatches)
        canonical_length = int(self.canonical_length)
        observed_length = int(self.observed_length)
        if (
            isinstance(self.canonical_length, bool)
            or not isinstance(self.canonical_length, Integral)
            or isinstance(self.observed_length, bool)
            or not isinstance(self.observed_length, Integral)
        ):
            raise TypeError("DTW sequence lengths must be integers")
        total_cost = _strict_float(self.total_cost, name="DTW total_cost")
        normalized_cost = _strict_float(self.normalized_cost, name="DTW normalized_cost")
        path = tuple(normalized_path)
        if not path or not (len(path) == len(local_costs) == len(mismatches)):
            raise ValueError("DTW path, costs and mismatch flags must align")
        if canonical_length < 1 or observed_length < 1:
            raise ValueError("DTW sequence lengths must be positive")
        if any(not math.isfinite(value) or value < 0 for value in local_costs):
            raise ValueError("DTW local costs must be finite and non-negative")
        if not math.isfinite(total_cost) or not math.isfinite(normalized_cost):
            raise ValueError("DTW costs must be finite")
        if total_cost < 0 or normalized_cost < 0:
            raise ValueError("DTW costs must be non-negative")
        if not math.isclose(total_cost, sum(local_costs), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("DTW total_cost must equal the path-local cost sum")
        if not math.isclose(
            normalized_cost,
            total_cost / len(path),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("DTW normalized_cost must use path length")
        if path[0] != (0, 0) or path[-1] != (
            canonical_length - 1,
            observed_length - 1,
        ):
            raise ValueError("DTW path must connect both sequence endpoints")
        for (left_i, left_j), (right_i, right_j) in zip(
            path,
            path[1:],
            strict=False,
        ):
            step = (right_i - left_i, right_j - left_j)
            if step not in {(1, 1), (1, 0), (0, 1)}:
                raise ValueError("DTW path must use monotone unit steps")
        if any(
            not 0 <= canonical_index < canonical_length or not 0 <= observed_index < observed_length
            for canonical_index, observed_index in path
        ):
            raise ValueError("DTW path index is outside its sequence")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "local_costs", local_costs)
        object.__setattr__(self, "mismatches", mismatches)
        object.__setattr__(self, "total_cost", total_cost)
        object.__setattr__(self, "normalized_cost", normalized_cost)
        object.__setattr__(self, "canonical_length", canonical_length)
        object.__setattr__(self, "observed_length", observed_length)
        for name in ("unit_space_digest", "distance_table_digest", "config_digest"):
            object.__setattr__(
                self,
                name,
                validate_sha256(getattr(self, name), name=name),
            )

    @property
    def path_length(self) -> int:
        return len(self.path)

    @property
    def mismatch_rate(self) -> float:
        return sum(self.mismatches) / len(self.mismatches)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "path": self.path,
                "localCosts": self.local_costs,
                "mismatches": self.mismatches,
                "totalCost": self.total_cost,
                "normalizedCost": self.normalized_cost,
                "canonicalLength": self.canonical_length,
                "observedLength": self.observed_length,
                "unitSpaceDigest": self.unit_space_digest,
                "distanceTableDigest": self.distance_table_digest,
                "configDigest": self.config_digest,
            }
        )

    def project_to_observed(
        self, mode: ProjectionMode
    ) -> tuple[tuple[float, ...], tuple[bool, ...]]:
        costs: list[list[float]] = [[] for _ in range(self.observed_length)]
        mismatches: list[bool] = [False] * self.observed_length
        for (_canonical_index, observed_index), local_cost, mismatch in zip(
            self.path,
            self.local_costs,
            self.mismatches,
            strict=True,
        ):
            costs[observed_index].append(local_cost)
            mismatches[observed_index] = mismatches[observed_index] or mismatch
        if any(not values for values in costs):
            raise RuntimeError("DTW path did not cover every observed position")
        if mode == "mean":
            projected = tuple(fmean(values) for values in costs)
        elif mode == "maximum":
            projected = tuple(max(values) for values in costs)
        else:
            raise ValueError("projection must be mean or maximum")
        return projected, tuple(mismatches)


def align_collapsed_units(
    canonical: CollapsedUnitSequence,
    observed: CollapsedUnitSequence,
    *,
    distance_table: CentroidDistanceTable,
    config: DTWConfig | None = None,
) -> DTWAlignment:
    """Align collapsed canonical and observed units with bounded deterministic DTW."""

    config = config or DTWConfig()
    ensure_same_unit_space(canonical.space, observed.space, name="observed sequence")
    ensure_same_unit_space(canonical.space, distance_table.space, name="distance table")
    rows = len(canonical.units)
    columns = len(observed.units)
    if rows * columns > config.max_cells:
        raise ValueError(
            f"DTW grid has {rows * columns} cells, exceeding max_cells={config.max_cells}"
        )

    infinity = math.inf
    unreachable_length = rows + columns + 1
    previous_costs = [infinity] * (columns + 1)
    previous_lengths = [unreachable_length] * (columns + 1)
    previous_costs[0] = 0.0
    previous_lengths[0] = 0
    backpointers = bytearray((rows + 1) * (columns + 1))
    stride = columns + 1
    for row_index in range(1, rows + 1):
        current_costs = [infinity] * (columns + 1)
        current_lengths = [unreachable_length] * (columns + 1)
        canonical_unit = canonical.units[row_index - 1]
        for column_index in range(1, columns + 1):
            # Minimize total DTW cost first, then choose the shortest path among
            # equal-cost optima. The secondary objective is global rather than a
            # local direction preference, so path-length normalization remains
            # invariant when canonical and observed inputs are transposed.
            predecessor_cost = previous_costs[column_index - 1]
            predecessor_length = previous_lengths[column_index - 1]
            direction = 0

            canonical_step_cost = previous_costs[column_index]
            canonical_step_length = previous_lengths[column_index]
            if canonical_step_cost < predecessor_cost or (
                canonical_step_cost == predecessor_cost
                and canonical_step_length < predecessor_length
            ):
                predecessor_cost = canonical_step_cost
                predecessor_length = canonical_step_length
                direction = 1

            observed_step_cost = current_costs[column_index - 1]
            observed_step_length = current_lengths[column_index - 1]
            if observed_step_cost < predecessor_cost or (
                observed_step_cost == predecessor_cost and observed_step_length < predecessor_length
            ):
                predecessor_cost = observed_step_cost
                predecessor_length = observed_step_length
                direction = 2
            local = distance_table.distance(canonical_unit, observed.units[column_index - 1])
            current_costs[column_index] = predecessor_cost + local
            current_lengths[column_index] = predecessor_length + 1
            backpointers[row_index * stride + column_index] = direction
        previous_costs = current_costs
        previous_lengths = current_lengths

    row_index = rows
    column_index = columns
    reversed_path: list[tuple[int, int]] = []
    while row_index > 0 and column_index > 0:
        reversed_path.append((row_index - 1, column_index - 1))
        direction = backpointers[row_index * stride + column_index]
        if direction == 0:
            row_index -= 1
            column_index -= 1
        elif direction == 1:
            row_index -= 1
        elif direction == 2:
            column_index -= 1
        else:  # pragma: no cover - bytearray only stores the three values above
            raise RuntimeError("invalid DTW backpointer")
    if row_index or column_index:
        raise RuntimeError("DTW backtracking terminated before reaching the origin")
    path = tuple(reversed(reversed_path))
    local_costs = tuple(
        distance_table.distance(canonical.units[left], observed.units[right])
        for left, right in path
    )
    mismatches = tuple(canonical.units[left] != observed.units[right] for left, right in path)
    total_cost = sum(local_costs)
    return DTWAlignment(
        path=path,
        local_costs=local_costs,
        mismatches=mismatches,
        total_cost=total_cost,
        normalized_cost=total_cost / len(path),
        canonical_length=rows,
        observed_length=columns,
        unit_space_digest=canonical.space.digest,
        distance_table_digest=distance_table.digest,
        config_digest=config.digest,
    )


@dataclass(frozen=True, slots=True)
class CentroidDTWFeatures:
    """Candidate-specific alignment features that do not require a token LM."""

    dtw_distance: float
    token_mismatch_rate: float
    alignment_path_length: int
    canonical_collapsed_units: int
    observed_collapsed_units: int
    alignment_digest: str
    observed_sequence_digest: str
    canonical_sequence_digest: str
    distance_table_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        dtw_distance = _strict_float(self.dtw_distance, name="dtw_distance")
        mismatch_rate = _strict_float(self.token_mismatch_rate, name="token_mismatch_rate")
        if not math.isfinite(dtw_distance) or dtw_distance < 0:
            raise ValueError("dtw_distance must be finite and non-negative")
        if not math.isfinite(mismatch_rate) or not 0 <= mismatch_rate <= 1:
            raise ValueError("token_mismatch_rate must be in [0, 1]")
        lengths: dict[str, int] = {}
        for name in (
            "alignment_path_length",
            "canonical_collapsed_units",
            "observed_collapsed_units",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            lengths[name] = int(value)
        if min(lengths.values()) < 1:
            raise ValueError("DTW feature lengths must be positive")
        object.__setattr__(self, "dtw_distance", dtw_distance)
        object.__setattr__(self, "token_mismatch_rate", mismatch_rate)
        for name, value in lengths.items():
            object.__setattr__(self, name, value)
        for name in (
            "alignment_digest",
            "observed_sequence_digest",
            "canonical_sequence_digest",
            "distance_table_digest",
            "config_digest",
        ):
            object.__setattr__(
                self,
                name,
                validate_sha256(getattr(self, name), name=name),
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "dtwDistance": self.dtw_distance,
            "tokenMismatchRate": self.token_mismatch_rate,
            "alignmentPathLength": self.alignment_path_length,
            "canonicalCollapsedUnits": self.canonical_collapsed_units,
            "observedCollapsedUnits": self.observed_collapsed_units,
            "alignmentDigest": self.alignment_digest,
            "observedSequenceDigest": self.observed_sequence_digest,
            "canonicalSequenceDigest": self.canonical_sequence_digest,
            "distanceTableDigest": self.distance_table_digest,
            "dtwConfigDigest": self.config_digest,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict())


def _align_sequences(
    observed: DiscreteUnitSequence,
    canonical: DiscreteUnitSequence,
    *,
    distance_table: CentroidDistanceTable,
    config: DTWConfig,
) -> tuple[CollapsedUnitSequence, CollapsedUnitSequence, DTWAlignment]:
    ensure_same_unit_space(observed.space, canonical.space, name="canonical sequence")
    ensure_same_unit_space(observed.space, distance_table.space, name="distance table")
    observed_collapsed = observed.collapse()
    canonical_collapsed = canonical.collapse()
    alignment = align_collapsed_units(
        canonical_collapsed,
        observed_collapsed,
        distance_table=distance_table,
        config=config,
    )
    return observed_collapsed, canonical_collapsed, alignment


def centroid_dtw_features(
    observed: DiscreteUnitSequence,
    canonical: DiscreteUnitSequence,
    *,
    distance_table: CentroidDistanceTable,
    config: DTWConfig | None = None,
) -> CentroidDTWFeatures:
    """Return candidate-specific zero-shot DTW features without a native token LM."""

    resolved_config = config or DTWConfig()
    observed_collapsed, canonical_collapsed, alignment = _align_sequences(
        observed,
        canonical,
        distance_table=distance_table,
        config=resolved_config,
    )
    return CentroidDTWFeatures(
        dtw_distance=alignment.normalized_cost,
        token_mismatch_rate=alignment.mismatch_rate,
        alignment_path_length=alignment.path_length,
        canonical_collapsed_units=len(canonical_collapsed.units),
        observed_collapsed_units=len(observed_collapsed.units),
        alignment_digest=alignment.digest,
        observed_sequence_digest=observed.digest,
        canonical_sequence_digest=canonical.digest,
        distance_table_digest=distance_table.digest,
        config_digest=resolved_config.digest,
    )


@dataclass(frozen=True, slots=True)
class TranscriptGuidedFeatures:
    dtw_distance: float
    token_mismatch_rate: float
    mismatch_surprisal_std_bits: float
    weighted_surprisal_std_bits: float
    mismatch_frame_count: int
    alignment_path_length: int
    canonical_collapsed_units: int
    observed_collapsed_units: int
    alpha: float
    projection: ProjectionMode
    alignment_digest: str
    token_lm_digest: str
    observed_sequence_digest: str
    canonical_sequence_digest: str
    distance_table_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        numeric = {
            "dtw_distance": _strict_float(self.dtw_distance, name="dtw_distance"),
            "token_mismatch_rate": _strict_float(
                self.token_mismatch_rate,
                name="token_mismatch_rate",
            ),
            "mismatch_surprisal_std_bits": _strict_float(
                self.mismatch_surprisal_std_bits,
                name="mismatch_surprisal_std_bits",
            ),
            "weighted_surprisal_std_bits": _strict_float(
                self.weighted_surprisal_std_bits,
                name="weighted_surprisal_std_bits",
            ),
            "alpha": _strict_float(self.alpha, name="alpha"),
        }
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("transcript-guided features must be finite")
        if numeric["dtw_distance"] < 0 or not 0 <= numeric["token_mismatch_rate"] <= 1:
            raise ValueError("invalid DTW distance or mismatch rate")
        if numeric["mismatch_surprisal_std_bits"] < 0 or numeric["weighted_surprisal_std_bits"] < 0:
            raise ValueError("surprisal standard deviations must be non-negative")
        if numeric["alpha"] < 0:
            raise ValueError("alpha must be non-negative")

        counts: dict[str, int] = {}
        for name in (
            "mismatch_frame_count",
            "alignment_path_length",
            "canonical_collapsed_units",
            "observed_collapsed_units",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            counts[name] = int(value)
        if counts["mismatch_frame_count"] < 0:
            raise ValueError("mismatch_frame_count must be non-negative")
        if (
            min(
                counts["alignment_path_length"],
                counts["canonical_collapsed_units"],
                counts["observed_collapsed_units"],
            )
            < 1
        ):
            raise ValueError("alignment and collapsed sequence lengths must be positive")
        if self.projection not in {"mean", "maximum"}:
            raise ValueError("projection must be mean or maximum")
        for name, value in numeric.items():
            object.__setattr__(self, name, value)
        for name, value in counts.items():
            object.__setattr__(self, name, value)
        for name in (
            "alignment_digest",
            "token_lm_digest",
            "observed_sequence_digest",
            "canonical_sequence_digest",
            "distance_table_digest",
            "config_digest",
        ):
            object.__setattr__(
                self,
                name,
                validate_sha256(getattr(self, name), name=name),
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "dtwDistance": self.dtw_distance,
            "tokenMismatchRate": self.token_mismatch_rate,
            "mismatchSurprisalStdBits": self.mismatch_surprisal_std_bits,
            "weightedSurprisalStdBits": self.weighted_surprisal_std_bits,
            "mismatchFrameCount": self.mismatch_frame_count,
            "alignmentPathLength": self.alignment_path_length,
            "canonicalCollapsedUnits": self.canonical_collapsed_units,
            "observedCollapsedUnits": self.observed_collapsed_units,
            "alpha": self.alpha,
            "projection": self.projection,
            "alignmentDigest": self.alignment_digest,
            "tokenLmDigest": self.token_lm_digest,
            "observedSequenceDigest": self.observed_sequence_digest,
            "canonicalSequenceDigest": self.canonical_sequence_digest,
            "distanceTableDigest": self.distance_table_digest,
            "dtwConfigDigest": self.config_digest,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict())


def transcript_guided_features(
    observed: DiscreteUnitSequence,
    canonical: DiscreteUnitSequence,
    *,
    token_lm: DiscreteTokenLanguageModel,
    distance_table: CentroidDistanceTable,
    alpha: float = 0.5,
    config: DTWConfig | None = None,
) -> TranscriptGuidedFeatures:
    """Extract the four transcript-guided features described by the paper.

    Repeated acoustic units are collapsed only for DTW. Token surprisal remains at
    the original frame rate, then inherits the local DTW distance of its collapsed
    position. When one observed position participates in multiple path steps, this
    implementation projects their mean (or maximum when configured) explicitly.
    """

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    config = config or DTWConfig()
    ensure_same_unit_space(observed.space, token_lm.space, name="token LM")
    observed_collapsed, canonical_collapsed, alignment = _align_sequences(
        observed,
        canonical,
        distance_table=distance_table,
        config=config,
    )
    collapsed_distances, collapsed_mismatches = alignment.project_to_observed(config.projection)
    surprisal = token_lm.token_surprisals(observed)
    frame_distances = tuple(
        collapsed_distances[collapsed_index]
        for collapsed_index in observed_collapsed.frame_to_collapsed
    )
    frame_mismatches = tuple(
        collapsed_mismatches[collapsed_index]
        for collapsed_index in observed_collapsed.frame_to_collapsed
    )
    mismatch_surprisal = tuple(
        value for value, mismatch in zip(surprisal, frame_mismatches, strict=True) if mismatch
    )
    weighted_surprisal = tuple(
        value * (1.0 + alpha * distance)
        for value, distance in zip(surprisal, frame_distances, strict=True)
    )
    return TranscriptGuidedFeatures(
        dtw_distance=alignment.normalized_cost,
        token_mismatch_rate=alignment.mismatch_rate,
        mismatch_surprisal_std_bits=_population_std(mismatch_surprisal),
        weighted_surprisal_std_bits=_population_std(weighted_surprisal),
        mismatch_frame_count=len(mismatch_surprisal),
        alignment_path_length=alignment.path_length,
        canonical_collapsed_units=len(canonical_collapsed.units),
        observed_collapsed_units=len(observed_collapsed.units),
        alpha=alpha,
        projection=config.projection,
        alignment_digest=alignment.digest,
        token_lm_digest=token_lm.digest,
        observed_sequence_digest=observed.digest,
        canonical_sequence_digest=canonical.digest,
        distance_table_digest=distance_table.digest,
        config_digest=config.digest,
    )


def pronunciation_feature_vector(
    audio: SurprisalProfile,
    transcript: TranscriptGuidedFeatures | None = None,
) -> dict[str, float]:
    """Return the paper feature names without fitting or claiming a calibrator."""

    output = {
        "surprisal_std_bits": audio.std_bits,
        "spike_rate": audio.spike_rate,
        "duration_units": float(audio.duration_units),
    }
    if transcript is None:
        return output
    if transcript.observed_sequence_digest != audio.sequence_digest:
        raise ValueError("audio and transcript-guided features use different observations")
    if transcript.token_lm_digest != audio.token_lm_digest:
        raise ValueError("audio and transcript-guided features use different token LMs")
    output.update(
        {
            "dtw_distance": transcript.dtw_distance,
            "token_mismatch_rate": transcript.token_mismatch_rate,
            "mismatch_surprisal_std_bits": transcript.mismatch_surprisal_std_bits,
            "weighted_surprisal_std_bits": transcript.weighted_surprisal_std_bits,
        }
    )
    return output
