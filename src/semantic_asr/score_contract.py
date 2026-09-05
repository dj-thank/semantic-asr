"""Canonical numeric score contract for Semantic ASR.

This module is the single source of truth for score meaning, normalization, provenance,
calibration receipts, and versioned serialization.  Legacy import paths re-export this
``EvidenceScore`` rather than defining another numeric type.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

SCORE_SCHEMA_VERSION = "semantic-asr.score.v2"
LEGACY_RICH_SCHEMA_VERSION = "semantic-asr.score.rich-v1"
LEGACY_SIMPLE_SCHEMA_VERSION = "semantic-asr.score.simple-v1"


class ScoreContractError(ValueError):
    """Base error for score-contract violations."""


class ScoreMigrationError(ScoreContractError):
    """Raised when a legacy score cannot be migrated without losing meaning."""


class ScoreSemantics(StrEnum):
    """Meaning and unit of a numeric score."""

    CUMULATIVE_LOG_LIKELIHOOD = "cumulative_log_likelihood"
    AVERAGE_LOG_LIKELIHOOD = "average_log_likelihood"
    LOG_PROBABILITY = "log_probability"
    PROBABILITY = "probability"
    LOGIT = "logit"
    UNCALIBRATED_SCORE = "uncalibrated_score"
    PREFERENCE = "preference"
    LOSS = "loss"
    COST = "cost"
    BOUNDED_UTILITY = "bounded_utility"


class ScoreNormalization(StrEnum):
    """Normalization used before interpreting or calibrating a score."""

    NONE = "none"
    SEQUENCE = "sequence"
    MEAN_TOKEN = "mean_token"
    MEAN_FRAME = "mean_frame"
    TOKEN_POWER = "token_power"
    PATH_NORMALIZED = "path_normalized"
    BOUNDED = "bounded"


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-min(80.0, value))
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(max(-80.0, value))
    return exponent / (1.0 + exponent)


def _is_sha256(value: str | None) -> bool:
    if value is None or len(value) != 64 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def require_sha256(value: str | None, *, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    assert value is not None
    return value


def _freeze_json(value: object, *, path: str = "metadata") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _strict_float(value, name=path)
    if isinstance(value, Mapping):
        rows: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            rows[key] = _freeze_json(item, path=f"{path}.{key}")
        return FrozenDict(rows)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class FrozenDict(dict[str, object]):
    """JSON-compatible dictionary that cannot be mutated after construction."""

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        rows: dict[str, object] = {}
        for key, value in dict(values or {}).items():
            if not isinstance(key, str) or not key:
                raise TypeError("metadata keys must be non-empty strings")
            rows[key] = _freeze_json(value, path=f"metadata.{key}")
        dict.__init__(self, rows)

    def _immutable(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("score metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> FrozenDict:
        return self


@dataclass(frozen=True, slots=True)
class ScoreDomain:
    """Identity of one score distribution.

    Arithmetic aggregation is legal only when this digest matches.  Model, span,
    prompt, decoding policy, normalization, and input condition therefore cannot
    be mixed accidentally.
    """

    producer: str
    normalization: ScoreNormalization
    input_condition_digest: str
    model: str | None = None
    revision: str | None = None
    span_digest: str | None = None
    prompt_digest: str | None = None
    configuration_digest: str | None = None
    decode_digest: str | None = None
    temperature: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.producer.strip():
            raise ValueError("score-domain producer is required")
        object.__setattr__(self, "normalization", ScoreNormalization(self.normalization))
        require_sha256(self.input_condition_digest, name="input_condition_digest")
        for name in ("span_digest", "prompt_digest", "configuration_digest", "decode_digest"):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=name)
        if self.temperature is not None:
            temperature = _strict_float(self.temperature, name="temperature")
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "producer": self.producer,
            "normalization": self.normalization.value,
            "inputConditionDigest": self.input_condition_digest,
            "model": self.model,
            "revision": self.revision,
            "spanDigest": self.span_digest,
            "promptDigest": self.prompt_digest,
            "configurationDigest": self.configuration_digest,
            "decodeDigest": self.decode_digest,
            "temperature": self.temperature,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScoreProvenance:
    scorer: str
    model: str | None = None
    revision: str | None = None
    runtime: str | None = None
    runtime_version: str | None = None
    normalization: ScoreNormalization = ScoreNormalization.NONE
    score_domain_digest: str | None = None
    configuration_digest: str | None = None
    calibration_digest: str | None = None
    input_evidence_digest: str | None = None
    input_condition_digest: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scorer.strip():
            raise ValueError("scorer is required")
        object.__setattr__(self, "normalization", ScoreNormalization(self.normalization))
        for name in (
            "score_domain_digest",
            "configuration_digest",
            "calibration_digest",
            "input_evidence_digest",
            "input_condition_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=name)
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "scorer": self.scorer,
            "model": self.model,
            "revision": self.revision,
            "runtime": self.runtime,
            "runtimeVersion": self.runtime_version,
            "normalization": self.normalization.value,
            "scoreDomainDigest": self.score_domain_digest,
            "configurationDigest": self.configuration_digest,
            "calibrationDigest": self.calibration_digest,
            "inputEvidenceDigest": self.input_evidence_digest,
            "inputConditionDigest": self.input_condition_digest,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> ScoreProvenance:
        return cls(
            scorer=str(row["scorer"]),
            model=None if row.get("model") is None else str(row["model"]),
            revision=None if row.get("revision") is None else str(row["revision"]),
            runtime=None if row.get("runtime") is None else str(row["runtime"]),
            runtime_version=(
                None
                if row.get("runtimeVersion", row.get("runtime_version")) is None
                else str(row.get("runtimeVersion", row.get("runtime_version")))
            ),
            normalization=ScoreNormalization(
                str(row.get("normalization") or ScoreNormalization.NONE.value)
            ),
            score_domain_digest=(
                None
                if row.get("scoreDomainDigest", row.get("score_domain_digest")) is None
                else str(row.get("scoreDomainDigest", row.get("score_domain_digest")))
            ),
            configuration_digest=(
                None
                if row.get("configurationDigest", row.get("configuration_digest")) is None
                else str(row.get("configurationDigest", row.get("configuration_digest")))
            ),
            calibration_digest=(
                None
                if row.get("calibrationDigest", row.get("calibration_digest")) is None
                else str(row.get("calibrationDigest", row.get("calibration_digest")))
            ),
            input_evidence_digest=(
                None
                if row.get("inputEvidenceDigest", row.get("input_evidence_digest")) is None
                else str(row.get("inputEvidenceDigest", row.get("input_evidence_digest")))
            ),
            input_condition_digest=(
                None
                if row.get("inputConditionDigest", row.get("input_condition_digest")) is None
                else str(row.get("inputConditionDigest", row.get("input_condition_digest")))
            ),
            metadata=dict(row.get("metadata") or {}),
        )


_LEGACY_KIND_TO_SEMANTICS: dict[str, ScoreSemantics] = {
    "raw": ScoreSemantics.UNCALIBRATED_SCORE,
    "probability": ScoreSemantics.PROBABILITY,
    "logit": ScoreSemantics.LOGIT,
    "preference": ScoreSemantics.PREFERENCE,
}

_SEMANTICS_TO_LEGACY_KIND: dict[ScoreSemantics, str] = {
    ScoreSemantics.UNCALIBRATED_SCORE: "raw",
    ScoreSemantics.PROBABILITY: "probability",
    ScoreSemantics.LOGIT: "logit",
    ScoreSemantics.PREFERENCE: "preference",
    ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD: "log_likelihood",
    ScoreSemantics.AVERAGE_LOG_LIKELIHOOD: "log_likelihood",
    ScoreSemantics.LOG_PROBABILITY: "log_likelihood",
}


def _legacy_log_likelihood_semantics(
    normalization: ScoreNormalization,
) -> ScoreSemantics:
    if normalization == ScoreNormalization.SEQUENCE:
        return ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD
    if normalization in {
        ScoreNormalization.MEAN_TOKEN,
        ScoreNormalization.MEAN_FRAME,
        ScoreNormalization.TOKEN_POWER,
    }:
        return ScoreSemantics.AVERAGE_LOG_LIKELIHOOD
    if normalization == ScoreNormalization.PATH_NORMALIZED:
        return ScoreSemantics.LOG_PROBABILITY
    raise ScoreMigrationError(
        "legacy log_likelihood is ambiguous; declare sequence, mean_token, "
        "mean_frame, or path_normalized normalization"
    )


@dataclass(frozen=True, slots=True, init=False)
class EvidenceScore:
    """Canonical score object with a compatibility constructor.

    Canonical callers pass ``semantics`` and ``ScoreProvenance``.  The legacy
    ``score_semantics`` constructor shape is accepted only when its meaning can be
    recovered without guessing.
    """

    value: float
    semantics: ScoreSemantics
    provenance: ScoreProvenance
    calibrated: bool
    higher_is_better: bool
    schema_version: str

    def __init__(
        self,
        value: object,
        semantics: ScoreSemantics | str | object | None = None,
        provenance: ScoreProvenance | str | None = None,
        calibrated: bool = False,
        *,
        kind: object | None = None,
        source: str | None = None,
        calibration_digest: str | None = None,
        higher_is_better: bool = True,
        metadata: Mapping[str, object] | None = None,
        normalization: ScoreNormalization | str | None = None,
        model: str | None = None,
        revision: str | None = None,
        runtime: str | None = None,
        runtime_version: str | None = None,
        configuration_digest: str | None = None,
        score_domain_digest: str | None = None,
        input_evidence_digest: str | None = None,
        input_condition_digest: str | None = None,
        schema_version: str = SCORE_SCHEMA_VERSION,
    ) -> None:
        numeric = _strict_float(value, name="score value")
        if not isinstance(calibrated, bool):
            raise TypeError("calibrated must be bool")
        if not isinstance(higher_is_better, bool):
            raise TypeError("higher_is_better must be bool")

        legacy_kind = kind
        legacy_source = source
        if (
            legacy_kind is None
            and provenance is not None
            and not isinstance(provenance, ScoreProvenance)
        ):
            # Positional legacy shape: EvidenceScore(value, ScoreKind.*, "source", ...)
            legacy_kind = semantics
            legacy_source = str(provenance)
            semantics = None
            provenance = None
        elif kind is not None and semantics is not None:
            raise TypeError("pass either canonical semantics or legacy kind, not both")

        if legacy_kind is not None:
            raw_kind = str(getattr(legacy_kind, "value", legacy_kind))
            declared_normalization = (
                normalization
                or (metadata or {}).get("normalization")
                or (metadata or {}).get("legacyNormalization")
            )
            if (
                raw_kind == "log_likelihood"
                and declared_normalization is None
                and legacy_source is not None
                and legacy_source.startswith(("ctc-phone:", "ctc-mora:"))
            ):
                # Lossless adapter for the repository's historical CTC producer:
                # its stored value is explicitly mean-frame likelihood and carries
                # frameCount in the receipt.
                declared_normalization = ScoreNormalization.MEAN_FRAME
            chosen_normalization = ScoreNormalization(
                declared_normalization or ScoreNormalization.NONE
            )
            if raw_kind == "log_likelihood":
                canonical_semantics = _legacy_log_likelihood_semantics(chosen_normalization)
            else:
                try:
                    canonical_semantics = _LEGACY_KIND_TO_SEMANTICS[raw_kind]
                except KeyError as exc:
                    raise ScoreMigrationError(f"unknown legacy score kind: {raw_kind!r}") from exc
            if not legacy_source:
                raise ValueError("legacy score source is required")
            canonical_provenance = ScoreProvenance(
                scorer=legacy_source,
                model=model,
                revision=revision,
                runtime=runtime,
                runtime_version=runtime_version,
                normalization=chosen_normalization,
                score_domain_digest=score_domain_digest,
                configuration_digest=configuration_digest,
                calibration_digest=calibration_digest,
                input_evidence_digest=input_evidence_digest,
                input_condition_digest=input_condition_digest,
                metadata=dict(metadata or {}),
            )
        else:
            if semantics is None or not isinstance(provenance, ScoreProvenance):
                raise TypeError("canonical scores require semantics and ScoreProvenance")
            canonical_semantics = ScoreSemantics(semantics)
            canonical_provenance = provenance
            if any(
                value is not None
                for value in (
                    source,
                    calibration_digest,
                    metadata,
                    normalization,
                    model,
                    revision,
                    runtime,
                    runtime_version,
                    configuration_digest,
                    score_domain_digest,
                    input_evidence_digest,
                    input_condition_digest,
                )
            ):
                raise TypeError("legacy provenance keywords cannot accompany ScoreProvenance")

        object.__setattr__(self, "value", numeric)
        object.__setattr__(self, "semantics", canonical_semantics)
        object.__setattr__(self, "provenance", canonical_provenance)
        object.__setattr__(self, "calibrated", calibrated)
        object.__setattr__(self, "higher_is_better", higher_is_better)
        object.__setattr__(self, "schema_version", schema_version)
        self._validate()

    def _validate(self) -> None:
        if self.schema_version != SCORE_SCHEMA_VERSION:
            raise ValueError(f"unsupported score schema: {self.schema_version!r}")
        if self.semantics == ScoreSemantics.PROBABILITY:
            if not 0.0 <= self.value <= 1.0:
                raise ValueError("probability must be in [0, 1]")
            if not self.calibrated:
                raise ValueError("probability must be produced by an explicit calibrator")
            if self.provenance.calibration_digest is None:
                raise ValueError("calibrated probability requires a calibration receipt digest")
        else:
            if self.calibrated:
                raise ValueError("only probability scores may be marked calibrated")
            if self.provenance.calibration_digest is not None:
                raise ValueError("non-probability scores cannot carry a calibration receipt")

    @classmethod
    def raw(
        cls,
        value: object,
        *,
        semantics: ScoreSemantics,
        scorer: str,
        model: str | None = None,
        revision: str | None = None,
        runtime: str | None = None,
        runtime_version: str | None = None,
        normalization: ScoreNormalization = ScoreNormalization.NONE,
        score_domain: ScoreDomain | None = None,
        score_domain_digest: str | None = None,
        configuration_digest: str | None = None,
        input_evidence_digest: str | None = None,
        input_condition_digest: str | None = None,
        metadata: Mapping[str, object] | None = None,
        higher_is_better: bool = True,
    ) -> Self:
        semantics = ScoreSemantics(semantics)
        if semantics == ScoreSemantics.PROBABILITY:
            raise ValueError("use an applicable calibrator profile to construct probabilities")
        normalization = ScoreNormalization(normalization)
        if normalization == ScoreNormalization.NONE:
            metadata_rows = dict(metadata or {})
            if semantics == ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD:
                normalization = ScoreNormalization.SEQUENCE
            elif semantics == ScoreSemantics.LOG_PROBABILITY:
                normalization = ScoreNormalization.PATH_NORMALIZED
            elif semantics == ScoreSemantics.AVERAGE_LOG_LIKELIHOOD:
                if metadata_rows.get("frameCount") is not None:
                    normalization = ScoreNormalization.MEAN_FRAME
                elif metadata_rows.get("lengthNormalizationAlpha") not in {None, 1}:
                    normalization = ScoreNormalization.TOKEN_POWER
                elif any(
                    metadata_rows.get(key) is not None
                    for key in ("tokenCount", "candidateTokenCount")
                ):
                    normalization = ScoreNormalization.MEAN_TOKEN
        if score_domain is not None:
            if score_domain_digest is not None and score_domain_digest != score_domain.digest:
                raise ValueError("score_domain and score_domain_digest disagree")
            score_domain_digest = score_domain.digest
            if normalization == ScoreNormalization.NONE:
                normalization = score_domain.normalization
            if input_condition_digest is None:
                input_condition_digest = score_domain.input_condition_digest
        return cls(
            value,
            semantics=semantics,
            provenance=ScoreProvenance(
                scorer=scorer,
                model=model,
                revision=revision,
                runtime=runtime,
                runtime_version=runtime_version,
                normalization=normalization,
                score_domain_digest=score_domain_digest,
                configuration_digest=configuration_digest,
                input_evidence_digest=input_evidence_digest,
                input_condition_digest=input_condition_digest,
                metadata=dict(metadata or {}),
            ),
            higher_is_better=higher_is_better,
        )

    @property
    def source(self) -> str:
        return self.provenance.scorer

    @property
    def calibration_digest(self) -> str | None:
        return self.provenance.calibration_digest

    @property
    def metadata(self) -> Mapping[str, object]:
        return self.provenance.metadata

    @property
    def kind(self) -> str:
        try:
            return _SEMANTICS_TO_LEGACY_KIND[self.semantics]
        except KeyError:
            return "raw"

    @property
    def usable_as_probability(self) -> bool:
        # A registry is still required before the value is accepted by decision logic.
        return (
            self.semantics == ScoreSemantics.PROBABILITY
            and self.calibrated
            and self.calibration_digest is not None
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def require_probability(
        self,
        registry: CalibrationProfileRegistry | None = None,
        *,
        source_score: EvidenceScore | None = None,
    ) -> float:
        if registry is None:
            raise ValueError("a frozen CalibrationProfileRegistry is required")
        registry.validate_probability(self, source_score=source_score)
        return self.value

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "value": self.value,
            "semantics": self.semantics.value,
            "calibrated": self.calibrated,
            "higherIsBetter": self.higher_is_better,
            "provenance": self.provenance.as_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        row: Mapping[str, object],
        *,
        legacy_normalization: ScoreNormalization | None = None,
    ) -> EvidenceScore:
        schema = row.get("schemaVersion", row.get("schema_version"))
        if schema == SCORE_SCHEMA_VERSION:
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise TypeError("canonical score provenance must be an object")
            return cls(
                row["value"],
                semantics=ScoreSemantics(str(row["semantics"])),
                provenance=ScoreProvenance.from_dict(provenance),
                calibrated=bool(row.get("calibrated", False)),
                higher_is_better=bool(row.get("higherIsBetter", row.get("higher_is_better", True))),
            )
        if "semantics" in row and "provenance" in row:
            # Legacy rich v1.
            provenance = row["provenance"]
            if not isinstance(provenance, Mapping):
                raise TypeError("legacy rich provenance must be an object")
            return cls(
                row["value"],
                semantics=ScoreSemantics(str(row["semantics"])),
                provenance=ScoreProvenance.from_dict(provenance),
                calibrated=bool(row.get("calibrated", False)),
                higher_is_better=bool(row.get("higher_is_better", True)),
            )
        if "kind" in row:
            metadata = dict(row.get("metadata") or {})
            if legacy_normalization is not None:
                metadata.setdefault("legacyNormalization", legacy_normalization.value)
            return cls(
                row["value"],
                kind=row["kind"],
                source=str(row.get("source") or ""),
                calibrated=bool(row.get("calibrated", False)),
                calibration_digest=(
                    None
                    if row.get("calibrationDigest", row.get("calibration_digest")) is None
                    else str(row.get("calibrationDigest", row.get("calibration_digest")))
                ),
                higher_is_better=bool(row.get("higherIsBetter", row.get("higher_is_better", True))),
                metadata=metadata,
            )
        raise ScoreMigrationError("unrecognized score serialization")

    def to_legacy_rich_dict(self) -> dict[str, object]:
        """Serialize to the historical rich shape with lossless extension fields.

        Existing readers ignore the added provenance keys; the canonical migrator reads
        them directly instead of polluting user metadata.
        """

        return {
            "value": self.value,
            "semantics": self.semantics.value,
            "provenance": {
                "scorer": self.provenance.scorer,
                "model": self.provenance.model,
                "revision": self.provenance.revision,
                "runtime": self.provenance.runtime,
                "runtime_version": self.provenance.runtime_version,
                "normalization": self.provenance.normalization.value,
                "score_domain_digest": self.provenance.score_domain_digest,
                "configuration_digest": self.provenance.configuration_digest,
                "calibration_digest": self.provenance.calibration_digest,
                "input_evidence_digest": self.provenance.input_evidence_digest,
                "input_condition_digest": self.provenance.input_condition_digest,
                "metadata": dict(_thaw_json(self.provenance.metadata)),
            },
            "calibrated": self.calibrated,
            "higher_is_better": self.higher_is_better,
        }

    def to_legacy_simple_dict(self) -> dict[str, object]:
        try:
            kind = _SEMANTICS_TO_LEGACY_KIND[self.semantics]
        except KeyError as exc:
            raise ScoreMigrationError(
                f"{self.semantics.value} has no lossless simple-v1 representation"
            ) from exc
        metadata = dict(_thaw_json(self.provenance.metadata))
        if kind == "log_likelihood":
            if self.provenance.normalization == ScoreNormalization.NONE:
                raise ScoreMigrationError(
                    "log-likelihood normalization is required for lossless legacy serialization"
                )
            metadata["normalization"] = self.provenance.normalization.value
        return {
            "value": self.value,
            "kind": kind,
            "source": self.provenance.scorer,
            "calibrated": self.calibrated,
            "calibrationDigest": self.provenance.calibration_digest,
            "higherIsBetter": self.higher_is_better,
            "metadata": metadata,
        }


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    """Frozen applicability record for one calibration artifact."""

    calibration_digest: str
    name: str
    method: str
    source_semantics: ScoreSemantics
    scorer: str
    dataset_split_digest: str
    split_name: str = "calibration"
    model: str | None = None
    revision: str | None = None
    normalization: ScoreNormalization = ScoreNormalization.NONE
    score_domain_digest: str | None = None
    configuration_digest: str | None = None
    input_condition_digest: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "semantic-asr.calibration-profile.v1"

    def __post_init__(self) -> None:
        require_sha256(self.calibration_digest, name="calibration_digest")
        require_sha256(self.dataset_split_digest, name="dataset_split_digest")
        if not self.name or not self.method or not self.scorer:
            raise ValueError("calibration profile name, method, and scorer are required")
        if self.split_name != "calibration":
            raise ValueError("correctness calibration must use the calibration split")
        source_semantics = ScoreSemantics(self.source_semantics)
        if source_semantics == ScoreSemantics.PROBABILITY:
            raise ValueError("calibration source semantics cannot already be probability")
        object.__setattr__(self, "source_semantics", source_semantics)
        object.__setattr__(self, "normalization", ScoreNormalization(self.normalization))
        for name in (
            "score_domain_digest",
            "configuration_digest",
            "input_condition_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name=name)
        object.__setattr__(self, "parameters", FrozenDict(self.parameters))
        object.__setattr__(self, "metadata", FrozenDict(self.metadata))
        self._validate_transform()

    def _validate_transform(self) -> None:
        if self.method == "platt":
            _strict_float(self.parameters.get("slope"), name="platt slope")
            _strict_float(self.parameters.get("intercept"), name="platt intercept")
            return
        if self.method == "isotonic-pav":
            thresholds = self.parameters.get("thresholds")
            probabilities = self.parameters.get("probabilities")
            if not isinstance(thresholds, tuple) or not isinstance(probabilities, tuple):
                raise ValueError("isotonic profile requires frozen thresholds and probabilities")
            if not thresholds or len(thresholds) != len(probabilities):
                raise ValueError("isotonic thresholds and probabilities must have equal length")
            threshold_rows = tuple(
                _strict_float(value, name="isotonic threshold") for value in thresholds
            )
            probability_rows = tuple(
                _strict_float(value, name="isotonic probability") for value in probabilities
            )
            if threshold_rows != tuple(sorted(threshold_rows)):
                raise ValueError("isotonic thresholds must be sorted")
            if any(not 0.0 <= value <= 1.0 for value in probability_rows):
                raise ValueError("isotonic probabilities must be in [0, 1]")
            if any(
                left > right
                for left, right in zip(probability_rows, probability_rows[1:], strict=False)
            ):
                raise ValueError("isotonic probabilities must be monotone")
            return
        raise ValueError(f"unsupported verifiable calibration method: {self.method!r}")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "calibrationDigest": self.calibration_digest,
            "name": self.name,
            "method": self.method,
            "sourceSemantics": self.source_semantics.value,
            "scorer": self.scorer,
            "model": self.model,
            "revision": self.revision,
            "normalization": self.normalization.value,
            "scoreDomainDigest": self.score_domain_digest,
            "configurationDigest": self.configuration_digest,
            "inputConditionDigest": self.input_condition_digest,
            "datasetSplitDigest": self.dataset_split_digest,
            "splitName": self.split_name,
            "parameters": _thaw_json(self.parameters),
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> CalibrationProfile:
        return cls(
            calibration_digest=str(row["calibrationDigest"]),
            name=str(row["name"]),
            method=str(row["method"]),
            source_semantics=ScoreSemantics(str(row["sourceSemantics"])),
            scorer=str(row["scorer"]),
            model=None if row.get("model") is None else str(row["model"]),
            revision=None if row.get("revision") is None else str(row["revision"]),
            normalization=ScoreNormalization(
                str(row.get("normalization") or ScoreNormalization.NONE.value)
            ),
            score_domain_digest=(
                None if row.get("scoreDomainDigest") is None else str(row["scoreDomainDigest"])
            ),
            configuration_digest=(
                None if row.get("configurationDigest") is None else str(row["configurationDigest"])
            ),
            input_condition_digest=(
                None
                if row.get("inputConditionDigest") is None
                else str(row["inputConditionDigest"])
            ),
            dataset_split_digest=str(row["datasetSplitDigest"]),
            split_name=str(row.get("splitName") or "calibration"),
            parameters=dict(row.get("parameters") or {}),
            metadata=dict(row.get("metadata") or {}),
            schema_version=str(row.get("schemaVersion") or "semantic-asr.calibration-profile.v1"),
        )

    def assert_applicable(self, score: EvidenceScore) -> None:
        expected = {
            "semantics": (score.semantics, self.source_semantics),
            "scorer": (score.provenance.scorer, self.scorer),
            "model": (score.provenance.model, self.model),
            "revision": (score.provenance.revision, self.revision),
            "normalization": (score.provenance.normalization, self.normalization),
            "score_domain_digest": (
                score.provenance.score_domain_digest,
                self.score_domain_digest,
            ),
            "configuration_digest": (
                score.provenance.configuration_digest,
                self.configuration_digest,
            ),
            "input_condition_digest": (
                score.provenance.input_condition_digest,
                self.input_condition_digest,
            ),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if mismatches:
            raise ValueError(
                "calibration profile is not applicable to this score: " + ", ".join(mismatches)
            )

    def expected_probability(self, source_score: EvidenceScore) -> float:
        self.assert_applicable(source_score)
        if self.method == "platt":
            slope = _strict_float(self.parameters["slope"], name="platt slope")
            intercept = _strict_float(self.parameters["intercept"], name="platt intercept")
            return _sigmoid(slope * source_score.value + intercept)
        thresholds = tuple(float(value) for value in self.parameters["thresholds"])
        probabilities = tuple(float(value) for value in self.parameters["probabilities"])
        index = 0
        while index < len(thresholds) - 1 and source_score.value > thresholds[index]:
            index += 1
        return probabilities[index]

    def probability(
        self,
        source_score: EvidenceScore,
        value: object | None = None,
    ) -> EvidenceScore:
        probability = self.expected_probability(source_score)
        if value is not None:
            claimed = _strict_float(value, name="calibrated probability")
            if not math.isclose(claimed, probability, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("claimed probability does not match the registered transform")
        metadata = {
            **dict(_thaw_json(source_score.provenance.metadata)),
            "sourceSemantics": source_score.semantics.value,
            "sourceScoreDigest": source_score.digest,
            "calibrationDatasetDigest": self.dataset_split_digest,
            "calibrationSplit": self.split_name,
            "calibrationProfileDigest": self.digest,
            "calibrationMethod": self.method,
            "calibrationName": self.name,
        }
        return EvidenceScore(
            probability,
            semantics=ScoreSemantics.PROBABILITY,
            provenance=ScoreProvenance(
                scorer=source_score.provenance.scorer,
                model=source_score.provenance.model,
                revision=source_score.provenance.revision,
                runtime=source_score.provenance.runtime,
                runtime_version=source_score.provenance.runtime_version,
                normalization=source_score.provenance.normalization,
                score_domain_digest=source_score.provenance.score_domain_digest,
                configuration_digest=source_score.provenance.configuration_digest,
                calibration_digest=self.calibration_digest,
                input_evidence_digest=source_score.provenance.input_evidence_digest,
                input_condition_digest=source_score.provenance.input_condition_digest,
                metadata=metadata,
            ),
            calibrated=True,
            higher_is_better=True,
        )


@dataclass(frozen=True, slots=True)
class CalibrationProfileRegistry:
    profiles: tuple[CalibrationProfile, ...]
    registry_name: str
    revision: str
    schema_version: str = "semantic-asr.calibration-registry.v1"

    def __post_init__(self) -> None:
        if not self.registry_name or not self.revision:
            raise ValueError("calibration registry name and revision are required")
        if not self.profiles:
            raise ValueError("calibration registry requires at least one profile")
        digests = [profile.calibration_digest for profile in self.profiles]
        if len(digests) != len(set(digests)):
            raise ValueError("calibration artifact digests must be unique in a registry")
        object.__setattr__(
            self,
            "profiles",
            tuple(sorted(self.profiles, key=lambda profile: profile.calibration_digest)),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": self.schema_version,
            "registryName": self.registry_name,
            "revision": self.revision,
            "profiles": [profile.as_dict() for profile in self.profiles],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, object]) -> CalibrationProfileRegistry:
        profiles = row.get("profiles")
        if not isinstance(profiles, list):
            raise TypeError("calibration registry profiles must be a list")
        return cls(
            profiles=tuple(
                CalibrationProfile.from_dict(profile)
                for profile in profiles
                if isinstance(profile, Mapping)
            ),
            registry_name=str(row["registryName"]),
            revision=str(row["revision"]),
            schema_version=str(row.get("schemaVersion") or "semantic-asr.calibration-registry.v1"),
        )

    def profile(self, calibration_digest: str | None) -> CalibrationProfile:
        require_sha256(calibration_digest, name="calibration_digest")
        for profile in self.profiles:
            if profile.calibration_digest == calibration_digest:
                return profile
        raise ValueError("unknown calibration profile digest")

    def validate_probability(
        self,
        score: EvidenceScore,
        *,
        source_score: EvidenceScore | None = None,
    ) -> CalibrationProfile:
        if score.semantics != ScoreSemantics.PROBABILITY or not score.calibrated:
            raise ValueError("score is not a calibrated probability")
        profile = self.profile(score.provenance.calibration_digest)
        if source_score is None:
            raise ValueError("the source score is required to validate a calibrated probability")
        metadata = score.provenance.metadata
        if metadata.get("sourceSemantics") != profile.source_semantics.value:
            raise ValueError("probability source semantics do not match calibration profile")
        if metadata.get("calibrationDatasetDigest") != profile.dataset_split_digest:
            raise ValueError("probability calibration split does not match profile")
        if metadata.get("calibrationSplit") != profile.split_name:
            raise ValueError("probability calibration split name does not match profile")
        if metadata.get("calibrationProfileDigest") != profile.digest:
            raise ValueError("probability is not bound to the registered profile receipt")
        expected = {
            "scorer": (score.provenance.scorer, profile.scorer),
            "model": (score.provenance.model, profile.model),
            "revision": (score.provenance.revision, profile.revision),
            "normalization": (score.provenance.normalization, profile.normalization),
            "score_domain_digest": (
                score.provenance.score_domain_digest,
                profile.score_domain_digest,
            ),
            "configuration_digest": (
                score.provenance.configuration_digest,
                profile.configuration_digest,
            ),
            "input_condition_digest": (
                score.provenance.input_condition_digest,
                profile.input_condition_digest,
            ),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if mismatches:
            raise ValueError(
                "probability provenance does not match calibration profile: "
                + ", ".join(mismatches)
            )
        profile.assert_applicable(source_score)
        if metadata.get("sourceScoreDigest") != source_score.digest:
            raise ValueError("probability is not bound to the supplied source score")
        expected_probability = profile.expected_probability(source_score)
        if not math.isclose(score.value, expected_probability, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "probability value does not match the registered calibration transform"
            )
        return profile

    def require_probability(
        self,
        score: EvidenceScore,
        *,
        source_score: EvidenceScore | None = None,
    ) -> float:
        self.validate_probability(score, source_score=source_score)
        return score.value


def require_same_score_domain(scores: Iterable[EvidenceScore]) -> str:
    rows = tuple(scores)
    if not rows:
        raise ValueError("at least one score is required")
    semantics = {score.semantics for score in rows}
    if len(semantics) != 1:
        raise ValueError("scores with different semantics cannot be aggregated")
    domains = {score.provenance.score_domain_digest for score in rows}
    if None in domains:
        raise ValueError("score-domain digest is required for aggregation")
    if len(domains) != 1:
        raise ValueError("scores from different score domains cannot be aggregated")
    normalizations = {score.provenance.normalization for score in rows}
    if len(normalizations) != 1:
        raise ValueError("scores with different normalization cannot be aggregated")
    domain = next(iter(domains))
    assert domain is not None
    return domain
