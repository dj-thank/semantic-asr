"""Domain, provenance, and semantic validation guards."""

from __future__ import annotations

from collections.abc import Mapping

from . import _score_contract_base as contract
from ._score_contract_primitives import (
    FrozenDict,
    _aliased,
    _normalization,
    _reject_unknown,
    _strict_mapping,
    _strict_str,
)


def _score_domain_post_init(self: contract.ScoreDomain) -> None:
    producer = _strict_str(self.producer, name="score-domain producer")
    if self.schema_version != "1":
        raise ValueError(f"unsupported score-domain schema: {self.schema_version!r}")
    normalization = _normalization(self.normalization)
    contract.require_sha256(self.input_condition_digest, name="input_condition_digest")
    for name in ("span_digest", "prompt_digest", "configuration_digest", "decode_digest"):
        value = getattr(self, name)
        if value is not None:
            contract.require_sha256(value, name=name)
    temperature = self.temperature
    if temperature is not None:
        temperature = contract._strict_float(temperature, name="temperature")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
    object.__setattr__(self, "producer", producer)
    object.__setattr__(self, "normalization", normalization)
    object.__setattr__(self, "temperature", temperature)
    object.__setattr__(self, "metadata", FrozenDict(self.metadata))


def _provenance_post_init(self: contract.ScoreProvenance) -> None:
    scorer = _strict_str(self.scorer, name="scorer")
    normalization = _normalization(self.normalization)
    for name in (
        "score_domain_digest",
        "configuration_digest",
        "calibration_digest",
        "input_evidence_digest",
        "input_condition_digest",
    ):
        value = getattr(self, name)
        if value is not None:
            contract.require_sha256(value, name=name)
    object.__setattr__(self, "scorer", scorer)
    object.__setattr__(self, "normalization", normalization)
    object.__setattr__(self, "metadata", FrozenDict(self.metadata))


def _provenance_from_dict(
    cls: type[contract.ScoreProvenance], row: Mapping[str, object]
) -> contract.ScoreProvenance:
    row = _strict_mapping(row, name="score provenance")
    allowed = {
        "scorer",
        "model",
        "revision",
        "runtime",
        "runtimeVersion",
        "runtime_version",
        "normalization",
        "scoreDomainDigest",
        "score_domain_digest",
        "configurationDigest",
        "configuration_digest",
        "calibrationDigest",
        "calibration_digest",
        "inputEvidenceDigest",
        "input_evidence_digest",
        "inputConditionDigest",
        "input_condition_digest",
        "metadata",
    }
    _reject_unknown(row, allowed=allowed, name="score provenance")
    metadata = row.get("metadata", {})
    return cls(
        scorer=_strict_str(row.get("scorer"), name="scorer"),
        model=None if row.get("model") is None else _strict_str(row["model"], name="model"),
        revision=(
            None
            if row.get("revision") is None
            else _strict_str(row["revision"], name="revision")
        ),
        runtime=(
            None
            if row.get("runtime") is None
            else _strict_str(row["runtime"], name="runtime")
        ),
        runtime_version=(
            None
            if _aliased(row, "runtimeVersion", "runtime_version") is None
            else _strict_str(
                _aliased(row, "runtimeVersion", "runtime_version"),
                name="runtimeVersion",
            )
        ),
        normalization=_normalization(row.get("normalization", "none")),
        score_domain_digest=_optional_digest(
            _aliased(row, "scoreDomainDigest", "score_domain_digest"),
            name="scoreDomainDigest",
        ),
        configuration_digest=_optional_digest(
            _aliased(row, "configurationDigest", "configuration_digest"),
            name="configurationDigest",
        ),
        calibration_digest=_optional_digest(
            _aliased(row, "calibrationDigest", "calibration_digest"),
            name="calibrationDigest",
        ),
        input_evidence_digest=_optional_digest(
            _aliased(row, "inputEvidenceDigest", "input_evidence_digest"),
            name="inputEvidenceDigest",
        ),
        input_condition_digest=_optional_digest(
            _aliased(row, "inputConditionDigest", "input_condition_digest"),
            name="inputConditionDigest",
        ),
        metadata=dict(_strict_mapping(metadata, name="metadata")),
    )


def _optional_digest(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    text = _strict_str(value, name=name)
    return contract.require_sha256(text, name=name)


def _validate_score(self: contract.EvidenceScore) -> None:
    if self.schema_version != contract.SCORE_SCHEMA_VERSION:
        raise ValueError(f"unsupported score schema: {self.schema_version!r}")
    semantics = contract.ScoreSemantics(self.semantics)
    normalization = _normalization(self.provenance.normalization)
    allowed: dict[contract.ScoreSemantics, set[contract.ScoreNormalization]] = {
        contract.ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD: {
            contract.ScoreNormalization.SEQUENCE
        },
        contract.ScoreSemantics.AVERAGE_LOG_LIKELIHOOD: {
            contract.ScoreNormalization.MEAN_TOKEN,
            contract.ScoreNormalization.MEAN_FRAME,
            contract.ScoreNormalization.TOKEN_POWER,
        },
        contract.ScoreSemantics.LOG_PROBABILITY: {
            contract.ScoreNormalization.PATH_NORMALIZED
        },
        contract.ScoreSemantics.BOUNDED_UTILITY: {
            contract.ScoreNormalization.BOUNDED
        },
    }
    if semantics in allowed and normalization not in allowed[semantics]:
        raise ValueError(
            f"{semantics.value} is incompatible with {normalization.value} normalization"
        )
    if normalization == contract.ScoreNormalization.TOKEN_POWER:
        alpha = self.provenance.metadata.get("lengthNormalizationAlpha")
        if alpha is None:
            raise ValueError("token_power normalization requires lengthNormalizationAlpha")
        if contract._strict_float(alpha, name="lengthNormalizationAlpha") <= 0:
            raise ValueError("lengthNormalizationAlpha must be positive")
    if semantics == contract.ScoreSemantics.PROBABILITY:
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
