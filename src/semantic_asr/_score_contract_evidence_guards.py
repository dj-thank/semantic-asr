"""Evidence-score construction and serialization guards."""

from __future__ import annotations

from collections.abc import Mapping

from . import _score_contract_base as contract
from ._score_contract_primitives import (
    ScoreKind,
    _aliased,
    _normalization,
    _reject_unknown,
    _strict_bool,
    _strict_mapping,
    _strict_str,
)
from ._score_contract_provenance_guards import _optional_digest


def _raw(
    cls: type[contract.EvidenceScore],
    value: object,
    *,
    semantics: contract.ScoreSemantics,
    scorer: str,
    model: str | None = None,
    revision: str | None = None,
    runtime: str | None = None,
    runtime_version: str | None = None,
    normalization: contract.ScoreNormalization | str = contract.ScoreNormalization.NONE,
    score_domain: contract.ScoreDomain | None = None,
    score_domain_digest: str | None = None,
    configuration_digest: str | None = None,
    input_evidence_digest: str | None = None,
    input_condition_digest: str | None = None,
    metadata: Mapping[str, object] | None = None,
    higher_is_better: bool = True,
) -> contract.EvidenceScore:
    semantics = contract.ScoreSemantics(semantics)
    if semantics == contract.ScoreSemantics.PROBABILITY:
        raise ValueError("use an applicable calibrator profile to construct probabilities")
    normalization = _normalization(normalization)
    metadata_rows = dict(metadata or {})
    if normalization == contract.ScoreNormalization.NONE:
        if semantics == contract.ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD:
            normalization = contract.ScoreNormalization.SEQUENCE
        elif semantics == contract.ScoreSemantics.LOG_PROBABILITY:
            normalization = contract.ScoreNormalization.PATH_NORMALIZED
        elif semantics == contract.ScoreSemantics.BOUNDED_UTILITY:
            normalization = contract.ScoreNormalization.BOUNDED
        elif semantics == contract.ScoreSemantics.AVERAGE_LOG_LIKELIHOOD:
            if metadata_rows.get("frameCount") is not None:
                normalization = contract.ScoreNormalization.MEAN_FRAME
            elif metadata_rows.get("lengthNormalizationAlpha") not in {None, 1, 1.0}:
                normalization = contract.ScoreNormalization.TOKEN_POWER
            elif any(
                metadata_rows.get(key) is not None
                for key in ("tokenCount", "candidateTokenCount")
            ):
                normalization = contract.ScoreNormalization.MEAN_TOKEN
    if score_domain is not None:
        if not isinstance(score_domain, contract.ScoreDomain):
            raise TypeError("score_domain must be ScoreDomain")
        if score_domain_digest is not None and score_domain_digest != score_domain.digest:
            raise ValueError("score_domain and score_domain_digest disagree")
        if normalization == contract.ScoreNormalization.NONE:
            normalization = score_domain.normalization
        if input_condition_digest is None:
            input_condition_digest = score_domain.input_condition_digest
        expected = {
            "scorer": (scorer, score_domain.producer),
            "normalization": (normalization, score_domain.normalization),
            "model": (model, score_domain.model),
            "revision": (revision, score_domain.revision),
            "configuration_digest": (
                configuration_digest,
                score_domain.configuration_digest,
            ),
            "input_condition_digest": (
                input_condition_digest,
                score_domain.input_condition_digest,
            ),
        }
        mismatches = [name for name, pair in expected.items() if pair[0] != pair[1]]
        if mismatches:
            raise ValueError(
                "score does not match supplied score domain: " + ", ".join(mismatches)
            )
        score_domain_digest = score_domain.digest
    return cls(
        value,
        semantics=semantics,
        provenance=contract.ScoreProvenance(
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
            metadata=metadata_rows,
        ),
        higher_is_better=higher_is_better,
    )


def _kind(self: contract.EvidenceScore) -> ScoreKind:
    mapping = {
        contract.ScoreSemantics.UNCALIBRATED_SCORE: ScoreKind.RAW,
        contract.ScoreSemantics.PROBABILITY: ScoreKind.PROBABILITY,
        contract.ScoreSemantics.LOGIT: ScoreKind.LOGIT,
        contract.ScoreSemantics.PREFERENCE: ScoreKind.PREFERENCE,
        contract.ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD: ScoreKind.LOG_LIKELIHOOD,
        contract.ScoreSemantics.AVERAGE_LOG_LIKELIHOOD: ScoreKind.LOG_LIKELIHOOD,
        contract.ScoreSemantics.LOG_PROBABILITY: ScoreKind.LOG_LIKELIHOOD,
    }
    return mapping.get(self.semantics, ScoreKind.RAW)


def _score_from_dict(
    cls: type[contract.EvidenceScore],
    row: Mapping[str, object],
    *,
    legacy_normalization: contract.ScoreNormalization | None = None,
) -> contract.EvidenceScore:
    row = _strict_mapping(row, name="score")
    schema = _aliased(row, "schemaVersion", "schema_version")
    canonical_shape = "semantics" in row or "provenance" in row
    simple_shape = "kind" in row
    if schema is not None:
        schema = _strict_str(schema, name="schemaVersion")
        if schema not in {
            contract.SCORE_SCHEMA_VERSION,
            contract.LEGACY_RICH_SCHEMA_VERSION,
            contract.LEGACY_SIMPLE_SCHEMA_VERSION,
        }:
            raise contract.ScoreMigrationError(f"unsupported score schema: {schema!r}")
        if schema == contract.SCORE_SCHEMA_VERSION and not canonical_shape:
            raise contract.ScoreMigrationError(
                "canonical score schema does not match payload shape"
            )
        if schema == contract.LEGACY_RICH_SCHEMA_VERSION and not canonical_shape:
            raise contract.ScoreMigrationError(
                "legacy rich schema does not match payload shape"
            )
        if schema == contract.LEGACY_SIMPLE_SCHEMA_VERSION and not simple_shape:
            raise contract.ScoreMigrationError(
                "legacy simple schema does not match payload shape"
            )
    if canonical_shape:
        allowed = {
            "schemaVersion",
            "schema_version",
            "value",
            "semantics",
            "provenance",
            "calibrated",
            "higherIsBetter",
            "higher_is_better",
        }
        _reject_unknown(row, allowed=allowed, name="score")
        provenance = _strict_mapping(row.get("provenance"), name="score provenance")
        return cls(
            row["value"],
            semantics=contract.ScoreSemantics(
                _strict_str(row.get("semantics"), name="semantics")
            ),
            provenance=contract.ScoreProvenance.from_dict(provenance),
            calibrated=_strict_bool(row.get("calibrated", False), name="calibrated"),
            higher_is_better=_strict_bool(
                _aliased(row, "higherIsBetter", "higher_is_better", default=True),
                name="higherIsBetter",
            ),
        )
    if simple_shape:
        allowed = {
            "schemaVersion",
            "schema_version",
            "value",
            "kind",
            "source",
            "calibrated",
            "calibrationDigest",
            "calibration_digest",
            "higherIsBetter",
            "higher_is_better",
            "metadata",
        }
        _reject_unknown(row, allowed=allowed, name="legacy simple score")
        metadata = dict(_strict_mapping(row.get("metadata", {}), name="metadata"))
        if legacy_normalization is not None:
            metadata.setdefault("legacyNormalization", legacy_normalization.value)
        return cls(
            row["value"],
            kind=row["kind"],
            source=_strict_str(row.get("source"), name="source"),
            calibrated=_strict_bool(row.get("calibrated", False), name="calibrated"),
            calibration_digest=_optional_digest(
                _aliased(row, "calibrationDigest", "calibration_digest"),
                name="calibrationDigest",
            ),
            higher_is_better=_strict_bool(
                _aliased(row, "higherIsBetter", "higher_is_better", default=True),
                name="higherIsBetter",
            ),
            metadata=metadata,
        )
    raise contract.ScoreMigrationError("unrecognized score serialization")
