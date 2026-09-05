"""Calibration profile parsing and applicability guards."""

from __future__ import annotations

import math
from collections.abc import Mapping

from . import _score_contract_base as contract
from ._score_contract_primitives import (
    _normalization,
    _reject_unknown,
    _strict_list,
    _strict_mapping,
    _strict_str,
)
from ._score_contract_provenance_guards import _optional_digest


def _profile_from_dict(
    cls: type[contract.CalibrationProfile], row: Mapping[str, object]
) -> contract.CalibrationProfile:
    row = _strict_mapping(row, name="calibration profile")
    allowed = {
        "schemaVersion",
        "calibrationDigest",
        "name",
        "method",
        "sourceSemantics",
        "scorer",
        "model",
        "revision",
        "normalization",
        "scoreDomainDigest",
        "configurationDigest",
        "inputConditionDigest",
        "datasetSplitDigest",
        "splitName",
        "parameters",
        "metadata",
    }
    _reject_unknown(row, allowed=allowed, name="calibration profile")
    schema = _strict_str(
        row.get("schemaVersion", "semantic-asr.calibration-profile.v1"),
        name="schemaVersion",
    )
    if schema != "semantic-asr.calibration-profile.v1":
        raise contract.ScoreMigrationError(
            f"unsupported calibration profile schema: {schema!r}"
        )
    parameters = _strict_mapping(row.get("parameters", {}), name="parameters")
    metadata = _strict_mapping(row.get("metadata", {}), name="metadata")
    return cls(
        calibration_digest=contract.require_sha256(
            _strict_str(row.get("calibrationDigest"), name="calibrationDigest"),
            name="calibration_digest",
        ),
        name=_strict_str(row.get("name"), name="name"),
        method=_strict_str(row.get("method"), name="method"),
        source_semantics=contract.ScoreSemantics(
            _strict_str(row.get("sourceSemantics"), name="sourceSemantics")
        ),
        scorer=_strict_str(row.get("scorer"), name="scorer"),
        model=(
            None
            if row.get("model") is None
            else _strict_str(row["model"], name="model")
        ),
        revision=(
            None
            if row.get("revision") is None
            else _strict_str(row["revision"], name="revision")
        ),
        normalization=_normalization(row.get("normalization", "none")),
        score_domain_digest=_optional_digest(
            row.get("scoreDomainDigest"), name="scoreDomainDigest"
        ),
        configuration_digest=_optional_digest(
            row.get("configurationDigest"), name="configurationDigest"
        ),
        input_condition_digest=_optional_digest(
            row.get("inputConditionDigest"), name="inputConditionDigest"
        ),
        dataset_split_digest=contract.require_sha256(
            _strict_str(row.get("datasetSplitDigest"), name="datasetSplitDigest"),
            name="dataset_split_digest",
        ),
        split_name=_strict_str(row.get("splitName", "calibration"), name="splitName"),
        parameters=dict(parameters),
        metadata=dict(metadata),
        schema_version=schema,
    )


def _registry_from_dict(
    cls: type[contract.CalibrationProfileRegistry], row: Mapping[str, object]
) -> contract.CalibrationProfileRegistry:
    row = _strict_mapping(row, name="calibration registry")
    allowed = {"schemaVersion", "registryName", "revision", "profiles"}
    _reject_unknown(row, allowed=allowed, name="calibration registry")
    schema = _strict_str(
        row.get("schemaVersion", "semantic-asr.calibration-registry.v1"),
        name="schemaVersion",
    )
    if schema != "semantic-asr.calibration-registry.v1":
        raise contract.ScoreMigrationError(
            f"unsupported calibration registry schema: {schema!r}"
        )
    rows = _strict_list(row.get("profiles"), name="profiles")
    profiles = []
    for index, profile in enumerate(rows):
        if not isinstance(profile, Mapping):
            raise TypeError(f"profiles[{index}] must be an object")
        profiles.append(contract.CalibrationProfile.from_dict(profile))
    return cls(
        profiles=tuple(profiles),
        registry_name=_strict_str(row.get("registryName"), name="registryName"),
        revision=_strict_str(row.get("revision"), name="revision"),
        schema_version=schema,
    )


def _profile_assert_applicable(
    self: contract.CalibrationProfile, score: contract.EvidenceScore
) -> None:
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
        "higher_is_better": (
            score.higher_is_better,
            bool(self.metadata.get("sourceHigherIsBetter", True)),
        ),
    }
    optional = {
        "runtime": (score.provenance.runtime, self.metadata.get("runtime")),
        "runtime_version": (
            score.provenance.runtime_version,
            self.metadata.get("runtimeVersion"),
        ),
        "input_evidence_digest": (
            score.provenance.input_evidence_digest,
            self.metadata.get("inputEvidenceDigest"),
        ),
    }
    expected.update(
        {name: pair for name, pair in optional.items() if pair[1] is not None}
    )
    mismatches = [name for name, pair in expected.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError(
            "calibration profile is not applicable to this score: "
            + ", ".join(mismatches)
        )


def _profile_probability(
    self: contract.CalibrationProfile,
    source_score: contract.EvidenceScore,
    value: object | None = None,
) -> contract.EvidenceScore:
    probability = self.expected_probability(source_score)
    if value is not None:
        claimed = contract._strict_float(value, name="calibrated probability")
        if not math.isclose(claimed, probability, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("claimed probability does not match the registered transform")
    metadata = {
        **dict(contract._thaw_json(source_score.provenance.metadata)),
        "sourceSemantics": source_score.semantics.value,
        "sourceScoreDigest": source_score.digest,
        "sourceHigherIsBetter": source_score.higher_is_better,
        "sourceRuntime": source_score.provenance.runtime,
        "sourceRuntimeVersion": source_score.provenance.runtime_version,
        "sourceInputEvidenceDigest": source_score.provenance.input_evidence_digest,
        "calibrationDatasetDigest": self.dataset_split_digest,
        "calibrationSplit": self.split_name,
        "calibrationProfileDigest": self.digest,
        "calibrationMethod": self.method,
        "calibrationName": self.name,
    }
    return contract.EvidenceScore(
        probability,
        semantics=contract.ScoreSemantics.PROBABILITY,
        provenance=contract.ScoreProvenance(
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
