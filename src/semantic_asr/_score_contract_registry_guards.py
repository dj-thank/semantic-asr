"""Registry validation and strict score-contract installer."""

from __future__ import annotations

import math
from collections.abc import Iterable

from . import _score_contract_base as contract
from ._score_contract_calibration_guards import (
    _profile_assert_applicable,
    _profile_from_dict,
    _profile_probability,
    _registry_from_dict,
)
from ._score_contract_evidence_guards import _kind, _raw, _score_from_dict
from ._score_contract_primitives import (
    FrozenDict,
    ScoreKind,
    _NORMALIZATION_ALIASES,
    _freeze_json,
    _thaw_json,
)
from ._score_contract_provenance_guards import (
    _provenance_from_dict,
    _provenance_post_init,
    _score_domain_post_init,
    _validate_score,
)


def _registry_validate_probability(
    self: contract.CalibrationProfileRegistry,
    score: contract.EvidenceScore,
    *,
    source_score: contract.EvidenceScore | None = None,
) -> contract.CalibrationProfile:
    if score.semantics != contract.ScoreSemantics.PROBABILITY or not score.calibrated:
        raise ValueError("score is not a calibrated probability")
    if not score.higher_is_better:
        raise ValueError("correctness probability must be higher-is-better")
    profile = self.profile(score.provenance.calibration_digest)
    if source_score is None:
        raise ValueError("the source score is required to validate a calibrated probability")
    profile.assert_applicable(source_score)
    metadata = score.provenance.metadata
    checks = {
        "sourceSemantics": profile.source_semantics.value,
        "sourceScoreDigest": source_score.digest,
        "sourceHigherIsBetter": source_score.higher_is_better,
        "sourceRuntime": source_score.provenance.runtime,
        "sourceRuntimeVersion": source_score.provenance.runtime_version,
        "sourceInputEvidenceDigest": source_score.provenance.input_evidence_digest,
        "calibrationDatasetDigest": profile.dataset_split_digest,
        "calibrationSplit": profile.split_name,
        "calibrationProfileDigest": profile.digest,
    }
    mismatched_metadata = [
        key for key, wanted in checks.items() if metadata.get(key) != wanted
    ]
    if mismatched_metadata:
        raise ValueError(
            "probability calibration split/receipt metadata mismatch: "
            + ", ".join(mismatched_metadata)
        )
    direct = {
        "scorer": (score.provenance.scorer, source_score.provenance.scorer),
        "model": (score.provenance.model, source_score.provenance.model),
        "revision": (score.provenance.revision, source_score.provenance.revision),
        "runtime": (score.provenance.runtime, source_score.provenance.runtime),
        "runtime_version": (
            score.provenance.runtime_version,
            source_score.provenance.runtime_version,
        ),
        "normalization": (
            score.provenance.normalization,
            source_score.provenance.normalization,
        ),
        "score_domain_digest": (
            score.provenance.score_domain_digest,
            source_score.provenance.score_domain_digest,
        ),
        "configuration_digest": (
            score.provenance.configuration_digest,
            source_score.provenance.configuration_digest,
        ),
        "input_evidence_digest": (
            score.provenance.input_evidence_digest,
            source_score.provenance.input_evidence_digest,
        ),
        "input_condition_digest": (
            score.provenance.input_condition_digest,
            source_score.provenance.input_condition_digest,
        ),
    }
    mismatches = [name for name, pair in direct.items() if pair[0] != pair[1]]
    if mismatches:
        raise ValueError(
            "probability provenance does not match the supplied source score: "
            + ", ".join(mismatches)
        )
    expected_probability = profile.expected_probability(source_score)
    if not math.isclose(score.value, expected_probability, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "probability value does not match the registered calibration transform"
        )
    return profile


def _same_domain(scores: Iterable[contract.EvidenceScore]) -> str:
    rows = tuple(scores)
    if not rows:
        raise ValueError("at least one score is required")
    if any(not isinstance(score, contract.EvidenceScore) for score in rows):
        raise TypeError("score-domain aggregation requires EvidenceScore values")
    fields = {
        "semantics": {score.semantics for score in rows},
        "normalization": {score.provenance.normalization for score in rows},
        "direction": {score.higher_is_better for score in rows},
        "domain": {score.provenance.score_domain_digest for score in rows},
    }
    if len(fields["semantics"]) != 1:
        raise ValueError("scores with different semantics cannot be aggregated")
    if len(fields["normalization"]) != 1:
        raise ValueError("scores with different normalization cannot be aggregated")
    if len(fields["direction"]) != 1:
        raise ValueError("scores with different directions cannot be aggregated")
    if None in fields["domain"]:
        raise ValueError("score-domain digest is required for aggregation")
    if len(fields["domain"]) != 1:
        raise ValueError("scores from different score domains cannot be aggregated")
    if next(iter(fields["semantics"])) == contract.ScoreSemantics.PROBABILITY:
        receipts = {score.provenance.calibration_digest for score in rows}
        if None in receipts or len(receipts) != 1:
            raise ValueError(
                "probabilities from different calibration profiles cannot be aggregated"
            )
    result = next(iter(fields["domain"]))
    assert isinstance(result, str)
    return result


def install() -> None:
    """Install strict hooks exactly once before the public package is imported."""

    if getattr(contract, "_STRICT_BOUNDARY_HARDENING", False):
        return
    contract.FrozenDict = FrozenDict
    contract._freeze_json = _freeze_json
    contract._thaw_json = _thaw_json
    contract.ScoreKind = ScoreKind

    def _missing_normalization(
        cls: type[contract.ScoreNormalization], value: object
    ) -> contract.ScoreNormalization | None:
        if isinstance(value, str) and value in _NORMALIZATION_ALIASES:
            return _NORMALIZATION_ALIASES[value]
        return None

    contract.ScoreNormalization._missing_ = classmethod(_missing_normalization)
    contract.ScoreDomain.__post_init__ = _score_domain_post_init
    contract.ScoreProvenance.__post_init__ = _provenance_post_init
    contract.ScoreProvenance.from_dict = classmethod(_provenance_from_dict)
    contract.EvidenceScore._validate = _validate_score
    contract.EvidenceScore.raw = classmethod(_raw)
    contract.EvidenceScore.kind = property(_kind)
    contract.EvidenceScore.usable_as_probability = property(lambda _self: False)
    contract.EvidenceScore.from_dict = classmethod(_score_from_dict)
    contract.CalibrationProfile.from_dict = classmethod(_profile_from_dict)
    contract.CalibrationProfile.assert_applicable = _profile_assert_applicable
    contract.CalibrationProfile.probability = _profile_probability
    contract.CalibrationProfileRegistry.from_dict = classmethod(_registry_from_dict)
    contract.CalibrationProfileRegistry.validate_probability = _registry_validate_probability
    contract.require_same_score_domain = _same_domain
    contract.CALIBRATION_PROFILE_SCHEMA_VERSION = "semantic-asr.calibration-profile.v1"
    contract.CALIBRATION_REGISTRY_SCHEMA_VERSION = "semantic-asr.calibration-registry.v1"
    contract.SCORE_DOMAIN_SCHEMA_VERSION = "1"
    if hasattr(contract, "__all__") and "ScoreKind" not in contract.__all__:
        contract.__all__.append("ScoreKind")
    contract._STRICT_BOUNDARY_HARDENING = True
