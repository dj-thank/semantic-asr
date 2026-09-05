"""Public canonical numeric score contract.

The stable import surface is kept in this module. The implementation core remains
separate so strict untrusted-input guards can be installed before any class is
re-exported; all callers still receive exactly one ``EvidenceScore`` class.
"""

from __future__ import annotations

from . import _score_contract_base as _base
from ._score_contract_registry_guards import install as _install_strict_score_guards

_install_strict_score_guards()

SCORE_SCHEMA_VERSION = _base.SCORE_SCHEMA_VERSION
LEGACY_RICH_SCHEMA_VERSION = _base.LEGACY_RICH_SCHEMA_VERSION
LEGACY_SIMPLE_SCHEMA_VERSION = _base.LEGACY_SIMPLE_SCHEMA_VERSION
SCORE_DOMAIN_SCHEMA_VERSION = _base.SCORE_DOMAIN_SCHEMA_VERSION
CALIBRATION_PROFILE_SCHEMA_VERSION = _base.CALIBRATION_PROFILE_SCHEMA_VERSION
CALIBRATION_REGISTRY_SCHEMA_VERSION = _base.CALIBRATION_REGISTRY_SCHEMA_VERSION

ScoreContractError = _base.ScoreContractError
ScoreMigrationError = _base.ScoreMigrationError
ScoreSemantics = _base.ScoreSemantics
ScoreNormalization = _base.ScoreNormalization
ScoreKind = _base.ScoreKind
FrozenDict = _base.FrozenDict
ScoreDomain = _base.ScoreDomain
ScoreProvenance = _base.ScoreProvenance
EvidenceScore = _base.EvidenceScore
CalibrationProfile = _base.CalibrationProfile
CalibrationProfileRegistry = _base.CalibrationProfileRegistry
require_sha256 = _base.require_sha256
require_same_score_domain = _base.require_same_score_domain

__all__ = [
    "CALIBRATION_PROFILE_SCHEMA_VERSION",
    "CALIBRATION_REGISTRY_SCHEMA_VERSION",
    "LEGACY_RICH_SCHEMA_VERSION",
    "LEGACY_SIMPLE_SCHEMA_VERSION",
    "SCORE_DOMAIN_SCHEMA_VERSION",
    "SCORE_SCHEMA_VERSION",
    "CalibrationProfile",
    "CalibrationProfileRegistry",
    "EvidenceScore",
    "FrozenDict",
    "ScoreContractError",
    "ScoreDomain",
    "ScoreKind",
    "ScoreMigrationError",
    "ScoreNormalization",
    "ScoreProvenance",
    "ScoreSemantics",
    "require_same_score_domain",
    "require_sha256",
]
