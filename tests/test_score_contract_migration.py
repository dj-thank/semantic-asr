from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from semantic_asr import score_semantics, score_types
from semantic_asr.score_contract import (
    SCORE_SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationProfileRegistry,
    EvidenceScore,
    ScoreDomain,
    ScoreMigrationError,
    ScoreNormalization,
    ScoreProvenance,
    ScoreSemantics,
    require_same_score_domain,
)
from semantic_asr.score_types import CalibrationExample, PlattCalibrator

FIXTURES = Path(__file__).parent / "fixtures" / "score_contract"
DATASET = "d" * 64
CONFIG = "c" * 64
CONDITION = "b" * 64
SPAN = "e" * 64


def domain(
    *,
    model: str = "fixture-model",
    normalization: ScoreNormalization = ScoreNormalization.MEAN_TOKEN,
    condition: str = CONDITION,
) -> ScoreDomain:
    return ScoreDomain(
        producer="fixture-sequence-scorer",
        model=model,
        revision="a" * 40,
        normalization=normalization,
        input_condition_digest=condition,
        configuration_digest=CONFIG,
        span_digest=SPAN,
    )


def raw_score(
    *,
    model: str = "fixture-model",
    semantics: ScoreSemantics = ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
    normalization: ScoreNormalization = ScoreNormalization.MEAN_TOKEN,
    condition: str = CONDITION,
) -> EvidenceScore:
    score_domain = domain(
        model=model,
        normalization=normalization,
        condition=condition,
    )
    return EvidenceScore.raw(
        -0.75,
        semantics=semantics,
        scorer="fixture-sequence-scorer",
        model=model,
        revision="a" * 40,
        normalization=normalization,
        score_domain=score_domain,
        configuration_digest=CONFIG,
        metadata={"tokenCount": 4, "nested": {"labels": ["a", "b"]}},
    )


def profile(
    score: EvidenceScore,
    *,
    digest: str = "f" * 64,
    model: str | None = None,
    semantics: ScoreSemantics | None = None,
    normalization: ScoreNormalization | None = None,
    condition: str | None = None,
    dataset: str = DATASET,
) -> CalibrationProfile:
    return CalibrationProfile(
        calibration_digest=digest,
        name="fixture-platt",
        method="platt",
        source_semantics=semantics or score.semantics,
        scorer=score.provenance.scorer,
        model=score.provenance.model if model is None else model,
        revision=score.provenance.revision,
        normalization=normalization or score.provenance.normalization,
        score_domain_digest=score.provenance.score_domain_digest,
        configuration_digest=score.provenance.configuration_digest,
        input_condition_digest=(
            score.provenance.input_condition_digest if condition is None else condition
        ),
        dataset_split_digest=dataset,
    )


def registry(row: CalibrationProfile) -> CalibrationProfileRegistry:
    return CalibrationProfileRegistry(
        profiles=(row,),
        registry_name="fixture-registry",
        revision="r1",
    )


def test_legacy_import_paths_share_the_exact_canonical_class() -> None:
    assert score_semantics.EvidenceScore is EvidenceScore
    assert score_types.EvidenceScore is EvidenceScore
    assert score_types.ScoreProvenance is ScoreProvenance
    assert score_types.ScoreSemantics is ScoreSemantics


def test_calibrated_probability_requires_a_receipt_digest_at_construction() -> None:
    with pytest.raises(ValueError, match="receipt digest"):
        EvidenceScore(
            0.9,
            score_semantics.ScoreKind.PROBABILITY,
            "fixture",
            calibrated=True,
        )


def test_probability_requires_a_frozen_registered_profile() -> None:
    source = raw_score()
    row = profile(source)
    probability = row.probability(source, 0.8)

    with pytest.raises(ValueError, match="Registry"):
        probability.require_probability()
    with pytest.raises(ValueError, match="unknown calibration"):
        probability.require_probability(
            CalibrationProfileRegistry(
                profiles=(replace(row, calibration_digest="1" * 64),),
                registry_name="other",
                revision="r1",
            )
        )

    assert probability.require_probability(registry(row), source_score=source) == pytest.approx(
        0.8
    )


@pytest.mark.parametrize(
    ("field", "changed", "message"),
    [
        ("model", "another-model", "model"),
        (
            "normalization",
            ScoreNormalization.MEAN_FRAME,
            "normalization",
        ),
        ("condition", "1" * 64, "input_condition_digest"),
    ],
)
def test_probability_rejects_profile_applicability_mismatch(
    field: str,
    changed: object,
    message: str,
) -> None:
    source = raw_score()
    kwargs = {field: changed}
    wrong = profile(source, **kwargs)
    probability = wrong.probability(
        EvidenceScore.raw(
            source.value,
            semantics=source.semantics,
            scorer=source.provenance.scorer,
            model=wrong.model,
            revision=source.provenance.revision,
            normalization=wrong.normalization,
            score_domain_digest=source.provenance.score_domain_digest,
            configuration_digest=source.provenance.configuration_digest,
            input_condition_digest=wrong.input_condition_digest,
        ),
        0.8,
    )
    forged = EvidenceScore(
        probability.value,
        semantics=ScoreSemantics.PROBABILITY,
        provenance=replace(
            source.provenance,
            calibration_digest=wrong.calibration_digest,
            metadata=probability.provenance.metadata,
        ),
        calibrated=True,
    )
    with pytest.raises(ValueError, match=message):
        forged.require_probability(registry(wrong))


def test_probability_rejects_score_kind_and_calibration_split_mismatch() -> None:
    source = raw_score()
    wrong_kind = profile(source, semantics=ScoreSemantics.LOGIT)
    with pytest.raises(ValueError, match="not applicable.*semantics"):
        wrong_kind.probability(source, 0.5)

    row = profile(source)
    probability = row.probability(source, 0.8)
    tampered_metadata = {
        **probability.provenance.metadata,
        "calibrationDatasetDigest": "1" * 64,
    }
    tampered = EvidenceScore(
        probability.value,
        semantics=ScoreSemantics.PROBABILITY,
        provenance=replace(probability.provenance, metadata=tampered_metadata),
        calibrated=True,
    )
    with pytest.raises(ValueError, match="split"):
        tampered.require_probability(registry(row))


def test_bool_nan_inf_invalid_enum_and_mutable_metadata_are_rejected() -> None:
    with pytest.raises(TypeError, match="bool"):
        EvidenceScore.raw(
            True,
            semantics=ScoreSemantics.LOGIT,
            scorer="fixture",
        )
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            EvidenceScore.raw(
                value,
                semantics=ScoreSemantics.LOGIT,
                scorer="fixture",
            )
    with pytest.raises(ValueError):
        EvidenceScore(
            0.1,
            semantics="not-a-score-kind",
            provenance=ScoreProvenance(scorer="fixture"),
        )

    source_metadata = {"nested": {"items": [1, 2]}}
    score = EvidenceScore.raw(
        0.1,
        semantics=ScoreSemantics.LOGIT,
        scorer="fixture",
        metadata=source_metadata,
    )
    source_metadata["nested"]["items"].append(3)
    assert score.provenance.metadata["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError, match="immutable"):
        score.provenance.metadata["new"] = "value"


def test_aggregation_requires_identical_semantics_normalization_and_domain() -> None:
    left = raw_score()
    right = EvidenceScore.raw(
        -0.5,
        semantics=left.semantics,
        scorer=left.provenance.scorer,
        model=left.provenance.model,
        revision=left.provenance.revision,
        normalization=left.provenance.normalization,
        score_domain_digest=left.provenance.score_domain_digest,
        configuration_digest=left.provenance.configuration_digest,
        input_condition_digest=left.provenance.input_condition_digest,
    )
    assert require_same_score_domain((left, right)) == left.provenance.score_domain_digest

    other_domain = raw_score(model="another-model")
    with pytest.raises(ValueError, match="different score domains"):
        require_same_score_domain((left, other_domain))
    other_kind = EvidenceScore.raw(
        -0.5,
        semantics=ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
        scorer="fixture-sequence-scorer",
        normalization=ScoreNormalization.SEQUENCE,
        score_domain_digest=left.provenance.score_domain_digest,
    )
    with pytest.raises(ValueError, match="different semantics"):
        require_same_score_domain((left, other_kind))


def test_legacy_log_likelihood_requires_lossless_normalization() -> None:
    with pytest.raises(ScoreMigrationError, match="ambiguous"):
        EvidenceScore.from_dict(
            {
                "value": -0.2,
                "kind": "log_likelihood",
                "source": "unknown-legacy-producer",
                "calibrated": False,
            }
        )

    row = json.loads((FIXTURES / "legacy_simple_mean_frame_v1.json").read_text())
    score = EvidenceScore.from_dict(row)
    assert score.semantics == ScoreSemantics.AVERAGE_LOG_LIKELIHOOD
    assert score.provenance.normalization == ScoreNormalization.MEAN_FRAME
    restored = EvidenceScore.from_dict(score.to_legacy_simple_dict())
    assert restored.as_dict() == score.as_dict()


def test_legacy_rich_and_canonical_golden_roundtrip() -> None:
    legacy = json.loads((FIXTURES / "legacy_rich_v1.json").read_text())
    migrated = EvidenceScore.from_dict(legacy)
    assert migrated.semantics == ScoreSemantics.LOGIT
    assert EvidenceScore.from_dict(migrated.to_legacy_rich_dict()).as_dict() == migrated.as_dict()

    canonical = json.loads((FIXTURES / "canonical_v2.json").read_text())
    score = EvidenceScore.from_dict(canonical)
    assert score.schema_version == SCORE_SCHEMA_VERSION
    assert score.as_dict() == canonical
    assert score.digest == "0e01d03bfaf69ad3a013d1d65f82ac71ccbabdaa42564827bdb47693c7f19946"


def test_platt_calibrator_can_issue_a_registry_validated_probability() -> None:
    source = EvidenceScore.raw(
        0.25,
        semantics=ScoreSemantics.LOGIT,
        scorer="fixture-reranker",
        model="fixture-model",
        revision="a" * 40,
        normalization=ScoreNormalization.NONE,
        score_domain_digest="1" * 64,
        configuration_digest=CONFIG,
        input_condition_digest=CONDITION,
    )
    examples = [
        CalibrationExample(score=-2.0, correct=False),
        CalibrationExample(score=-1.0, correct=False),
        CalibrationExample(score=1.0, correct=True),
        CalibrationExample(score=2.0, correct=True),
    ]
    calibrator = PlattCalibrator.fit(
        examples,
        source_semantics=ScoreSemantics.LOGIT,
        dataset_digest=DATASET,
        iterations=50,
    )
    frozen_profile = calibrator.profile_for(source)
    probability = calibrator.probability(source, profile=frozen_profile)
    frozen_registry = registry(frozen_profile)

    assert frozen_registry.require_probability(
        probability, source_score=source
    ) == pytest.approx(probability.value)
    assert probability.provenance.calibration_digest == calibrator.digest


def test_legacy_calibrator_output_is_receipt_bearing_but_not_trusted() -> None:
    source = EvidenceScore.raw(
        0.25,
        semantics=ScoreSemantics.LOGIT,
        scorer="fixture-reranker",
    )
    examples = [
        CalibrationExample(score=-2.0, correct=False),
        CalibrationExample(score=-1.0, correct=False),
        CalibrationExample(score=1.0, correct=True),
        CalibrationExample(score=2.0, correct=True),
    ]
    calibrator = PlattCalibrator.fit(
        examples,
        source_semantics=ScoreSemantics.LOGIT,
        dataset_digest="legacy-human-label",
        iterations=20,
    )
    probability = calibrator.probability(source)
    assert probability.calibrated
    with pytest.raises(ValueError, match="Registry"):
        probability.require_probability()
    with pytest.raises(ValueError, match="SHA-256"):
        calibrator.profile_for(source)


def test_numeric_range_does_not_turn_a_preference_into_probability() -> None:
    score = EvidenceScore.raw(
        0.9,
        semantics=ScoreSemantics.PREFERENCE,
        scorer="chat-reranker",
    )
    assert not score.usable_as_probability
    with pytest.raises(ValueError, match="calibrated probability"):
        registry(profile(raw_score())).validate_probability(score)
