from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_asr.score_contract import (
    CALIBRATION_PROFILE_SCHEMA_VERSION,
    CALIBRATION_REGISTRY_SCHEMA_VERSION,
    LEGACY_RICH_SCHEMA_VERSION,
    LEGACY_SIMPLE_SCHEMA_VERSION,
    CalibrationProfile,
    CalibrationProfileRegistry,
    EvidenceScore,
    ScoreDomain,
    ScoreKind,
    ScoreMigrationError,
    ScoreNormalization,
    ScoreProvenance,
    ScoreSemantics,
    require_same_score_domain,
)
from semantic_asr.sequence_scoring import SequenceScore

CONFIG = "c" * 64
CONDITION = "b" * 64
INPUT_EVIDENCE = "e" * 64
DATASET = "d" * 64
CALIBRATION = "f" * 64


def source_score(*, higher_is_better: bool = True) -> EvidenceScore:
    domain = ScoreDomain(
        producer="fixture-ranker",
        model="fixture-model",
        revision="revision-r1",
        normalization=ScoreNormalization.NONE,
        input_condition_digest=CONDITION,
        configuration_digest=CONFIG,
    )
    return EvidenceScore.raw(
        0.25,
        semantics=ScoreSemantics.LOGIT,
        scorer="fixture-ranker",
        model="fixture-model",
        revision="revision-r1",
        runtime="fixture-runtime",
        runtime_version="1.0",
        score_domain=domain,
        configuration_digest=CONFIG,
        input_evidence_digest=INPUT_EVIDENCE,
        input_condition_digest=CONDITION,
        higher_is_better=higher_is_better,
    )


def profile(source: EvidenceScore) -> CalibrationProfile:
    return CalibrationProfile(
        calibration_digest=CALIBRATION,
        name="fixture-platt",
        method="platt",
        source_semantics=source.semantics,
        scorer=source.provenance.scorer,
        model=source.provenance.model,
        revision=source.provenance.revision,
        normalization=source.provenance.normalization,
        score_domain_digest=source.provenance.score_domain_digest,
        configuration_digest=source.provenance.configuration_digest,
        input_condition_digest=source.provenance.input_condition_digest,
        dataset_split_digest=DATASET,
        parameters={"slope": 1.0, "intercept": 0.0},
        metadata={
            "sourceHigherIsBetter": source.higher_is_better,
            "runtime": source.provenance.runtime,
            "runtimeVersion": source.provenance.runtime_version,
            "inputEvidenceDigest": source.provenance.input_evidence_digest,
        },
    )


def registry(row: CalibrationProfile) -> CalibrationProfileRegistry:
    return CalibrationProfileRegistry(
        profiles=(row,),
        registry_name="fixture-registry",
        revision="r1",
    )


@pytest.mark.parametrize("field", ["calibrated", "higherIsBetter"])
@pytest.mark.parametrize("invalid", [1, 0, "true", "false", None])
def test_canonical_boolean_fields_are_not_coerced(field: str, invalid: object) -> None:
    row = source_score().as_dict()
    row[field] = invalid
    with pytest.raises((TypeError, ScoreMigrationError)):
        EvidenceScore.from_dict(row)


def test_unknown_score_schema_and_schema_shape_mismatches_fail_closed() -> None:
    row = source_score().as_dict()
    row["schemaVersion"] = "semantic-asr.score.v999"
    with pytest.raises(ScoreMigrationError, match="unsupported score schema"):
        EvidenceScore.from_dict(row)

    with pytest.raises(ScoreMigrationError, match="does not match"):
        EvidenceScore.from_dict(
            {
                "schemaVersion": LEGACY_RICH_SCHEMA_VERSION,
                "value": 0.1,
                "kind": "logit",
                "source": "fixture",
            }
        )
    with pytest.raises(ScoreMigrationError, match="does not match"):
        EvidenceScore.from_dict(
            {
                "schemaVersion": LEGACY_SIMPLE_SCHEMA_VERSION,
                "value": 0.1,
                "semantics": "logit",
                "provenance": {"scorer": "fixture"},
            }
        )


def test_metadata_is_deeply_immutable_even_via_dict_base_methods() -> None:
    source = {"nested": {"values": [1, 2]}}
    score = EvidenceScore.raw(
        0.1,
        semantics=ScoreSemantics.LOGIT,
        scorer="fixture",
        metadata=source,
    )
    original_digest = score.digest
    source["nested"]["values"].append(3)
    assert score.metadata["nested"]["values"] == (1, 2)
    assert score.digest == original_digest
    with pytest.raises(TypeError):
        dict.__setitem__(score.metadata, "forged", True)  # type: ignore[arg-type]
    assert score.digest == original_digest


def test_recursive_metadata_is_rejected_instead_of_recursing_forever() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="recursive"):
        EvidenceScore.raw(
            0.1,
            semantics=ScoreSemantics.LOGIT,
            scorer="fixture",
            metadata=cyclic,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scorer", "other"),
        ("model", "other-model"),
        ("revision", "other-revision"),
        ("configuration_digest", "1" * 64),
        ("input_condition_digest", "2" * 64),
    ],
)
def test_score_domain_object_cannot_be_attached_to_inconsistent_score(
    field: str, value: object
) -> None:
    domain = ScoreDomain(
        producer="fixture",
        model="model",
        revision="r1",
        normalization=ScoreNormalization.MEAN_TOKEN,
        input_condition_digest=CONDITION,
        configuration_digest=CONFIG,
    )
    kwargs: dict[str, object] = {
        "value": -0.2,
        "semantics": ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
        "scorer": "fixture",
        "model": "model",
        "revision": "r1",
        "normalization": ScoreNormalization.MEAN_TOKEN,
        "score_domain": domain,
        "configuration_digest": CONFIG,
        "input_condition_digest": CONDITION,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="score domain"):
        EvidenceScore.raw(**kwargs)  # type: ignore[arg-type]


def test_semantics_and_normalization_must_be_compatible() -> None:
    with pytest.raises(ValueError, match="average_log_likelihood"):
        EvidenceScore.raw(
            -0.2,
            semantics=ScoreSemantics.AVERAGE_LOG_LIKELIHOOD,
            scorer="fixture",
        )
    with pytest.raises(ValueError, match="cumulative_log_likelihood"):
        EvidenceScore(
            -1.0,
            semantics=ScoreSemantics.CUMULATIVE_LOG_LIKELIHOOD,
            provenance=ScoreProvenance(
                scorer="fixture",
                normalization=ScoreNormalization.MEAN_TOKEN,
            ),
        )
    with pytest.raises(ValueError, match="bounded_utility"):
        EvidenceScore(
            0.4,
            semantics=ScoreSemantics.BOUNDED_UTILITY,
            provenance=ScoreProvenance(
                scorer="fixture",
                normalization=ScoreNormalization.NONE,
            ),
        )


def test_historical_sequence_normalization_aliases_remain_lossless() -> None:
    row = SequenceScore(
        candidate_id="candidate",
        sum_logprob=-4.0,
        average_logprob=-1.0,
        token_count=4,
        source="fixture-lm",
    )
    average = row.as_evidence(average=True)
    cumulative = row.as_evidence(average=False)
    assert average.kind is ScoreKind.LOG_LIKELIHOOD
    assert average.kind.value == "log_likelihood"
    assert average.provenance.normalization is ScoreNormalization.MEAN_TOKEN
    assert cumulative.provenance.normalization is ScoreNormalization.SEQUENCE


def test_calibration_serializers_reject_untrusted_shapes() -> None:
    row = profile(source_score()).as_dict()
    row["schemaVersion"] = "semantic-asr.calibration-profile.v999"
    with pytest.raises(ValueError, match="unsupported calibration profile schema"):
        CalibrationProfile.from_dict(row)

    registry_row = {
        "schemaVersion": CALIBRATION_REGISTRY_SCHEMA_VERSION,
        "registryName": "fixture",
        "revision": "r1",
        "profiles": [profile(source_score()).as_dict(), "not-an-object"],
    }
    with pytest.raises(TypeError, match="must be an object"):
        CalibrationProfileRegistry.from_dict(registry_row)

    bad_schema = dict(registry_row)
    bad_schema["profiles"] = [profile(source_score()).as_dict()]
    bad_schema["schemaVersion"] = "semantic-asr.calibration-registry.v999"
    with pytest.raises(ValueError, match="unsupported calibration registry schema"):
        CalibrationProfileRegistry.from_dict(bad_schema)


def test_probability_rejects_runtime_direction_and_input_evidence_rebinding() -> None:
    source = source_score()
    row = profile(source)
    probability = row.probability(source)
    trusted = registry(row)
    assert trusted.require_probability(probability, source_score=source) == pytest.approx(
        probability.value
    )

    attacks = (
        replace(probability.provenance, runtime="other-runtime"),
        replace(probability.provenance, runtime_version="2.0"),
        replace(probability.provenance, input_evidence_digest="1" * 64),
    )
    for forged_provenance in attacks:
        forged = EvidenceScore(
            probability.value,
            semantics=ScoreSemantics.PROBABILITY,
            provenance=forged_provenance,
            calibrated=True,
        )
        with pytest.raises(ValueError, match="provenance|receipt"):
            trusted.require_probability(forged, source_score=source)

    wrong_direction_source = source_score(higher_is_better=False)
    with pytest.raises(ValueError, match="higher_is_better"):
        row.probability(wrong_direction_source)


def test_domain_aggregation_rejects_direction_and_calibration_mixing() -> None:
    left = source_score()
    opposite = EvidenceScore.raw(
        left.value,
        semantics=left.semantics,
        scorer=left.source,
        model=left.provenance.model,
        revision=left.provenance.revision,
        runtime=left.provenance.runtime,
        runtime_version=left.provenance.runtime_version,
        normalization=left.provenance.normalization,
        score_domain_digest=left.provenance.score_domain_digest,
        configuration_digest=left.provenance.configuration_digest,
        input_evidence_digest=left.provenance.input_evidence_digest,
        input_condition_digest=left.provenance.input_condition_digest,
        higher_is_better=False,
    )
    with pytest.raises(ValueError, match="different directions"):
        require_same_score_domain((left, opposite))

    first_profile = profile(left)
    first = first_profile.probability(left)
    second_profile = replace(first_profile, calibration_digest="1" * 64)
    second = second_profile.probability(left)
    with pytest.raises(ValueError, match="different calibration"):
        require_same_score_domain((first, second))


def test_schema_constants_match_current_profile_versions() -> None:
    assert profile(source_score()).schema_version == CALIBRATION_PROFILE_SCHEMA_VERSION
    assert registry(profile(source_score())).schema_version == CALIBRATION_REGISTRY_SCHEMA_VERSION
