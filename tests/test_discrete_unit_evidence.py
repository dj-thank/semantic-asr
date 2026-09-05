from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.discrete_unit_evidence import (
    CentroidDistanceTable,
    CentroidDTWFeatures,
    CollapsedUnitSequence,
    DiscreteTokenLanguageModel,
    DiscreteUnitAcousticRanker,
    DiscreteUnitCandidateScore,
    DiscreteUnitSequence,
    DiscreteUnitSpace,
    DTWConfig,
    StaticTextToDiscreteUnitEncoder,
    SurprisalProfile,
    SurprisalThreshold,
    align_collapsed_units,
    centroid_dtw_features,
    pronunciation_feature_vector,
    transcript_guided_features,
)
from semantic_asr.score_types import ScoreSemantics


def _space(*, digest: str = "a" * 64) -> DiscreteUnitSpace:
    return DiscreteUnitSpace(
        encoder="fixture/hubert-ja",
        encoder_revision="1" * 40,
        layer=9,
        codebook_size=4,
        codebook_sha256=digest,
        language="ja",
    )


def _sequence(units: tuple[int, ...], *, space: DiscreteUnitSpace | None = None):
    return DiscreteUnitSequence(units=units, space=space or _space())


def _table(space: DiscreteUnitSpace | None = None) -> CentroidDistanceTable:
    return CentroidDistanceTable.from_centroids(
        space or _space(),
        ((0.0,), (1.0,), (3.0,), (9.0,)),
    )


def _lm(space: DiscreteUnitSpace | None = None) -> DiscreteTokenLanguageModel:
    resolved = space or _space()
    return DiscreteTokenLanguageModel.fit(
        (
            _sequence((0, 1, 2, 0, 1, 2), space=resolved),
            _sequence((0, 1, 2, 0, 1, 2), space=resolved),
            _sequence((0, 1, 2, 1, 2), space=resolved),
        ),
        order=3,
    )


def test_collapse_preserves_raw_frame_projection() -> None:
    sequence = _sequence((0, 0, 1, 1, 1, 3))
    collapsed = sequence.collapse()
    assert collapsed.units == (0, 1, 3)
    assert collapsed.run_lengths == (2, 3, 1)
    assert collapsed.frame_to_collapsed == (0, 0, 1, 1, 1, 2)
    assert collapsed.raw_length == len(sequence.units)


def test_native_token_lm_surprisal_is_typed_as_candidate_independent() -> None:
    space = _space()
    lm = _lm(space)
    native = (
        _sequence((0, 1, 2, 0, 1, 2), space=space),
        _sequence((0, 1, 2, 1, 2), space=space),
    )
    threshold = lm.fit_spike_threshold(native, quantile=0.90)
    familiar = lm.profile(native[0], threshold=threshold)
    anomalous = lm.profile(_sequence((0, 3, 0, 3, 2, 3), space=space), threshold=threshold)

    assert threshold.quantile == pytest.approx(0.90)
    assert anomalous.mean_bits > familiar.mean_bits
    assert anomalous.spike_rate > familiar.spike_rate
    evidence = anomalous.as_uncertainty_evidence()
    assert evidence.semantics == ScoreSemantics.UNCALIBRATED_SCORE
    assert evidence.provenance.metadata["candidateIndependent"] is True


def test_centroid_dtw_distinguishes_close_and_distant_substitutions() -> None:
    space = _space()
    table = _table(space)
    canonical = _sequence((0,), space=space).collapse()
    close = _sequence((1,), space=space).collapse()
    distant = _sequence((3,), space=space).collapse()

    close_alignment = align_collapsed_units(canonical, close, distance_table=table)
    distant_alignment = align_collapsed_units(canonical, distant, distance_table=table)

    assert close_alignment.normalized_cost == pytest.approx(1.0)
    assert distant_alignment.normalized_cost == pytest.approx(9.0)
    assert close_alignment.normalized_cost < distant_alignment.normalized_cost


def test_dtw_normalizes_by_path_length_and_has_deterministic_projection() -> None:
    space = _space()
    table = _table(space)
    canonical = _sequence((0, 1), space=space).collapse()
    observed = _sequence((0, 2, 1), space=space).collapse()
    config = DTWConfig(projection="mean")
    alignment = align_collapsed_units(
        canonical,
        observed,
        distance_table=table,
        config=config,
    )

    assert alignment.normalized_cost == pytest.approx(alignment.total_cost / alignment.path_length)
    projected_costs, projected_mismatches = alignment.project_to_observed("mean")
    assert len(projected_costs) == len(observed.units)
    assert len(projected_mismatches) == len(observed.units)
    assert all(math.isfinite(value) for value in projected_costs)


def test_transcript_guided_features_return_frame_rate_mismatch_statistics() -> None:
    space = _space()
    observed = _sequence((0, 0, 1, 1, 3, 3), space=space)
    canonical = _sequence((0, 1, 2), space=space)
    features = transcript_guided_features(
        observed,
        canonical,
        token_lm=_lm(space),
        distance_table=_table(space),
        alpha=0.5,
    )

    assert features.dtw_distance == pytest.approx(2.0)
    assert features.token_mismatch_rate == pytest.approx(1 / 3)
    assert features.mismatch_frame_count == 2
    assert features.canonical_collapsed_units == 3
    assert features.observed_collapsed_units == 3
    assert features.weighted_surprisal_std_bits >= 0


def test_acoustic_ranker_uses_only_negative_dtw_for_zero_shot_ordering() -> None:
    space = _space()
    observed = _sequence((0, 0, 1, 1, 2, 2), space=space)
    encoder = StaticTextToDiscreteUnitEncoder(
        {
            "正しい候補": (0, 1, 2),
            "遠い候補": (0, 1, 3),
        },
        space=space,
        revision="fixture-v1",
    )
    ranker = DiscreteUnitAcousticRanker(
        observed=observed,
        distance_table=_table(space),
        text_encoder=encoder,
    )
    candidates = (
        CandidateEvidence(candidate_id="good", text="正しい候補"),
        CandidateEvidence(candidate_id="bad", text="遠い候補"),
    )

    scores = ranker.score(candidates)
    detailed = {row.candidate_id: row for row in ranker.score_detailed(candidates)}

    assert scores["good"] > scores["bad"]
    assert detailed["good"].alignment_cost.semantics == ScoreSemantics.COST
    assert detailed["good"].rank_score.semantics == ScoreSemantics.UNCALIBRATED_SCORE
    assert detailed["good"].rank_score.value == -detailed["good"].alignment_cost.value
    assert isinstance(detailed["good"].features, CentroidDTWFeatures)
    assert detailed["good"].includes_surprisal_features is False
    assert (
        detailed["good"].rank_score.provenance.metadata[
            "candidateIndependentSurprisalUsedForRanking"
        ]
        is False
    )


def test_unit_space_mismatch_fails_closed() -> None:
    first = _space(digest="a" * 64)
    second = _space(digest="b" * 64)
    with pytest.raises(ValueError, match="different discrete-unit space"):
        transcript_guided_features(
            _sequence((0, 1), space=first),
            _sequence((0, 1), space=second),
            token_lm=_lm(first),
            distance_table=_table(first),
        )


def test_dtw_resource_guard_fails_before_allocating_oversized_grid() -> None:
    space = _space()
    canonical = _sequence((0, 1, 2, 3), space=space).collapse()
    observed = _sequence((0, 1, 2, 3), space=space).collapse()
    with pytest.raises(ValueError, match="exceeding max_cells"):
        align_collapsed_units(
            canonical,
            observed,
            distance_table=_table(space),
            config=DTWConfig(max_cells=15),
        )


def test_feature_vector_keeps_audio_and_candidate_features_explicit() -> None:
    space = _space()
    lm = _lm(space)
    observed = _sequence((0, 0, 1, 1, 3, 3), space=space)
    threshold = lm.fit_spike_threshold(
        (_sequence((0, 1, 2, 0, 1, 2), space=space),),
    )
    audio = lm.profile(observed, threshold=threshold)
    transcript = transcript_guided_features(
        observed,
        _sequence((0, 1, 2), space=space),
        token_lm=lm,
        distance_table=_table(space),
    )

    audio_only = pronunciation_feature_vector(audio)
    combined = pronunciation_feature_vector(audio, transcript)
    assert set(audio_only) == {"surprisal_std_bits", "spike_rate", "duration_units"}
    assert set(combined) == {
        *audio_only,
        "dtw_distance",
        "token_mismatch_rate",
        "mismatch_surprisal_std_bits",
        "weighted_surprisal_std_bits",
    }


def test_distance_table_digest_is_bound_to_matrix_and_space() -> None:
    first = _table(_space(digest="a" * 64))
    second = _table(_space(digest="b" * 64))
    assert len(first.matrix_sha256) == 64
    assert first.digest != second.digest


def test_ranker_records_surprisal_features_only_when_native_lm_is_supplied() -> None:
    space = _space()
    observed = _sequence((0, 0, 1, 1, 2, 2), space=space)
    encoder = StaticTextToDiscreteUnitEncoder(
        {"候補": (0, 1, 2)},
        space=space,
        revision="fixture-v1",
    )
    ranker = DiscreteUnitAcousticRanker(
        observed=observed,
        token_lm=_lm(space),
        distance_table=_table(space),
        text_encoder=encoder,
    )
    row = ranker.score_detailed((CandidateEvidence(candidate_id="candidate", text="候補"),))[0]
    assert row.includes_surprisal_features is True
    assert row.rank_score.value == pytest.approx(-row.features.dtw_distance)
    assert row.rank_score.provenance.metadata["candidateFeatureSet"] == ("centroid-dtw-surprisal")


def test_path_normalized_dtw_is_orientation_invariant_for_equal_cost_ties() -> None:
    space = DiscreteUnitSpace(
        encoder="fixture/ssl",
        encoder_revision="revision-1",
        layer=0,
        codebook_size=3,
        codebook_sha256="a" * 64,
    )
    table = CentroidDistanceTable.from_centroids(space, ((0.0,), (1.0,), (2.0,)))
    first = _sequence((0, 2, 0), space=space).collapse()
    second = _sequence((0, 1, 0, 2), space=space).collapse()

    forward = align_collapsed_units(first, second, distance_table=table)
    reverse = align_collapsed_units(second, first, distance_table=table)

    assert forward.total_cost == pytest.approx(reverse.total_cost)
    assert forward.path_length == reverse.path_length == 4
    assert forward.normalized_cost == pytest.approx(reverse.normalized_cost)


def test_centroid_dtw_features_do_not_require_a_native_token_lm() -> None:
    space = _space()
    features = centroid_dtw_features(
        _sequence((0, 0, 1, 1, 2), space=space),
        _sequence((0, 1, 3), space=space),
        distance_table=_table(space),
    )
    assert features.dtw_distance > 0
    assert features.alignment_path_length >= 3
    assert len(features.digest) == 64


def test_static_text_encoder_requires_revision_and_does_not_leak_missing_text() -> None:
    space = _space()
    with pytest.raises(ValueError, match="immutable text encoder revision"):
        StaticTextToDiscreteUnitEncoder({"候補": (0,)}, space=space, revision="")
    encoder = StaticTextToDiscreteUnitEncoder(
        {"候補": (0,)},
        space=space,
        revision="fixture-v1",
    )
    secret = "非公開の候補名"
    with pytest.raises(KeyError) as caught:
        encoder.encode(secret)
    assert secret not in str(caught.value)
    with pytest.raises(TypeError):
        encoder.mapping["追加"] = (1,)  # type: ignore[index]


def test_collapsed_unit_sequence_rejects_noncanonical_projection() -> None:
    sequence = _sequence((0, 0, 1))
    digest = sequence.digest
    with pytest.raises(ValueError, match="exactly follow run_lengths"):
        CollapsedUnitSequence(
            units=(0, 1),
            frame_to_collapsed=(0, 1, 1),
            run_lengths=(2, 1),
            space=sequence.space,
            source_sequence_digest=digest,
        )
    with pytest.raises(ValueError, match="consecutive duplicates"):
        CollapsedUnitSequence(
            units=(0, 0),
            frame_to_collapsed=(0, 1),
            run_lengths=(1, 1),
            space=sequence.space,
            source_sequence_digest=digest,
        )


def test_token_lm_count_tables_are_immutable() -> None:
    model = _lm()
    with pytest.raises(TypeError):
        model.counts[0][(0,)] = 100  # type: ignore[index]


def test_surprisal_profile_exposes_explicit_routing_features() -> None:
    space = _space()
    model = _lm(space)
    sequence = _sequence((0, 1, 3, 2), space=space)
    threshold = model.fit_spike_threshold((sequence,))
    profile = model.profile(sequence, threshold=threshold)
    assert profile.routing_features() == {
        "discrete_surprisal_std_bits": profile.std_bits,
        "discrete_spike_rate": profile.spike_rate,
        "discrete_duration_units": float(profile.duration_units),
    }


def test_unit_space_and_sequence_normalize_identity_fields() -> None:
    space = DiscreteUnitSpace(
        encoder=" fixture/hubert-ja ",
        encoder_revision=" revision-1 ",
        layer=9,
        codebook_size=4,
        codebook_sha256="A" * 64,
        language=" JA ",
    )
    sequence = DiscreteUnitSequence(
        units=(0, 1),
        space=space,
        frame_hop_ms=20,
        source_sha256="B" * 64,
    )
    assert space.encoder == "fixture/hubert-ja"
    assert space.encoder_revision == "revision-1"
    assert space.codebook_sha256 == "a" * 64
    assert space.language == "ja"
    assert sequence.frame_hop_ms == 20.0
    assert sequence.source_sha256 == "b" * 64
    assert DiscreteUnitSpace.from_dict(space.as_dict()) == space
    assert DiscreteUnitSequence.from_dict(sequence.as_dict()) == sequence


def test_token_lm_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    model = _lm()
    path = tmp_path / "native-token-lm.json"
    model.save(path)
    loaded = DiscreteTokenLanguageModel.load(path)
    assert loaded.digest == model.digest
    assert loaded.as_dict() == model.as_dict()

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"]["counts"][0][0][1] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent|digest mismatch"):
        DiscreteTokenLanguageModel.load(path)


def test_distance_table_artifact_round_trip_and_immutable_matrix(tmp_path) -> None:
    space = _space()
    table = CentroidDistanceTable(
        space=space,
        distances=[
            [0, 1, 3, 9],
            [1, 0, 2, 8],
            [3, 2, 0, 6],
            [9, 8, 6, 0],
        ],
    )
    assert isinstance(table.distances, tuple)
    assert isinstance(table.distances[0], tuple)
    with pytest.raises(TypeError):
        table.distances[0][1] = 99.0  # type: ignore[index]

    path = tmp_path / "centroid-distances.json"
    table.save(path)
    loaded = CentroidDistanceTable.load(path)
    assert loaded.digest == table.digest
    assert loaded.distances == table.distances

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"]["distances"][0][1] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="symmetric|digest mismatch"):
        CentroidDistanceTable.load(path)


def test_zero_shot_ranker_does_not_require_native_token_lm() -> None:
    space = _space()
    encoder = StaticTextToDiscreteUnitEncoder(
        {"近い": (0, 1, 2), "遠い": (0, 1, 3)},
        space=space,
        revision="fixture-v1",
    )
    ranker = DiscreteUnitAcousticRanker(
        observed=_sequence((0, 0, 1, 1, 2, 2), space=space),
        distance_table=_table(space),
        text_encoder=encoder,
    )
    rows = ranker.score_detailed(
        (
            CandidateEvidence(candidate_id="near", text="近い"),
            CandidateEvidence(candidate_id="far", text="遠い"),
        )
    )
    assert rows[0].includes_surprisal_features is False
    assert rows[0].rank_score.value > rows[1].rank_score.value
    assert rows[0].rank_score.provenance.metadata["tokenLmDigest"] is None
    assert rows[0].rank_score.provenance.metadata["candidateFeatureSet"] == "centroid-dtw"


def test_transcript_features_bind_distance_table_and_dtw_configuration() -> None:
    space = _space()
    table = _table(space)
    config = DTWConfig(max_cells=100, projection="maximum")
    features = transcript_guided_features(
        _sequence((0, 0, 1, 2), space=space),
        _sequence((0, 1, 3), space=space),
        token_lm=_lm(space),
        distance_table=table,
        config=config,
    )
    assert features.distance_table_digest == table.digest
    assert features.config_digest == config.digest
    assert features.as_dict()["distanceTableDigest"] == table.digest
    assert features.as_dict()["dtwConfigDigest"] == config.digest


def test_invalid_boolean_integer_configuration_fails_closed() -> None:
    with pytest.raises(TypeError, match="layer must be an integer"):
        DiscreteUnitSpace(
            encoder="fixture",
            encoder_revision="1",
            layer=True,
            codebook_size=4,
            codebook_sha256="a" * 64,
        )
    with pytest.raises(TypeError, match="max_cells must be an integer"):
        DTWConfig(max_cells=True)


def test_surprisal_threshold_artifact_round_trip_and_binding(tmp_path) -> None:
    space = _space()
    model = _lm(space)
    sequence = _sequence((0, 1, 2, 0, 1, 2), space=space)
    threshold = model.fit_spike_threshold((sequence,), quantile=0.90)
    path = tmp_path / "surprisal-threshold.json"
    threshold.save(path)
    loaded = SurprisalThreshold.load(path)
    assert loaded == threshold
    assert loaded.token_lm_digest == model.digest
    assert loaded.unit_space_digest == space.digest

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"]["unitSpaceDigest"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        SurprisalThreshold.load(path)


def test_surprisal_profile_rejects_inconsistent_spike_rate() -> None:
    with pytest.raises(ValueError, match="spike_rate must match"):
        SurprisalProfile(
            token_surprisal_bits=(1.0, 10.0),
            mean_bits=5.5,
            std_bits=4.5,
            spike_rate=0.0,
            duration_units=2,
            threshold_bits=9.0,
            sequence_digest="a" * 64,
            token_lm_digest="b" * 64,
            threshold_digest="c" * 64,
        )


def test_token_lm_loader_rejects_unknown_schema_and_duplicate_rows(tmp_path) -> None:
    model = _lm()
    row = model.as_dict()
    row["schemaVersion"] = "discrete-token-lm-v2"
    with pytest.raises(ValueError, match="unsupported discrete token LM schema"):
        DiscreteTokenLanguageModel.from_dict(row)

    path = tmp_path / "duplicate-token-lm.json"
    model.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact"]["counts"][0].append(payload["artifact"]["counts"][0][0])
    # Keep the digest of the canonical model: the loader must reject duplicate
    # serialized keys before dictionary construction can silently collapse them.
    payload["artifactSha256"] = model.digest
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate serialized keys"):
        DiscreteTokenLanguageModel.load(path)


def test_ranker_rejects_text_encoder_output_not_bound_to_candidate() -> None:
    space = _space()

    class UnboundEncoder:
        name = "unbound-text2dunit"
        revision = "fixture-v1"
        configuration_digest = "d" * 64

        def __init__(self, unit_space: DiscreteUnitSpace) -> None:
            self.space = unit_space

        def encode(self, text: str) -> DiscreteUnitSequence:
            del text
            return DiscreteUnitSequence(
                units=(0, 1, 2),
                space=self.space,
                source_sha256="e" * 64,
            )

    ranker = DiscreteUnitAcousticRanker(
        observed=_sequence((0, 1, 2), space=space),
        distance_table=_table(space),
        text_encoder=UnboundEncoder(space),
    )
    with pytest.raises(ValueError, match="source SHA-256"):
        ranker.score_detailed((CandidateEvidence(candidate_id="candidate", text="候補"),))


def test_candidate_score_binds_features_and_provenance() -> None:
    space = _space()
    encoder = StaticTextToDiscreteUnitEncoder(
        {"候補": (0, 1, 2)},
        space=space,
        revision="fixture-v1",
    )
    row = DiscreteUnitAcousticRanker(
        observed=_sequence((0, 1, 2), space=space),
        distance_table=_table(space),
        text_encoder=encoder,
    ).score_detailed((CandidateEvidence(candidate_id="candidate", text="候補"),))[0]

    with pytest.raises(ValueError, match="attached DTW features"):
        DiscreteUnitCandidateScore(
            candidate_id=row.candidate_id,
            alignment_cost=row.alignment_cost,
            rank_score=row.rank_score,
            features=replace(row.features, dtw_distance=row.features.dtw_distance + 1.0),
        )

    with pytest.raises(ValueError, match="identical provenance"):
        DiscreteUnitCandidateScore(
            candidate_id=row.candidate_id,
            alignment_cost=row.alignment_cost,
            rank_score=replace(
                row.rank_score,
                provenance=replace(row.rank_score.provenance, scorer="different-scorer"),
            ),
            features=row.features,
        )

    with pytest.raises(ValueError, match="bind the attached candidate features"):
        DiscreteUnitCandidateScore(
            candidate_id=row.candidate_id,
            alignment_cost=row.alignment_cost,
            rank_score=row.rank_score,
            features=replace(row.features, alignment_digest="f" * 64),
        )


def test_unit_space_loader_rejects_unknown_or_missing_schema() -> None:
    row = _space().as_dict()
    row["schemaVersion"] = "discrete-unit-space-v2"
    with pytest.raises(ValueError, match="unsupported discrete-unit space schema"):
        DiscreteUnitSpace.from_dict(row)
    row.pop("schemaVersion")
    with pytest.raises(ValueError, match="unsupported discrete-unit space schema"):
        DiscreteUnitSpace.from_dict(row)


def test_boolean_real_number_configuration_fails_closed() -> None:
    space = _space()
    with pytest.raises(TypeError, match="frame_hop_ms must be a real number"):
        DiscreteUnitSequence(units=(0,), space=space, frame_hop_ms=True)
    with pytest.raises(TypeError, match="add_k must be a real number"):
        DiscreteTokenLanguageModel.fit((_sequence((0, 1), space=space),), add_k=True)
    model = _lm(space)
    with pytest.raises(TypeError, match="quantile must be a real number"):
        model.fit_spike_threshold((_sequence((0, 1), space=space),), quantile=True)
    with pytest.raises(TypeError, match="numeric square matrix"):
        CentroidDistanceTable(
            space=space,
            distances=((False, 1, 3, 9), (1, 0, 2, 8), (3, 2, 0, 6), (9, 8, 6, 0)),
        )
    encoder = StaticTextToDiscreteUnitEncoder(
        {"候補": (0, 1)},
        space=space,
        revision="fixture-v1",
    )
    with pytest.raises(TypeError, match="alpha must be a real number"):
        DiscreteUnitAcousticRanker(
            observed=_sequence((0, 1), space=space),
            distance_table=_table(space),
            text_encoder=encoder,
            alpha=True,
        )
