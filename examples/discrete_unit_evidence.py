"""Model-free contract demo for discrete-unit acoustic evidence.

The token IDs and centroids below are synthetic fixtures. Production use requires
frozen Audio2DUnit and Text2DUnit adapters trained against the same codebook.
"""

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.discrete_unit_evidence import (
    CentroidDistanceTable,
    DiscreteTokenLanguageModel,
    DiscreteUnitAcousticRanker,
    DiscreteUnitSequence,
    DiscreteUnitSpace,
    StaticTextToDiscreteUnitEncoder,
)


def main() -> None:
    space = DiscreteUnitSpace(
        encoder="fixture/hubert-ja",
        encoder_revision="1111111111111111111111111111111111111111",
        layer=9,
        codebook_size=4,
        codebook_sha256="a" * 64,
        language="ja",
    )
    native = (
        DiscreteUnitSequence((0, 1, 2, 0, 1, 2), space),
        DiscreteUnitSequence((0, 1, 2, 1, 2), space),
    )
    token_lm = DiscreteTokenLanguageModel.fit(native, order=3)
    threshold = token_lm.fit_spike_threshold(native, quantile=0.90)

    observed = DiscreteUnitSequence((0, 0, 1, 1, 2, 2), space)
    profile = token_lm.profile(observed, threshold=threshold)
    print("routing_features=", profile.routing_features())

    distances = CentroidDistanceTable.from_centroids(
        space,
        ((0.0,), (1.0,), (3.0,), (9.0,)),
    )
    text2unit = StaticTextToDiscreteUnitEncoder(
        {
            "正しい候補": (0, 1, 2),
            "音響的に遠い候補": (0, 1, 3),
        },
        space=space,
        revision="fixture-v1",
    )
    ranker = DiscreteUnitAcousticRanker(
        observed=observed,
        distance_table=distances,
        text_encoder=text2unit,
        token_lm=token_lm,
    )
    candidates = (
        CandidateEvidence(candidate_id="good", text="正しい候補"),
        CandidateEvidence(candidate_id="bad", text="音響的に遠い候補"),
    )
    for row in ranker.score_detailed(candidates):
        print(
            row.candidate_id,
            "rank_score=",
            row.rank_score.value,
            "dtw_cost=",
            row.alignment_cost.value,
            "surprisal_features=",
            row.includes_surprisal_features,
        )


if __name__ == "__main__":
    main()
