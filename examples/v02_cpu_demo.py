from __future__ import annotations

import json

from semantic_asr.candidate_pool import CandidatePath, CandidatePool
from semantic_asr.mbr import decode_mbr
from semantic_asr.ngram import CountNGramLanguageModel, CountNGramScorer
from semantic_asr.reranking import (
    FeatureSchema,
    FeatureVector,
    RankingGroup,
    TrainingCandidate,
    train_constrained_linear_reranker,
)
from semantic_asr.risk_control import (
    AdaptivePolicy,
    PolicyOutcome,
    RuntimeRiskState,
    fit_risk_control,
    select_policy,
)
from semantic_asr.sequence_scorers import TextCandidate


def build_pool() -> CandidatePool:
    return CandidatePool.from_paths(
        (
            CandidatePath(
                path_id="beam-1",
                text="今日は東京に行きます",
                cumulative_log_likelihood=-1.0,
                token_ids=(1, 2, 3),
                source="demo-whisper",
            ),
            CandidatePath(
                path_id="beam-2",
                text="今日は東京に行きます",
                cumulative_log_likelihood=-1.8,
                token_ids=(1, 4, 3),
                source="demo-whisper",
            ),
            CandidatePath(
                path_id="beam-3",
                text="今日は東京へ行きます",
                cumulative_log_likelihood=-1.2,
                token_ids=(1, 5, 3),
                source="demo-whisper",
            ),
            CandidatePath(
                path_id="second-ear-1",
                text="今日は京都に行きます",
                cumulative_log_likelihood=-1.9,
                token_ids=(7, 8, 9),
                source="demo-second-ear",
            ),
        )
    )


def train_demo_reranker() -> tuple[object, object, FeatureSchema]:
    schema = FeatureSchema(
        names=("acoustic", "ngram", "path_mass", "critical_risk"),
        monotonicity={
            "acoustic": 1,
            "ngram": 1,
            "path_mass": 1,
            "critical_risk": -1,
        },
    )

    def features(
        acoustic: float,
        ngram: float,
        path_mass: float,
        critical_risk: float,
    ) -> FeatureVector:
        return FeatureVector.create(
            schema,
            {
                "acoustic": acoustic,
                "ngram": ngram,
                "path_mass": path_mass,
                "critical_risk": critical_risk,
            },
        )

    groups = (
        RankingGroup(
            group_id="demo-train-1",
            candidates=(
                TrainingCandidate("correct", features(2.0, 1.8, 0.6, 0.0), 0.0),
                TrainingCandidate("near", features(1.2, 1.4, 0.0, 0.2), 0.25),
                TrainingCandidate("wrong-city", features(0.2, 0.7, 0.0, 1.0), 0.9),
            ),
        ),
        RankingGroup(
            group_id="demo-train-2",
            candidates=(
                TrainingCandidate("correct", features(2.2, 1.9, 0.4, 0.0), 0.0),
                TrainingCandidate("near", features(1.1, 1.2, 0.0, 0.3), 0.35),
                TrainingCandidate("wrong-number", features(0.5, 0.9, 0.0, 1.0), 1.0),
            ),
        ),
        RankingGroup(
            group_id="demo-train-3",
            candidates=(
                TrainingCandidate("correct", features(1.9, 1.7, 0.7, 0.0), 0.0),
                TrainingCandidate("near", features(1.0, 1.1, 0.0, 0.2), 0.3),
                TrainingCandidate("negation", features(0.3, 0.8, 0.0, 1.0), 1.0),
            ),
        ),
    )
    model, report = train_constrained_linear_reranker(
        groups,
        schema=schema,
        objective="hybrid",
        epochs=160,
        seed=13,
    )
    return model, report, schema


def main() -> int:
    pool = build_pool()
    text_candidates = [
        TextCandidate(candidate.candidate_id, candidate.text)
        for candidate in pool.candidates
    ]
    ngram_model = CountNGramLanguageModel.fit(
        (
            "今日は東京に行きます",
            "今日は東京に行きます",
            "明日は東京へ行きます",
            "東京で会議をします",
        ),
        order=4,
    )
    ngram_scores = {
        row.candidate_id: row.average.value
        for row in CountNGramScorer(ngram_model).score(text_candidates)
    }
    acoustic_posterior = pool.posterior()
    mbr = decode_mbr(pool, posterior=acoustic_posterior)

    reranker, report, schema = train_demo_reranker()
    vectors: dict[str, FeatureVector] = {}
    for candidate in pool.candidates:
        critical_risk = float("京都" in candidate.text)
        vectors[candidate.candidate_id] = FeatureVector.create(
            schema,
            {
                "acoustic": candidate.aggregate_log_likelihood,
                "ngram": ngram_scores[candidate.candidate_id],
                "path_mass": candidate.path_mass_bonus,
                "critical_risk": critical_risk,
            },
        )
    reranked = reranker.rank(vectors)

    policies = (
        AdaptivePolicy("cpu-k1", candidate_count=1, stages=("acoustic",)),
        AdaptivePolicy(
            "cpu-k3-rerank",
            candidate_count=3,
            stages=("acoustic", "ngram", "mbr", "compact-reranker"),
            minimum_entropy=0.1,
            maximum_cost_ms=200.0,
        ),
        AdaptivePolicy(
            "cpu-k3-verified-fallback",
            candidate_count=3,
            stages=(
                "acoustic",
                "ngram",
                "mbr",
                "compact-reranker",
                "verifier",
            ),
            conservative_fallback=True,
        ),
    )
    outcomes: list[PolicyOutcome] = []
    for index in range(600):
        outcomes.extend(
            (
                PolicyOutcome("cpu-k1", 0.22, 25.0, f"cal-{index}"),
                PolicyOutcome(
                    "cpu-k3-rerank",
                    1.0 if index % 150 == 0 else 0.0,
                    90.0,
                    f"cal-{index}",
                ),
                PolicyOutcome(
                    "cpu-k3-verified-fallback",
                    1.0 if index % 300 == 0 else 0.0,
                    320.0,
                    f"cal-{index}",
                ),
            )
        )
    profile = fit_risk_control(
        policies,
        outcomes,
        target_risk=0.08,
        minimum_samples=100,
    )
    selected_policy = select_policy(
        profile,
        RuntimeRiskState(
            entropy=pool.diagnostics().normalized_path_entropy,
            posterior_margin=0.25,
            disagreement=0.20,
            evidence_coverage=0.85,
            semantic_criticality=0.75,
            available_candidates=len(pool.candidates),
        ),
        cost_budget_ms=250.0,
    )

    print(
        json.dumps(
            {
                "candidateDiagnostics": {
                    "pathCount": pool.diagnostics().path_count,
                    "surfaceCount": pool.diagnostics().surface_count,
                    "uniqueSurfaceRatio": pool.diagnostics().unique_surface_ratio,
                },
                "mbrSelected": mbr.selected.text,
                "rerankerSelected": next(
                    candidate.text
                    for candidate in pool.candidates
                    if candidate.candidate_id == reranked[0][0]
                ),
                "rerankerPairwiseAccuracy": report.pairwise_accuracy,
                "riskControlledPolicy": selected_policy.policy.policy_id,
                "riskUpperBound": selected_policy.bound.upper_risk,
                "modelDigest": reranker.digest,
                "riskProfileDigest": profile.digest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
