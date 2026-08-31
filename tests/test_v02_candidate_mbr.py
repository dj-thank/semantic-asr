import math

import pytest

from semantic_asr.adaptive import AdaptiveKConfig, select_adaptive_k
from semantic_asr.candidate_pool import aggregate_surface_candidates, logsumexp
from semantic_asr.cascade import run_candidate_cascade
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.mbr import minimum_bayes_risk, surface_loss


def _path(candidate_id: str, text: str, cumulative: float, domain: str) -> CandidateEvidence:
    token_ids = (1, 2)
    return CandidateEvidence(
        candidate_id,
        text,
        token_ids=token_ids,
        acoustic=cumulative / 3,
        avg_logprob=cumulative / 3,
        source="whisper",
        metadata={
            "scoreDomain": domain,
            "cumulativeLogprob": cumulative,
            "pathCount": 1,
        },
    )


def test_surface_pool_aggregates_probability_mass_inside_one_domain() -> None:
    pooled = aggregate_surface_candidates(
        [
            _path("p1", "同じ文", -1.0, "same"),
            _path("p2", "同じ文", -2.0, "same"),
        ],
        id_prefix="x",
    )
    assert len(pooled) == 1
    expected = logsumexp([-1.0, -2.0])
    assert pooled[0].metadata["aggregateCumulativeLogprob"] == pytest.approx(expected)
    assert pooled[0].avg_logprob == pytest.approx(expected / 3)
    assert pooled[0].metadata["pathCount"] == 2
    assert pooled[0].metadata["pathProbabilityMassAggregated"] is True


def test_surface_pool_never_adds_incomparable_score_domains() -> None:
    pooled = aggregate_surface_candidates(
        [
            _path("base", "同じ文", -1.0, "base-window"),
            _path("other", "同じ文", -0.5, "other-model"),
        ]
    )
    assert len(pooled) == 1
    assert len(pooled[0].metadata["scoreDomains"]) == 2
    assert pooled[0].metadata["surfacePathCount"] == 2
    assert pooled[0].metadata["aggregateCumulativeLogprob"] == pytest.approx(-0.5)


def test_mbr_can_select_a_consensus_candidate_instead_of_mode() -> None:
    candidates = [
        CandidateEvidence("a", "東京へ行く"),
        CandidateEvidence("b", "東京に行く"),
        CandidateEvidence("c", "京都に行く"),
    ]
    posterior = {"a": 0.40, "b": 0.35, "c": 0.25}
    decision = minimum_bayes_risk(
        candidates,
        posterior=posterior,
        loss=surface_loss,
    )
    assert decision.selected_candidate_id in {"a", "b"}
    assert math.isfinite(decision.expected_risk)
    assert decision.risk_margin >= 0


def test_adaptive_k_expands_under_risk_and_criticality() -> None:
    candidates = [CandidateEvidence(str(index), f"候補{index}") for index in range(8)]
    posterior = {
        candidate.candidate_id: probability
        for candidate, probability in zip(
            candidates,
            [0.32, 0.22, 0.15, 0.10, 0.08, 0.06, 0.04, 0.03],
            strict=True,
        )
    }
    low = select_adaptive_k(
        candidates,
        posterior,
        config=AdaptiveKConfig(minimum_k=2, maximum_k=8, posterior_mass_target=0.50),
    )
    high = select_adaptive_k(
        candidates,
        posterior,
        selective_risk=0.9,
        semantic_criticality=1.0,
        config=AdaptiveKConfig(minimum_k=2, maximum_k=8, posterior_mass_target=0.50),
    )
    assert low.k >= 2
    assert high.k > low.k
    assert high.cumulative_posterior >= low.cumulative_posterior


def test_cascade_reports_fusion_mbr_disagreement_without_inventing_text() -> None:
    candidates = [
        CandidateEvidence("a", "料金は3000円です", acoustic=0.55, mora=0.55),
        CandidateEvidence("b", "料金は30000円です", acoustic=0.54, mora=0.54),
        CandidateEvidence("c", "料金は3000円でした", acoustic=0.53, mora=0.53),
    ]
    decision = run_candidate_cascade(candidates)
    assert decision.selected_text in {candidate.text for candidate in candidates}
    assert decision.path_aggregated_candidate_count == 3
    assert decision.mbr.loss_name == "semantic-mbr"
