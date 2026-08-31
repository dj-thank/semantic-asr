from __future__ import annotations

import pytest

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.fusion import ACOUSTIC_FAMILY
from semantic_asr.learned_fusion import (
    FusionTrainingExample,
    LearnedFusionConfig,
    train_constrained_fusion,
)


def _examples() -> list[FusionTrainingExample]:
    output: list[FusionTrainingExample] = []
    for index in range(20):
        correct_id = f"correct-{index}"
        fluent_id = f"fluent-{index}"
        alternative_id = f"alternative-{index}"
        candidates = (
            CandidateEvidence(
                correct_id,
                "実際の発話",
                acoustic=0.90,
                mora=0.92,
                lexical=0.12,
                preservation=0.86,
                cross_model=0.82,
            ),
            CandidateEvidence(
                fluent_id,
                "文法的に自然な捏造",
                acoustic=0.18,
                mora=0.22,
                lexical=0.98,
                preservation=0.35,
                cross_model=0.20,
            ),
            CandidateEvidence(
                alternative_id,
                "近いが違う発話",
                acoustic=0.55,
                mora=0.50,
                lexical=0.66,
                preservation=0.60,
                cross_model=0.48,
            ),
        )
        output.append(
            FusionTrainingExample(
                example_id=f"example-{index}",
                group_id=f"speaker-{index % 5}",
                candidates=candidates,
                target_distribution={
                    correct_id: 1.0,
                    fluent_id: 0.0,
                    alternative_id: 0.0,
                },
            )
        )
    return output


def test_constrained_fusion_improves_training_cross_entropy() -> None:
    result = train_constrained_fusion(
        _examples(),
        config=LearnedFusionConfig(
            epochs=160,
            learning_rate=0.06,
            acoustic_family_floor=0.72,
            seed=11,
        ),
    )
    assert result.after.cross_entropy < result.before.cross_entropy
    assert result.after.top1_accuracy >= result.before.top1_accuracy
    assert sum(result.profile.weights.values()) == pytest.approx(1.0)
    acoustic_weight = sum(
        result.profile.weights[stream] for stream in ACOUSTIC_FAMILY
    )
    assert acoustic_weight >= 0.72
    runtime = result.profile.to_fusion_config()
    assert runtime.priors == result.profile.weights
    assert runtime.acoustic_family_floor == pytest.approx(0.72)


def test_learned_fusion_rejects_uncalibrated_out_of_range_streams() -> None:
    candidates = (
        CandidateEvidence("a", "候補A", acoustic=-3.0, mora=0.8),
        CandidateEvidence("b", "候補B", acoustic=-4.0, mora=0.7),
    )
    with pytest.raises(ValueError, match="held-out calibrated"):
        FusionTrainingExample(
            example_id="invalid",
            group_id="speaker",
            candidates=candidates,
            target_distribution={"a": 1.0, "b": 0.0},
        )


def test_fusion_training_refuses_calibration_split() -> None:
    candidates = (
        CandidateEvidence("a", "候補A", acoustic=0.8, mora=0.8),
        CandidateEvidence("b", "候補B", acoustic=0.2, mora=0.2),
    )
    with pytest.raises(ValueError, match="training split"):
        FusionTrainingExample(
            example_id="invalid",
            group_id="speaker",
            candidates=candidates,
            target_distribution={"a": 1.0, "b": 0.0},
            split="calibration",
        )
