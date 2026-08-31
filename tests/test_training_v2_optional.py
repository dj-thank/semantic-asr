from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from semantic_asr.training_v2 import (  # noqa: E402
    AcousticTextVerifier,
    MultiObjectiveRankingLoss,
    SparseEvidenceReranker,
    verifier_loss,
)


def test_sparse_evidence_reranker_shapes_constraints_and_backward() -> None:
    model = SparseEvidenceReranker(
        feature_size=10,
        state_size=6,
        hidden_size=32,
        expert_count=5,
        top_k_experts=2,
        acoustic_feature_indices=(0, 1, 2),
        acoustic_floor=0.60,
    )
    candidate_features = torch.randn(3, 4, 10, requires_grad=True)
    state_features = torch.randn(3, 6)
    candidate_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    output = model(
        candidate_features=candidate_features,
        state_features=state_features,
        candidate_mask=candidate_mask,
        hard_routing=False,
    )
    assert output.logits.shape == (3, 4)
    assert output.probabilities.shape == (3, 4)
    assert output.router_probabilities.shape == (3, 5)
    assert output.selected_experts.shape == (3, 2)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.all(output.probabilities[~candidate_mask] == 0)

    task_losses = torch.tensor(
        [
            [0.0, 0.3, 0.9, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.7, 0.4, 0.0, 0.8],
        ]
    )
    critical_losses = torch.tensor(
        [
            [0.0, 0.2, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [1.0, 0.3, 0.0, 1.0],
        ]
    )
    loss_output = MultiObjectiveRankingLoss()(
        logits=output.logits,
        candidate_mask=candidate_mask,
        task_losses=task_losses,
        critical_losses=critical_losses,
        teacher_logits=torch.randn(3, 4),
    )
    assert torch.isfinite(loss_output.loss)
    loss_output.loss.backward()
    assert candidate_features.grad is not None
    assert torch.all(torch.isfinite(candidate_features.grad))


def test_acoustic_text_verifier_scores_candidates_and_backward() -> None:
    verifier = AcousticTextVerifier(
        audio_size=24,
        text_size=20,
        projection_size=16,
        hidden_size=12,
    )
    audio_hidden = torch.randn(2, 18, 24, requires_grad=True)
    audio_mask = torch.tensor(
        [
            [True] * 16 + [False] * 2,
            [True] * 18,
        ]
    )
    text_hidden = torch.randn(2, 3, 7, 20, requires_grad=True)
    text_mask = torch.tensor(
        [
            [
                [True] * 7,
                [True] * 5 + [False] * 2,
                [True] * 6 + [False],
            ],
            [
                [True] * 4 + [False] * 3,
                [True] * 7,
                [True] * 5 + [False] * 2,
            ],
        ]
    )
    duration_features = torch.randn(2, 3, 3)
    output = verifier(
        audio_hidden=audio_hidden,
        audio_mask=audio_mask,
        text_hidden=text_hidden,
        text_mask=text_mask,
        duration_features=duration_features,
    )
    assert output.logits.shape == (2, 3)
    assert output.probabilities.shape == (2, 3)
    assert output.global_similarity.shape == (2, 3)
    assert output.local_alignment.shape == (2, 3)
    assert torch.all((0 <= output.probabilities) & (output.probabilities <= 1))

    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss = verifier_loss(output, labels, positive_weight=1.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert audio_hidden.grad is not None
    assert text_hidden.grad is not None


def test_verifier_rejects_empty_candidate_masks() -> None:
    verifier = AcousticTextVerifier(audio_size=4, text_size=4, projection_size=4, hidden_size=4)
    with pytest.raises(ValueError, match="valid tokens"):
        verifier(
            audio_hidden=torch.randn(1, 3, 4),
            audio_mask=torch.tensor([[True, True, True]]),
            text_hidden=torch.randn(1, 1, 2, 4),
            text_mask=torch.tensor([[[False, False]]]),
        )
