from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
QuerySelectedAcousticVerifier = import_module(
    "semantic_asr.acoustic_verifier"
).QuerySelectedAcousticVerifier


def test_query_selected_acoustic_verifier_forward_backward() -> None:
    model = QuerySelectedAcousticVerifier(
        acoustic_hidden_size=12,
        mora_vocab_size=32,
        model_size=16,
        dropout=0.0,
    )
    acoustic = torch.randn(2, 9, 12)
    candidates = torch.tensor(
        [
            [[1, 2, 3, 0], [1, 2, 4, 0], [5, 6, 0, 0]],
            [[7, 8, 9, 0], [7, 8, 10, 0], [11, 12, 0, 0]],
        ]
    )
    output = model(
        acoustic_hidden=acoustic,
        candidate_mora_ids=candidates,
        acoustic_mask=torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 0],
            ],
            dtype=torch.bool,
        ),
        targets=torch.tensor([0, 1]),
    )
    assert output.loss is not None
    assert output.logits.shape == (2, 3)
    assert output.attention.shape == (2, 3, 9)
    assert output.branch_gates.shape == (2, 3, 3)
    assert torch.allclose(
        output.branch_gates.sum(dim=-1),
        torch.ones((2, 3)),
        atol=1e-5,
    )
    output.loss.backward()
    assert model.output[-1].weight.grad is not None
    assert model.query_projection.weight.grad is not None


def test_verifier_rejects_empty_acoustic_mask() -> None:
    model = QuerySelectedAcousticVerifier(
        acoustic_hidden_size=4,
        mora_vocab_size=8,
        model_size=8,
    )
    with pytest.raises(ValueError, match="valid frame"):
        model(
            acoustic_hidden=torch.randn(1, 3, 4),
            candidate_mora_ids=torch.tensor([[[1, 2]]]),
            acoustic_mask=torch.zeros((1, 3), dtype=torch.bool),
        )


def test_verifier_requires_even_model_size() -> None:
    with pytest.raises(ValueError, match="even"):
        QuerySelectedAcousticVerifier(
            acoustic_hidden_size=4,
            mora_vocab_size=8,
            model_size=9,
        )


def test_verifier_rejects_all_padding_candidate() -> None:
    model = QuerySelectedAcousticVerifier(
        acoustic_hidden_size=4,
        mora_vocab_size=8,
        model_size=8,
    )
    with pytest.raises(ValueError, match="valid mora token"):
        model(
            acoustic_hidden=torch.randn(1, 3, 4),
            candidate_mora_ids=torch.tensor([[[0, 0], [1, 2]]]),
        )


def test_verifier_rejects_out_of_vocabulary_mora_id() -> None:
    model = QuerySelectedAcousticVerifier(
        acoustic_hidden_size=4,
        mora_vocab_size=8,
        model_size=8,
    )
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model(
            acoustic_hidden=torch.randn(1, 3, 4),
            candidate_mora_ids=torch.tensor([[[1, 8]]]),
        )
