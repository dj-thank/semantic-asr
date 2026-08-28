from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from semantic_asr.training import SemanticASRMultiTask


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_features):
        return SimpleNamespace(last_hidden_state=self.projection(input_features))


def test_multitask_heads_forward_and_backward() -> None:
    model = SemanticASRMultiTask(
        FakeEncoder(8),
        hidden_size=8,
        mora_vocab_size=12,
        phone_vocab_size=16,
    )
    features = torch.randn(2, 7, 8)
    output = model(
        input_features=features,
        encoder_lengths=torch.tensor([7, 6]),
        mora_labels=torch.tensor([[1, 2, 3], [2, 3, -100]]),
        phone_labels=torch.tensor([[1, 2, 3, 4], [2, 3, -100, -100]]),
        boundary_labels=torch.tensor([[0, 1, 0, 2, 0, 1, 0], [0, 1, 0, 2, 0, 1, -100]]),
        accent_labels=torch.tensor([[0, 1, 2, 3, 0, 1, 2], [0, 1, 2, 3, 0, 1, -100]]),
        f0_labels=torch.tensor(
            [
                [100.0, 110.0, 120.0, 130.0, 125.0, 118.0, 111.0],
                [95.0, 100.0, 105.0, 110.0, 108.0, 104.0, -100.0],
            ]
        ),
        preservation_labels=torch.tensor([[0, 0, 1, 0, 2, 0, 3], [0, 1, 0, 2, 0, 3, -100]]),
    )
    assert output.loss is not None
    output.loss.backward()
    assert model.mora_head.weight.grad is not None
    assert model.preservation_head.weight.grad is not None
    assert model.f0_head.weight.grad is not None


def test_ctc_blank_and_padding_are_rejected() -> None:
    model = SemanticASRMultiTask(FakeEncoder(4), hidden_size=4, mora_vocab_size=8)
    features = torch.randn(1, 5, 4)
    with pytest.raises(ValueError):
        model(
            input_features=features,
            mora_labels=torch.tensor([[1, 0, 2]]),
        )
    with pytest.raises(ValueError):
        model(
            input_features=features,
            mora_labels=torch.tensor([[1, -100, 2]]),
        )
