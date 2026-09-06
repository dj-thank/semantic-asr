"""Reject invalid training intent before it becomes an ignored or inverted loss."""

from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
training = import_module("semantic_asr.training")


def model(**kwargs):
    values = {"hidden_size": 4, "mora_vocab_size": 4}
    values.update(kwargs)
    return training.SemanticASRMultiTask(torch.nn.Identity(), **values)


@pytest.mark.parametrize("name", ["mora", "phone", "boundary", "accent", "f0", "preservation"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "0.5"])
def test_invalid_loss_weight_is_rejected(name, value):
    with pytest.raises((ValueError, TypeError), match="weight"):
        model(**{f"{name}_weight": value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 0),
        ("hidden_size", True),
        ("mora_vocab_size", 1),
        ("phone_vocab_size", True),
        ("boundary_classes", 0),
        ("accent_classes", 1.5),
        ("preservation_classes", True),
        ("mora_blank_id", 4),
        ("mora_blank_id", True),
        ("phone_blank_id", -1),
    ],
)
def test_invalid_head_dimensions_or_blank_are_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        model(**{field: value})


def test_absent_phone_head_rejects_supervision_before_encoder_call():
    class Encoder(torch.nn.Module):
        def forward(self, _):
            raise AssertionError("encoder must not run for unsupported supervision")

    instance = training.SemanticASRMultiTask(Encoder(), hidden_size=4, mora_vocab_size=4)
    with pytest.raises(ValueError, match="phone.*head"):
        instance(input_features=torch.zeros(1, 4, 4), phone_labels=torch.tensor([[1]]))


@pytest.mark.parametrize("lengths", [torch.tensor([2.5]), torch.tensor([True])])
def test_frame_only_training_rejects_noninteger_lengths(lengths):
    with pytest.raises(TypeError, match="encoder_lengths"):
        model()(
            input_features=torch.zeros(1, 4, 4),
            encoder_lengths=lengths,
            boundary_labels=torch.tensor([[0, 1, -100, -100]]),
        )


@pytest.mark.parametrize("shape", [(0, 4, 4), (1, 0, 4), (1, 4, 3)])
def test_empty_or_mismatched_hidden_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="hidden"):
        model()(input_features=torch.zeros(shape))


def test_int32_and_int64_auxiliary_labels_match():
    instance = model()
    features = torch.randn(1, 4, 4)
    labels = torch.tensor([[0, 1, 2, -100]])
    a = instance(input_features=features, boundary_labels=labels)
    b = instance(input_features=features, boundary_labels=labels.to(torch.int32))
    assert torch.equal(a.loss, b.loss)


@pytest.mark.parametrize("labels", [torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool)])
def test_auxiliary_labels_require_integer_class_ids(labels):
    with pytest.raises(TypeError, match="integer"):
        model()(input_features=torch.zeros(1, 4, 4), boundary_labels=labels)


def test_mutated_weights_cannot_bypass_forward_validation():
    instance = model()
    instance.loss_weights["mora"] = -1
    with pytest.raises(ValueError, match="weight"):
        instance(input_features=torch.zeros(1, 4, 4), mora_labels=torch.tensor([[1]]))


def test_nonfinite_auxiliary_loss_is_not_returned_as_training_success():
    with pytest.raises(ValueError, match="finite"):
        model()(
            input_features=torch.full((1, 4, 4), float("nan")),
            boundary_labels=torch.tensor([[0, 1, 2, 0]]),
        )


def test_valid_weighting_and_backward_and_ignored_labels_remain_unchanged():
    instance = model(phone_vocab_size=4)
    labels = torch.tensor([[1, 2]])
    result = instance(input_features=torch.randn(1, 6, 4), mora_labels=labels, phone_labels=labels)
    assert torch.allclose(result.loss, 0.45 * result.mora_ctc_loss + 0.20 * result.phone_ctc_loss)
    result.loss.backward()
    assert instance.mora_head.weight.grad is not None
    assert instance.phone_head.weight.grad is not None
    assert instance.accent_head.weight.grad is None
    ignored = instance(input_features=torch.zeros(1, 4, 4), accent_labels=torch.full((1, 4), -100))
    assert ignored.loss is None
