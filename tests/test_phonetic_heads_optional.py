from __future__ import annotations

import pytest

from semantic_asr.phonetic_training import (
    JointPhoneticHeadConfig,
    PhoneticLabelInventory,
)


torch = pytest.importorskip("torch")

from semantic_asr.phonetic_heads_optional import JointPhoneMoraCTCHead  # noqa: E402


def inventory(kind: str, labels: tuple[str, ...]) -> PhoneticLabelInventory:
    return PhoneticLabelInventory(
        kind=kind,
        labels=labels,
        blank_symbol="<blk>",
        revision=f"{kind}-r1",
        source_manifest_sha256=("a" if kind == "phone" else "b") * 64,
    )


def config() -> JointPhoneticHeadConfig:
    return JointPhoneticHeadConfig(
        input_dimension=8,
        hidden_dimension=12,
        phone_inventory=inventory("phone", ("<blk>", "m", "a", "d")),
        mora_inventory=inventory("mora", ("<blk>", "マ", "ダ")),
        encoder_id="frozen-test-encoder",
        encoder_revision="1" * 40,
        encoder_artifact_sha256="c" * 64,
        dropout=0.0,
        blank_regularization_weight=0.1,
    )


def test_joint_head_emits_both_label_spaces_and_backpropagates() -> None:
    model = JointPhoneMoraCTCHead(config())
    features = torch.randn(2, 7, 8, requires_grad=True)

    output = model(features)
    loss = model.loss(
        output,
        input_lengths=torch.tensor([7, 6], dtype=torch.long),
        phone_targets=torch.tensor([1, 2, 3, 1, 2], dtype=torch.long),
        phone_target_lengths=torch.tensor([3, 2], dtype=torch.long),
        mora_targets=torch.tensor([1, 2, 1, 2], dtype=torch.long),
        mora_target_lengths=torch.tensor([2, 2], dtype=torch.long),
    )
    loss.total.backward()

    assert output.phone_logits.shape == (2, 7, 4)
    assert output.mora_logits.shape == (2, 7, 3)
    assert torch.isfinite(loss.total)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_joint_head_rejects_blank_targets() -> None:
    model = JointPhoneMoraCTCHead(config())
    output = model(torch.randn(1, 5, 8))

    with pytest.raises(ValueError, match="phone targets"):
        model.loss(
            output,
            input_lengths=torch.tensor([5], dtype=torch.long),
            phone_targets=torch.tensor([0], dtype=torch.long),
            phone_target_lengths=torch.tensor([1], dtype=torch.long),
            mora_targets=torch.tensor([1], dtype=torch.long),
            mora_target_lengths=torch.tensor([1], dtype=torch.long),
        )


def test_joint_head_rejects_wrong_feature_dimension() -> None:
    model = JointPhoneMoraCTCHead(config())

    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.randn(1, 5, 7))
