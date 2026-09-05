from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from _phonetic_runtime_fixture import (  # noqa: E402
    model_config,
    mora_inventory,
    phone_inventory,
)

from semantic_asr.phonetic_runtime.torch_model import (  # noqa: E402
    DualPhoneMoraCTC,
    greedy_ctc_ids,
    multitask_ctc_loss,
)


def test_dual_ctc_forward_backward_uses_shared_frame_grid() -> None:
    config = model_config()
    phone = phone_inventory()
    mora = mora_inventory()
    model = DualPhoneMoraCTC(config, phone, mora)
    waveforms = torch.randn(2, 4_000)
    lengths = torch.tensor([4_000, 3_200], dtype=torch.long)

    output = model(waveforms, lengths)

    assert output.phone_logits.shape[:2] == output.mora_logits.shape[:2]
    assert output.phone_logits.shape[-1] == phone.size
    assert output.mora_logits.shape[-1] == mora.size
    assert torch.all(output.output_lengths > 0)
    phone_targets = torch.tensor(
        [*phone.encode(("k", "a")), *phone.encode(("m", "a"))],
        dtype=torch.long,
    )
    mora_targets = torch.tensor(
        [*mora.encode(("カ",)), *mora.encode(("マ",))],
        dtype=torch.long,
    )
    total, phone_loss, mora_loss = multitask_ctc_loss(
        output,
        phone_targets=phone_targets,
        phone_target_lengths=torch.tensor([2, 2], dtype=torch.long),
        mora_targets=mora_targets,
        mora_target_lengths=torch.tensor([1, 1], dtype=torch.long),
        phone_blank_id=phone.blank_id,
        mora_blank_id=mora.blank_id,
    )
    total.backward()

    assert torch.isfinite(total)
    assert torch.isfinite(phone_loss)
    assert torch.isfinite(mora_loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_greedy_ctc_collapse_removes_blanks_and_repeats() -> None:
    logits = torch.full((1, 6, 4), -10.0)
    for index, label in enumerate((0, 1, 1, 0, 2, 2)):
        logits[0, index, label] = 10.0

    decoded = greedy_ctc_ids(logits, torch.tensor([6]), blank_id=0)

    assert decoded == ((1, 2),)
