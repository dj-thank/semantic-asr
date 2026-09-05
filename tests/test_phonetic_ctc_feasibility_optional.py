from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from _phonetic_runtime_fixture import (  # noqa: E402
    model_config,
    mora_inventory,
    phone_inventory,
)

from semantic_asr.phonetic_runtime.artifact import (  # noqa: E402
    read_dual_ctc_metadata,
    save_dual_ctc_artifact,
)
from semantic_asr.phonetic_runtime.torch_model import (  # noqa: E402
    DualCTCOutput,
    DualPhoneMoraCTC,
    minimum_ctc_frames,
    multitask_ctc_loss,
)


def test_repeated_labels_require_intervening_ctc_frames() -> None:
    targets = torch.tensor([1, 1, 2, 3, 3, 3], dtype=torch.long)
    lengths = torch.tensor([3, 3], dtype=torch.long)

    required = minimum_ctc_frames(targets, lengths)

    assert required.tolist() == [4, 5]


def test_impossible_repeated_label_alignment_fails_before_loss() -> None:
    phone_logits = torch.randn(1, 2, phone_inventory().size, requires_grad=True)
    mora_logits = torch.randn(1, 2, mora_inventory().size, requires_grad=True)
    output = DualCTCOutput(
        phone_logits=phone_logits,
        mora_logits=mora_logits,
        output_lengths=torch.tensor([2], dtype=torch.long),
        encoded=torch.randn(1, 2, model_config().hidden_dimension),
    )

    with pytest.raises(ValueError, match="phone target requires more CTC frames"):
        multitask_ctc_loss(
            output,
            phone_targets=torch.tensor([1, 1], dtype=torch.long),
            phone_target_lengths=torch.tensor([2], dtype=torch.long),
            mora_targets=torch.tensor([1], dtype=torch.long),
            mora_target_lengths=torch.tensor([1], dtype=torch.long),
            phone_blank_id=0,
            mora_blank_id=0,
        )


def test_identical_state_dicts_produce_identical_weight_archives(tmp_path: Path) -> None:
    torch.manual_seed(41)
    model = DualPhoneMoraCTC(model_config(), phone_inventory(), mora_inventory())
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first = save_dual_ctc_artifact(
        first_directory,
        model,
        name="deterministic",
        revision="r1",
        model_config=model_config(),
        phone_inventory=phone_inventory(),
        mora_inventory=mora_inventory(),
        training_manifest_sha256="a" * 64,
        runtime_revision="runtime-r1",
    )
    second = save_dual_ctc_artifact(
        second_directory,
        model,
        name="deterministic",
        revision="r1",
        model_config=model_config(),
        phone_inventory=phone_inventory(),
        mora_inventory=mora_inventory(),
        training_manifest_sha256="a" * 64,
        runtime_revision="runtime-r1",
    )

    assert first.weights_sha256 == second.weights_sha256
    assert (first_directory / "weights.npz").read_bytes() == (
        second_directory / "weights.npz"
    ).read_bytes()
    assert read_dual_ctc_metadata(first_directory).digest == first.digest
