#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing patch anchor for {label}: {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


def patch_training() -> None:
    path = ROOT / "src/semantic_asr/training.py"
    replace_once(
        path,
        '''def _frame_cross_entropy(logits: Tensor, labels: Tensor, *, name: str) -> Tensor:
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"{name} labels must match encoder frame shape")
    valid = labels.ne(-100)
    if torch.any(valid & ((labels < 0) | (labels >= logits.shape[-1]))):
        raise ValueError(f"{name} label is outside class range")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
''',
        '''def _frame_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    *,
    name: str,
) -> Tensor | None:
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"{name} labels must match encoder frame shape")
    valid = labels.ne(-100)
    if torch.any(valid & ((labels < 0) | (labels >= logits.shape[-1]))):
        raise ValueError(f"{name} label is outside class range")
    if not torch.any(valid):
        # PyTorch's mean-reduced cross entropy is NaN when every item is ignored.
        # An all-padding head contributes no supervision and therefore no loss.
        return None
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )
''',
        label="all-ignored frame loss",
    )


def patch_tests() -> None:
    path = ROOT / "tests/test_training_optional.py"
    text = path.read_text(encoding="utf-8")
    marker = "def test_all_ignored_frame_labels_do_not_create_nan_loss() -> None:"
    if marker in text:
        return
    addition = '''


def test_all_ignored_frame_labels_do_not_create_nan_loss() -> None:
    model = SemanticASRMultiTask(
        ToyEncoder(),
        hidden_size=4,
        mora_vocab_size=6,
        phone_vocab_size=7,
    )
    ignored = torch.full((1, 4), -100, dtype=torch.long)
    output = model(
        input_features=torch.randn(1, 4, 4),
        boundary_labels=ignored,
        accent_labels=ignored,
        preservation_labels=ignored,
    )
    assert output.loss is None
    assert output.boundary_loss is None
    assert output.accent_loss is None
    assert output.preservation_loss is None


def test_frame_labels_beyond_encoder_length_fail_closed() -> None:
    model = SemanticASRMultiTask(
        ToyEncoder(),
        hidden_size=4,
        mora_vocab_size=6,
        phone_vocab_size=7,
    )
    with pytest.raises(
        ValueError,
        match="boundary labels contain values beyond encoder_lengths",
    ):
        model(
            input_features=torch.randn(1, 4, 4),
            encoder_lengths=torch.tensor([2]),
            boundary_labels=torch.tensor([[0, 1, 0, -100]]),
        )
'''
    path.write_text(text.rstrip() + addition + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    patch_training()
    patch_tests()
    Path(__file__).unlink()
    print("training loss edge fix applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
