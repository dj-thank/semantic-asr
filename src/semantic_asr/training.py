from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("semantic_asr.training requires the 'train' extra") from exc


@dataclass(slots=True)
class MultiTaskOutput:
    loss: Tensor | None
    mora_ctc_loss: Tensor | None
    phone_ctc_loss: Tensor | None
    boundary_loss: Tensor | None
    accent_loss: Tensor | None
    f0_loss: Tensor | None
    preservation_loss: Tensor | None
    mora_logits: Tensor
    phone_logits: Tensor | None
    boundary_logits: Tensor
    accent_logits: Tensor
    f0_prediction: Tensor
    preservation_logits: Tensor
    hidden_states: Tensor


def _hidden_states(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if not isinstance(hidden, Tensor):
        raise TypeError("encoder must return a tensor or last_hidden_state")
    return hidden


def _validate_ctc_targets(labels: Tensor, *, blank_id: int, name: str) -> None:
    if labels.ndim != 2:
        raise ValueError(f"{name} labels must be [batch, target]")
    valid = labels.ne(-100)
    if torch.any(labels.masked_select(valid).eq(blank_id)):
        raise ValueError(f"{name} labels must not contain the CTC blank ID")
    for row in range(labels.shape[0]):
        row_valid = valid[row]
        if torch.any(~row_valid[:-1] & row_valid[1:]):
            raise ValueError(f"{name} labels must use right padding only")


def _ctc_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    input_lengths: Tensor,
    blank_id: int,
    name: str,
) -> Tensor:
    _validate_ctc_targets(labels, blank_id=blank_id, name=name)
    valid = labels.ne(-100)
    target_lengths = valid.sum(dim=1).to(dtype=torch.long)
    if torch.any(target_lengths > input_lengths):
        raise ValueError(f"{name} target length exceeds encoder length")
    targets = labels.masked_select(valid).to(dtype=torch.long)
    return F.ctc_loss(
        logits.log_softmax(dim=-1).transpose(0, 1),
        targets,
        input_lengths.to(dtype=torch.long),
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )


def _frame_cross_entropy(logits: Tensor, labels: Tensor, *, name: str) -> Tensor:
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


class SemanticASRMultiTask(nn.Module):
    """Attach Japanese acoustic heads to an arbitrary speech encoder.

    The encoder is shared once. The module adds mora CTC, optional phone CTC,
    mora-boundary, accent, F0, and error-preservation heads. It intentionally does
    not replace a Whisper decoder; text decoding can remain a separate loss path.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_size: int,
        mora_vocab_size: int,
        phone_vocab_size: int | None = None,
        boundary_classes: int = 3,
        accent_classes: int = 4,
        preservation_classes: int = 4,
        mora_blank_id: int = 0,
        phone_blank_id: int = 0,
        mora_weight: float = 0.45,
        phone_weight: float = 0.20,
        boundary_weight: float = 0.10,
        accent_weight: float = 0.08,
        f0_weight: float = 0.07,
        preservation_weight: float = 0.10,
    ) -> None:
        super().__init__()
        if mora_vocab_size < 2:
            raise ValueError("mora_vocab_size must include blank and labels")
        if phone_vocab_size is not None and phone_vocab_size < 2:
            raise ValueError("phone_vocab_size must include blank and labels")
        self.encoder = encoder
        self.mora_head = nn.Linear(hidden_size, mora_vocab_size)
        self.phone_head = (
            nn.Linear(hidden_size, phone_vocab_size) if phone_vocab_size is not None else None
        )
        self.boundary_head = nn.Linear(hidden_size, boundary_classes)
        self.accent_head = nn.Linear(hidden_size, accent_classes)
        self.f0_head = nn.Linear(hidden_size, 1)
        self.preservation_head = nn.Linear(hidden_size, preservation_classes)
        self.mora_blank_id = mora_blank_id
        self.phone_blank_id = phone_blank_id
        self.loss_weights = {
            "mora": float(mora_weight),
            "phone": float(phone_weight),
            "boundary": float(boundary_weight),
            "accent": float(accent_weight),
            "f0": float(f0_weight),
            "preservation": float(preservation_weight),
        }

    def forward(
        self,
        *,
        input_features: Tensor,
        encoder_lengths: Tensor | None = None,
        mora_labels: Tensor | None = None,
        phone_labels: Tensor | None = None,
        boundary_labels: Tensor | None = None,
        accent_labels: Tensor | None = None,
        f0_labels: Tensor | None = None,
        preservation_labels: Tensor | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> MultiTaskOutput:
        encoded = self.encoder(input_features, **(encoder_kwargs or {}))
        hidden = _hidden_states(encoded)
        if hidden.ndim != 3:
            raise ValueError("encoder hidden states must be [batch, frames, hidden]")
        batch, frames, _ = hidden.shape
        if encoder_lengths is None:
            encoder_lengths = torch.full((batch,), frames, dtype=torch.long, device=hidden.device)
        if encoder_lengths.shape != (batch,):
            raise ValueError("encoder_lengths must have one value per batch item")
        if torch.any(encoder_lengths <= 0) or torch.any(encoder_lengths > frames):
            raise ValueError("encoder_lengths must be within encoder frame range")

        mora_logits = self.mora_head(hidden)
        phone_logits = self.phone_head(hidden) if self.phone_head is not None else None
        boundary_logits = self.boundary_head(hidden)
        accent_logits = self.accent_head(hidden)
        f0_prediction = self.f0_head(hidden).squeeze(-1)
        preservation_logits = self.preservation_head(hidden)

        mora_loss = (
            _ctc_loss(
                mora_logits,
                mora_labels,
                input_lengths=encoder_lengths,
                blank_id=self.mora_blank_id,
                name="mora",
            )
            if mora_labels is not None
            else None
        )
        phone_loss = (
            _ctc_loss(
                phone_logits,
                phone_labels,
                input_lengths=encoder_lengths,
                blank_id=self.phone_blank_id,
                name="phone",
            )
            if phone_logits is not None and phone_labels is not None
            else None
        )
        boundary_loss = (
            _frame_cross_entropy(boundary_logits, boundary_labels, name="boundary")
            if boundary_labels is not None
            else None
        )
        accent_loss = (
            _frame_cross_entropy(accent_logits, accent_labels, name="accent")
            if accent_labels is not None
            else None
        )
        preservation_loss = (
            _frame_cross_entropy(
                preservation_logits,
                preservation_labels,
                name="preservation",
            )
            if preservation_labels is not None
            else None
        )
        f0_loss = None
        if f0_labels is not None:
            if f0_labels.shape != f0_prediction.shape:
                raise ValueError("f0 labels must match encoder frame shape")
            mask = torch.isfinite(f0_labels) & f0_labels.ne(-100)
            frame_index = torch.arange(frames, device=hidden.device).unsqueeze(0)
            mask &= frame_index < encoder_lengths.unsqueeze(1)
            if torch.any(mask):
                f0_loss = F.smooth_l1_loss(
                    f0_prediction.masked_select(mask),
                    f0_labels.masked_select(mask),
                )

        weighted: list[Tensor] = []
        for name, loss in (
            ("mora", mora_loss),
            ("phone", phone_loss),
            ("boundary", boundary_loss),
            ("accent", accent_loss),
            ("f0", f0_loss),
            ("preservation", preservation_loss),
        ):
            if loss is not None:
                weighted.append(self.loss_weights[name] * loss)
        total_loss = torch.stack(weighted).sum() if weighted else None
        return MultiTaskOutput(
            loss=total_loss,
            mora_ctc_loss=mora_loss,
            phone_ctc_loss=phone_loss,
            boundary_loss=boundary_loss,
            accent_loss=accent_loss,
            f0_loss=f0_loss,
            preservation_loss=preservation_loss,
            mora_logits=mora_logits,
            phone_logits=phone_logits,
            boundary_logits=boundary_logits,
            accent_logits=accent_logits,
            f0_prediction=f0_prediction,
            preservation_logits=preservation_logits,
            hidden_states=hidden,
        )

    def save_auxiliary_heads(self, path: str) -> None:
        torch.save(
            {
                "mora_head": self.mora_head.state_dict(),
                "phone_head": (None if self.phone_head is None else self.phone_head.state_dict()),
                "boundary_head": self.boundary_head.state_dict(),
                "accent_head": self.accent_head.state_dict(),
                "f0_head": self.f0_head.state_dict(),
                "preservation_head": self.preservation_head.state_dict(),
                "loss_weights": self.loss_weights,
                "mora_blank_id": self.mora_blank_id,
                "phone_blank_id": self.phone_blank_id,
            },
            path,
        )
