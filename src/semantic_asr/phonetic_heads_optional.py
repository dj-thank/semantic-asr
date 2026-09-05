"""Optional PyTorch implementation of the shared phone/mora CTC head.

Importing this module does not silently download an encoder or weights. The caller supplies frozen
frame features and exact sequence lengths. Weight serialization is deliberately outside this module
and must use the safetensors artifact contract from ``phonetic_training``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .phonetic_training import JointPhoneticHeadConfig

try:  # pragma: no cover - the dependency-free suite intentionally has no torch
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    Tensor = Any
    nn = None
    F = None


@dataclass(frozen=True, slots=True)
class JointPhoneticOutput:
    phone_logits: Tensor
    mora_logits: Tensor


@dataclass(frozen=True, slots=True)
class JointPhoneticLoss:
    total: Tensor
    phone_ctc: Tensor
    mora_ctc: Tensor
    blank_regularization: Tensor


if nn is not None:

    class JointPhoneMoraCTCHead(nn.Module):
        """Shared bottleneck followed by independent phone and mora CTC projections."""

        def __init__(self, config: JointPhoneticHeadConfig) -> None:
            super().__init__()
            self.config = config
            self.normalization = nn.LayerNorm(config.input_dimension)
            self.shared = nn.Sequential(
                nn.Linear(config.input_dimension, config.hidden_dimension),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.phone_head = nn.Linear(
                config.hidden_dimension,
                len(config.phone_inventory.labels),
            )
            self.mora_head = nn.Linear(
                config.hidden_dimension,
                len(config.mora_inventory.labels),
            )

        def forward(self, features: Tensor) -> JointPhoneticOutput:
            if features.ndim != 3:
                raise ValueError("features must have shape [batch, frames, dimension]")
            if features.shape[-1] != self.config.input_dimension:
                raise ValueError("feature dimension does not match the frozen configuration")
            if not torch.isfinite(features).all():
                raise ValueError("features must be finite")
            hidden = self.shared(self.normalization(features))
            return JointPhoneticOutput(
                phone_logits=self.phone_head(hidden),
                mora_logits=self.mora_head(hidden),
            )

        def loss(
            self,
            output: JointPhoneticOutput,
            *,
            input_lengths: Tensor,
            phone_targets: Tensor,
            phone_target_lengths: Tensor,
            mora_targets: Tensor,
            mora_target_lengths: Tensor,
        ) -> JointPhoneticLoss:
            batch, frames, _ = output.phone_logits.shape
            if output.mora_logits.shape[:2] != (batch, frames):
                raise ValueError("phone and mora logits must share batch and frame dimensions")
            if input_lengths.shape != (batch,):
                raise ValueError("input_lengths must contain one value per batch item")
            if phone_target_lengths.shape != (batch,) or mora_target_lengths.shape != (batch,):
                raise ValueError("target lengths must contain one value per batch item")
            if torch.any(input_lengths < 1) or torch.any(input_lengths > frames):
                raise ValueError("input lengths are outside the emitted frame range")
            if torch.any(phone_target_lengths < 1) or torch.any(mora_target_lengths < 1):
                raise ValueError("phone and mora target lengths must be positive")
            if int(phone_target_lengths.sum().item()) != int(phone_targets.numel()):
                raise ValueError("phone target lengths do not match flattened targets")
            if int(mora_target_lengths.sum().item()) != int(mora_targets.numel()):
                raise ValueError("mora target lengths do not match flattened targets")
            if torch.any(phone_targets == self.config.phone_inventory.blank_index):
                raise ValueError("phone targets must not contain the CTC blank label")
            if torch.any(mora_targets == self.config.mora_inventory.blank_index):
                raise ValueError("mora targets must not contain the CTC blank label")
            phone_log_probs = F.log_softmax(output.phone_logits, dim=-1).transpose(0, 1)
            mora_log_probs = F.log_softmax(output.mora_logits, dim=-1).transpose(0, 1)
            phone_ctc = F.ctc_loss(
                phone_log_probs,
                phone_targets,
                input_lengths,
                phone_target_lengths,
                blank=self.config.phone_inventory.blank_index,
                reduction="mean",
                zero_infinity=True,
            )
            mora_ctc = F.ctc_loss(
                mora_log_probs,
                mora_targets,
                input_lengths,
                mora_target_lengths,
                blank=self.config.mora_inventory.blank_index,
                reduction="mean",
                zero_infinity=True,
            )
            phone_blank = phone_log_probs.exp()[
                :, :, self.config.phone_inventory.blank_index
            ].mean()
            mora_blank = mora_log_probs.exp()[
                :, :, self.config.mora_inventory.blank_index
            ].mean()
            # Penalize only pathological all-blank collapse. The hinge leaves ordinary blank mass
            # untouched and is disabled by default through a zero configuration weight.
            blank_regularization = (
                torch.relu(phone_blank - 0.95) + torch.relu(mora_blank - 0.95)
            )
            total = (
                self.config.phone_loss_weight * phone_ctc
                + self.config.mora_loss_weight * mora_ctc
                + self.config.blank_regularization_weight * blank_regularization
            )
            return JointPhoneticLoss(
                total=total,
                phone_ctc=phone_ctc,
                mora_ctc=mora_ctc,
                blank_regularization=blank_regularization,
            )

else:

    class JointPhoneMoraCTCHead:  # pragma: no cover - explicit error without optional dependency
        def __init__(self, config: JointPhoneticHeadConfig) -> None:
            del config
            raise RuntimeError("JointPhoneMoraCTCHead requires PyTorch")
