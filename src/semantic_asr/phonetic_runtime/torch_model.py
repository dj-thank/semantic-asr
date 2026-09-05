"""Optional PyTorch shared-encoder model with independent phone and mora CTC heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import DualCTCModelConfig, PhoneticInventory


def _hz_to_mel(value: float) -> float:
    return 2595.0 * math.log10(1.0 + value / 700.0)


def _mel_to_hz(value: float) -> float:
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def _mel_filterbank(config: DualCTCModelConfig) -> Tensor:
    frontend = config.frontend
    mel_min = _hz_to_mel(frontend.frequency_min)
    mel_max = _hz_to_mel(frontend.frequency_max)
    points = [
        _mel_to_hz(mel_min + (mel_max - mel_min) * index / (frontend.n_mels + 1))
        for index in range(frontend.n_mels + 2)
    ]
    bin_count = frontend.n_fft // 2 + 1
    frequencies = torch.linspace(0.0, frontend.sample_rate / 2.0, bin_count)
    filters = torch.zeros(frontend.n_mels, bin_count, dtype=torch.float32)
    for mel_index in range(frontend.n_mels):
        left = points[mel_index]
        center = points[mel_index + 1]
        right = points[mel_index + 2]
        rising = (frequencies - left) / max(center - left, 1e-12)
        falling = (right - frequencies) / max(right - center, 1e-12)
        filters[mel_index] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    normalization = filters.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return filters / normalization


def _frame_lengths(lengths: Tensor, config: DualCTCModelConfig) -> Tensor:
    frontend = config.frontend
    lengths = torch.maximum(lengths, torch.full_like(lengths, frontend.n_fft))
    return 1 + torch.div(
        lengths - frontend.n_fft,
        frontend.hop_length,
        rounding_mode="floor",
    )


def _subsample_lengths(lengths: Tensor, config: DualCTCModelConfig) -> Tensor:
    kernel = config.convolution_kernel
    padding = kernel // 2
    output = lengths
    for _ in range(config.subsampling_layers):
        output = 1 + torch.div(
            output + 2 * padding - (kernel - 1) - 1,
            2,
            rounding_mode="floor",
        )
    return torch.clamp(output, min=1)


def _padding_mask(lengths: Tensor, maximum: int) -> Tensor:
    positions = torch.arange(maximum, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def _sinusoidal_positions(length: int, dimension: int, *, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class LogMelFrontend(nn.Module):
    def __init__(self, config: DualCTCModelConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("mel_filters", _mel_filterbank(config), persistent=True)
        self.register_buffer(
            "window",
            torch.hann_window(config.frontend.window_length, periodic=True),
            persistent=True,
        )

    def forward(self, waveforms: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        if lengths.ndim != 1 or lengths.shape[0] != waveforms.shape[0]:
            raise ValueError("lengths must have shape [batch]")
        if waveforms.shape[1] < self.config.frontend.n_fft:
            waveforms = F.pad(
                waveforms,
                (0, self.config.frontend.n_fft - waveforms.shape[1]),
            )
        spectrum = torch.stft(
            waveforms,
            n_fft=self.config.frontend.n_fft,
            hop_length=self.config.frontend.hop_length,
            win_length=self.config.frontend.window_length,
            window=self.window.to(dtype=waveforms.dtype),
            center=False,
            return_complex=True,
        )
        power = spectrum.abs().square()
        mel = torch.einsum(
            "mf,bft->bmt",
            self.mel_filters.to(dtype=power.dtype),
            power,
        )
        features = torch.log(mel.clamp_min(self.config.frontend.log_floor)).transpose(1, 2)
        frame_lengths = _frame_lengths(lengths, self.config)
        frame_lengths = torch.minimum(
            frame_lengths,
            torch.full_like(frame_lengths, features.shape[1]),
        )
        if self.config.frontend.normalize_per_utterance:
            normalized: list[Tensor] = []
            for index, frame_length in enumerate(frame_lengths.tolist()):
                valid = features[index, :frame_length]
                mean = valid.mean(dim=0, keepdim=True)
                standard_deviation = valid.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
                row = (features[index] - mean) / standard_deviation
                normalized.append(row)
            features = torch.stack(normalized)
        return features, frame_lengths


@dataclass(frozen=True, slots=True)
class DualCTCOutput:
    phone_logits: Tensor
    mora_logits: Tensor
    output_lengths: Tensor
    encoded: Tensor

    def __post_init__(self) -> None:
        if self.phone_logits.ndim != 3 or self.mora_logits.ndim != 3:
            raise ValueError("CTC logits must have shape [batch, frames, labels]")
        if self.phone_logits.shape[:2] != self.mora_logits.shape[:2]:
            raise ValueError("phone and mora heads must share batch and frame dimensions")
        if self.encoded.shape[:2] != self.phone_logits.shape[:2]:
            raise ValueError("encoded representation and logits have different frame grids")
        if self.output_lengths.ndim != 1:
            raise ValueError("output_lengths must have shape [batch]")


class DualPhoneMoraCTC(nn.Module):
    """Shared log-Mel/Transformer encoder with independent CTC output spaces."""

    def __init__(
        self,
        config: DualCTCModelConfig,
        phone_inventory: PhoneticInventory,
        mora_inventory: PhoneticInventory,
    ) -> None:
        super().__init__()
        if phone_inventory.kind != "phone" or mora_inventory.kind != "mora":
            raise ValueError("phone and mora inventories are assigned to the wrong heads")
        self.config = config
        self.phone_inventory = phone_inventory
        self.mora_inventory = mora_inventory
        self.frontend = LogMelFrontend(config)
        convolution: list[nn.Module] = []
        input_channels = config.frontend.n_mels
        for _ in range(config.subsampling_layers):
            convolution.extend(
                (
                    nn.Conv1d(
                        input_channels,
                        config.hidden_dimension,
                        kernel_size=config.convolution_kernel,
                        stride=2,
                        padding=config.convolution_kernel // 2,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )
            input_channels = config.hidden_dimension
        self.subsampler = nn.Sequential(*convolution)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dimension,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dimension,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.hidden_dimension),
            enable_nested_tensor=False,
        )
        self.phone_head = nn.Linear(config.hidden_dimension, phone_inventory.size)
        self.mora_head = nn.Linear(config.hidden_dimension, mora_inventory.size)

    def forward(self, waveforms: Tensor, lengths: Tensor) -> DualCTCOutput:
        features, frame_lengths = self.frontend(waveforms, lengths)
        encoded = self.subsampler(features.transpose(1, 2)).transpose(1, 2)
        output_lengths = _subsample_lengths(frame_lengths, self.config)
        if encoded.shape[1] > self.config.maximum_frames:
            raise ValueError("encoded frame count exceeds maximum_frames")
        positions = _sinusoidal_positions(
            encoded.shape[1],
            encoded.shape[2],
            device=encoded.device,
        ).to(dtype=encoded.dtype)
        encoded = encoded + positions.unsqueeze(0)
        mask = _padding_mask(output_lengths, encoded.shape[1])
        encoded = self.encoder(encoded, src_key_padding_mask=mask)
        return DualCTCOutput(
            phone_logits=self.phone_head(encoded),
            mora_logits=self.mora_head(encoded),
            output_lengths=output_lengths,
            encoded=encoded,
        )


def minimum_ctc_frames(targets: Tensor, target_lengths: Tensor) -> Tensor:
    """Return the minimum CTC frame count, including repeated-label separators."""

    if targets.ndim != 1:
        raise ValueError("CTC targets must be one-dimensional and concatenated")
    if target_lengths.ndim != 1:
        raise ValueError("CTC target_lengths must have shape [batch]")
    if torch.any(target_lengths < 1):
        raise ValueError("CTC target lengths must be positive")
    if int(target_lengths.sum().item()) != targets.numel():
        raise ValueError("CTC target lengths do not match the concatenated target tensor")
    output: list[int] = []
    cursor = 0
    for length in target_lengths.tolist():
        row = targets[cursor : cursor + length]
        repeats = int((row[1:] == row[:-1]).sum().item()) if length > 1 else 0
        output.append(int(length) + repeats)
        cursor += int(length)
    return torch.tensor(output, dtype=torch.long, device=target_lengths.device)


def _validate_ctc_feasibility(
    *,
    name: str,
    targets: Tensor,
    target_lengths: Tensor,
    output_lengths: Tensor,
    blank_id: int,
) -> None:
    if output_lengths.ndim != 1 or output_lengths.shape != target_lengths.shape:
        raise ValueError(f"{name} output and target lengths have incompatible shapes")
    if torch.any(targets == blank_id):
        raise ValueError(f"{name} targets must not contain the CTC blank")
    required = minimum_ctc_frames(targets, target_lengths)
    available = output_lengths.to(required.device)
    if torch.any(required > available):
        index = int(torch.nonzero(required > available, as_tuple=False)[0].item())
        raise ValueError(
            f"{name} target requires more CTC frames than available for batch row {index}: "
            f"required={int(required[index])}, available={int(available[index])}"
        )


def multitask_ctc_loss(
    output: DualCTCOutput,
    *,
    phone_targets: Tensor,
    phone_target_lengths: Tensor,
    mora_targets: Tensor,
    mora_target_lengths: Tensor,
    phone_blank_id: int,
    mora_blank_id: int,
    phone_weight: float = 1.0,
    mora_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_ctc_feasibility(
        name="phone",
        targets=phone_targets,
        target_lengths=phone_target_lengths,
        output_lengths=output.output_lengths,
        blank_id=phone_blank_id,
    )
    _validate_ctc_feasibility(
        name="mora",
        targets=mora_targets,
        target_lengths=mora_target_lengths,
        output_lengths=output.output_lengths,
        blank_id=mora_blank_id,
    )
    if phone_weight < 0.0 or mora_weight < 0.0 or phone_weight + mora_weight <= 0.0:
        raise ValueError("CTC loss weights must be non-negative with positive total")
    phone_loss = F.ctc_loss(
        output.phone_logits.log_softmax(dim=-1).transpose(0, 1),
        phone_targets,
        output.output_lengths,
        phone_target_lengths,
        blank=phone_blank_id,
        reduction="mean",
        zero_infinity=False,
    )
    mora_loss = F.ctc_loss(
        output.mora_logits.log_softmax(dim=-1).transpose(0, 1),
        mora_targets,
        output.output_lengths,
        mora_target_lengths,
        blank=mora_blank_id,
        reduction="mean",
        zero_infinity=False,
    )
    denominator = phone_weight + mora_weight
    total = (phone_weight * phone_loss + mora_weight * mora_loss) / denominator
    return total, phone_loss, mora_loss


def greedy_ctc_ids(
    logits: Tensor, lengths: Tensor, *, blank_id: int = 0
) -> tuple[tuple[int, ...], ...]:
    if logits.ndim != 3 or lengths.ndim != 1 or logits.shape[0] != lengths.shape[0]:
        raise ValueError("greedy CTC inputs have incompatible shapes")
    predictions = logits.argmax(dim=-1)
    output: list[tuple[int, ...]] = []
    for row, length in zip(predictions, lengths.tolist(), strict=True):
        collapsed: list[int] = []
        previous = blank_id
        for value in row[:length].tolist():
            if value != blank_id and value != previous:
                collapsed.append(int(value))
            previous = int(value)
        output.append(tuple(collapsed))
    return tuple(output)
