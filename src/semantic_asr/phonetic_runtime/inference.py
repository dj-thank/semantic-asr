"""Source-audio-bound phone and mora posterior inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..phonetic_evidence import PosteriorFrame, PosteriorSequence
from .artifact import LoadedDualCTCArtifact, load_dual_ctc_artifact, metadata_runtime_digest
from .audio import LoadedWaveform, load_pcm16_wav
from .contracts import PhoneticInventory, PhoneticRuntimeLimits


def _posterior_frames(
    probabilities,
    inventory: PhoneticInventory,
    waveform: LoadedWaveform,
) -> tuple[PosteriorFrame, ...]:
    if probabilities.ndim != 2 or probabilities.shape[1] != inventory.size:
        raise ValueError("posterior tensor shape does not match its label inventory")
    frame_count = int(probabilities.shape[0])
    if frame_count < 1:
        raise ValueError("posterior tensor contains no frames")
    duration_ms = waveform.end_ms - waveform.start_ms
    if duration_ms < frame_count:
        raise ValueError("posterior frame grid is finer than integer-millisecond provenance")
    output: list[PosteriorFrame] = []
    for index in range(frame_count):
        start_ms = waveform.start_ms + math.floor(duration_ms * index / frame_count)
        end_ms = waveform.start_ms + math.floor(duration_ms * (index + 1) / frame_count)
        end_ms = max(start_ms + 1, min(end_ms, waveform.end_ms))
        row = [float(value) for value in probabilities[index].tolist()]
        total = sum(row)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("posterior frame has invalid probability mass")
        distribution = {
            symbol: max(0.0, value / total)
            for symbol, value in zip(inventory.symbols, row, strict=True)
        }
        renormalizer = sum(distribution.values())
        distribution[inventory.symbols[-1]] += 1.0 - renormalizer
        output.append(
            PosteriorFrame.from_mapping(
                start_ms=start_ms,
                end_ms=end_ms,
                probabilities=distribution,
            )
        )
    return tuple(output)


@dataclass(slots=True)
class DualCTCPosteriorRuntime:
    artifact: LoadedDualCTCArtifact
    device: str = "cpu"
    limits: PhoneticRuntimeLimits = PhoneticRuntimeLimits()

    def __post_init__(self) -> None:
        if not self.device:
            raise ValueError("runtime device is required")
        self.artifact.model.to(self.device)
        self.artifact.model.eval()

    @classmethod
    def from_artifact(
        cls,
        directory: str | Path,
        *,
        device: str = "cpu",
        limits: PhoneticRuntimeLimits | None = None,
    ) -> DualCTCPosteriorRuntime:
        return cls(
            artifact=load_dual_ctc_artifact(directory, device=device),
            device=device,
            limits=limits or PhoneticRuntimeLimits(),
        )

    @property
    def profile_digest(self) -> str:
        return metadata_runtime_digest(self.artifact.metadata)

    @property
    def source(self) -> str:
        metadata = self.artifact.metadata
        return f"dual-ctc:{metadata.name}@{metadata.revision}"

    def infer(
        self,
        audio_path: str | Path,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        expected_source_audio_sha256: str | None = None,
    ) -> tuple[PosteriorSequence, PosteriorSequence]:
        import torch

        metadata = self.artifact.metadata
        waveform = load_pcm16_wav(
            audio_path,
            expected_sample_rate=metadata.model_config.frontend.sample_rate,
            limits=self.limits,
            start_ms=start_ms,
            end_ms=end_ms,
            expected_source_audio_sha256=expected_source_audio_sha256,
        )
        samples = torch.frombuffer(waveform.samples, dtype=torch.float32).clone().unsqueeze(0)
        lengths = torch.tensor([samples.shape[1]], dtype=torch.long)
        samples = samples.to(self.device)
        lengths = lengths.to(self.device)
        with torch.inference_mode():
            output = self.artifact.model(samples, lengths)
            output_length = int(output.output_lengths[0].item())
            phone = output.phone_logits[0, :output_length].softmax(dim=-1).cpu()
            mora = output.mora_logits[0, :output_length].softmax(dim=-1).cpu()
        encoder_revision = self.profile_digest
        phone_sequence = PosteriorSequence(
            kind="phone",
            blank_symbol=metadata.phone_inventory.blank_symbol,
            vocabulary=metadata.phone_inventory.symbols,
            frames=_posterior_frames(phone, metadata.phone_inventory, waveform),
            encoder=self.source,
            encoder_revision=encoder_revision,
            label_set_revision=metadata.phone_inventory.revision,
            source_audio_sha256=waveform.source_audio_sha256,
        )
        mora_sequence = PosteriorSequence(
            kind="mora",
            blank_symbol=metadata.mora_inventory.blank_symbol,
            vocabulary=metadata.mora_inventory.symbols,
            frames=_posterior_frames(mora, metadata.mora_inventory, waveform),
            encoder=self.source,
            encoder_revision=encoder_revision,
            label_set_revision=metadata.mora_inventory.revision,
            source_audio_sha256=waveform.source_audio_sha256,
        )
        return phone_sequence, mora_sequence
