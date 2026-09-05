#!/usr/bin/env python3
"""Train a source-audio-only dual phone/mora CTC artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_asr.phonetic_runtime.contracts import (
    DualCTCModelConfig,
    LogMelFrontendConfig,
    PhoneticInventory,
)
from semantic_asr.phonetic_runtime.manifest import load_phonetic_manifest
from semantic_asr.phonetic_runtime.training import (
    DualCTCTrainingConfig,
    train_dual_ctc_model,
)


def _load_inventory(path: Path, *, expected_kind: str) -> PhoneticInventory:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("inventory file must contain one JSON object")
    expected = {
        "kind",
        "symbols",
        "blankSymbol",
        "unknownSymbol",
        "language",
        "revision",
    }
    if set(payload) != expected:
        raise ValueError(
            f"inventory keys mismatch; missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )
    if payload["kind"] != expected_kind:
        raise ValueError(f"inventory kind must be {expected_kind!r}")
    symbols = payload["symbols"]
    if not isinstance(symbols, list):
        raise TypeError("inventory symbols must be an array")
    return PhoneticInventory(
        kind=expected_kind,  # type: ignore[arg-type]
        symbols=tuple(symbols),
        blank_symbol=payload["blankSymbol"],
        unknown_symbol=payload["unknownSymbol"],
        language=payload["language"],
        revision=payload["revision"],
    )


def _outside_checkout(path: Path) -> Path:
    destination = path.resolve()
    repository = Path(__file__).resolve().parents[1]
    if destination == repository or repository in destination.parents:
        raise ValueError("training artifacts and reports must be written outside the checkout")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phone-inventory", type=Path, required=True)
    parser.add_argument("--mora-inventory", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--phone-loss-weight", type=float, default=1.0)
    parser.add_argument("--mora-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--maximum-audio-seconds", type=float, default=35.0)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--window-length", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--frequency-min", type=float, default=20.0)
    parser.add_argument("--frequency-max", type=float, default=7_600.0)
    parser.add_argument("--hidden-dimension", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dimension", type=int, default=1_024)
    parser.add_argument("--convolution-kernel", type=int, default=5)
    parser.add_argument("--subsampling-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--maximum-frames", type=int, default=12_000)
    args = parser.parse_args()

    artifact_directory = _outside_checkout(args.artifact_dir)
    report_path = _outside_checkout(args.report)
    if artifact_directory == report_path or artifact_directory in report_path.parents:
        raise ValueError(
            "training report must not be nested inside the immutable artifact directory"
        )
    manifest = load_phonetic_manifest(args.manifest)
    phone_inventory = _load_inventory(args.phone_inventory, expected_kind="phone")
    mora_inventory = _load_inventory(args.mora_inventory, expected_kind="mora")
    model_config = DualCTCModelConfig(
        frontend=LogMelFrontendConfig(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            window_length=args.window_length,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            frequency_min=args.frequency_min,
            frequency_max=args.frequency_max,
        ),
        hidden_dimension=args.hidden_dimension,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        feedforward_dimension=args.feedforward_dimension,
        convolution_kernel=args.convolution_kernel,
        subsampling_layers=args.subsampling_layers,
        dropout=args.dropout,
        maximum_frames=args.maximum_frames,
    )
    training_config = DualCTCTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        phone_loss_weight=args.phone_loss_weight,
        mora_loss_weight=args.mora_loss_weight,
        seed=args.seed,
        device=args.device,
        maximum_audio_seconds=args.maximum_audio_seconds,
    )
    result = train_dual_ctc_model(
        manifest,
        phone_inventory=phone_inventory,
        mora_inventory=mora_inventory,
        model_config=model_config,
        training_config=training_config,
        artifact_directory=artifact_directory,
        artifact_name=args.artifact_name,
        artifact_revision=args.artifact_revision,
        runtime_revision=args.runtime_revision,
    )
    result.write_report(report_path)
    print(
        json.dumps(
            {
                "artifactDirectory": str(artifact_directory),
                "artifactDigest": result.artifact.digest,
                "trainingResultDigest": result.digest,
                "manifestDigest": manifest.digest,
                "bestEpoch": result.best_epoch,
                "bestValidationLoss": result.best_validation_loss,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
