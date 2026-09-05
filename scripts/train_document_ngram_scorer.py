#!/usr/bin/env python3
"""Train a frozen dependency-free document-context scorer artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semantic_asr.document_experiment.artifacts import BidirectionalNgramArtifact
from semantic_asr.document_experiment.ngram_scorer import (
    NgramCalibrationSequence,
    fit_character_ngram_model,
    fit_ngram_normalization,
)


@dataclass(frozen=True, slots=True)
class ManifestRow:
    text: str
    speaker_id: str
    session_id: str
    license_id: str
    left_context: str = ""
    right_context: str = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: Any, *, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {name} must be a non-empty string")
    return value


def _read_manifest(path: Path) -> tuple[ManifestRow, ...]:
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"line {line_number}: manifest row must be an object")
            if payload.get("rightsDecision") != "allow":
                raise ValueError(
                    f"line {line_number}: rightsDecision must be 'allow' for derived artifacts"
                )
            rows.append(
                ManifestRow(
                    text=_required_text(
                        payload.get("text"),
                        name="text",
                        line_number=line_number,
                    ),
                    speaker_id=_required_text(
                        payload.get("speakerId"),
                        name="speakerId",
                        line_number=line_number,
                    ),
                    session_id=_required_text(
                        payload.get("sessionId"),
                        name="sessionId",
                        line_number=line_number,
                    ),
                    license_id=_required_text(
                        payload.get("licenseId"),
                        name="licenseId",
                        line_number=line_number,
                    ),
                    left_context=str(payload.get("leftContext") or ""),
                    right_context=str(payload.get("rightContext") or ""),
                )
            )
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return tuple(rows)


def _disjoint(train: tuple[ManifestRow, ...], calibration: tuple[ManifestRow, ...]) -> None:
    train_speakers = {row.speaker_id for row in train}
    calibration_speakers = {row.speaker_id for row in calibration}
    if train_speakers.intersection(calibration_speakers):
        raise ValueError("training and calibration speakers overlap")
    train_sessions = {row.session_id for row in train}
    calibration_sessions = {row.session_id for row in calibration}
    if train_sessions.intersection(calibration_sessions):
        raise ValueError("training and calibration sessions overlap")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default="document-char-ngram")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument(
        "--allow-derived-artifact",
        action="store_true",
        help="Acknowledge that n-gram count artifacts can retain information about source text.",
    )
    args = parser.parse_args()
    if not args.allow_derived_artifact:
        parser.error("--allow-derived-artifact is required")

    train_path = args.train.resolve(strict=True)
    calibration_path = args.calibration.resolve(strict=True)
    if train_path == calibration_path:
        raise ValueError("training and calibration manifests must be different files")
    train_rows = _read_manifest(train_path)
    calibration_rows = _read_manifest(calibration_path)
    _disjoint(train_rows, calibration_rows)
    train_digest = _sha256_file(train_path)
    calibration_digest = _sha256_file(calibration_path)

    forward = fit_character_ngram_model(
        (row.text for row in train_rows),
        order=args.order,
        alpha=args.alpha,
        training_manifest_sha256=train_digest,
        revision=f"{args.revision}:forward",
    )
    backward = fit_character_ngram_model(
        (row.text for row in train_rows),
        order=args.order,
        alpha=args.alpha,
        training_manifest_sha256=train_digest,
        revision=f"{args.revision}:backward",
        reversed_text=True,
    )
    normalization = fit_ngram_normalization(
        forward,
        backward,
        tuple(
            NgramCalibrationSequence(
                text=row.text,
                left_context=row.left_context,
                right_context=row.right_context,
            )
            for row in calibration_rows
        ),
        calibration_manifest_sha256=calibration_digest,
        revision=f"{args.revision}:normalization",
    )
    artifact = BidirectionalNgramArtifact(
        forward=forward,
        backward=backward,
        normalization=normalization,
        name=args.name,
        revision=args.revision,
    )
    destination = artifact.write(args.output)
    print(
        json.dumps(
            {
                "artifact": str(destination),
                "artifactDigest": artifact.digest,
                "trainingManifestSha256": train_digest,
                "calibrationManifestSha256": calibration_digest,
                "trainingRows": len(train_rows),
                "calibrationRows": len(calibration_rows),
                "speakerDisjoint": True,
                "sessionDisjoint": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
