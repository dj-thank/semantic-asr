#!/usr/bin/env python3
"""Fit phone/mora candidate utilities on the frozen calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from semantic_asr.contracts import sha256_json
from semantic_asr.phonetic_runtime.calibration import (
    PhoneticCalibrationCandidate,
    PhoneticCalibrationExample,
    fit_ctc_utility_calibration,
)
from semantic_asr.phonetic_runtime.calibration_artifact import DualCTCUtilityArtifact
from semantic_asr.phonetic_runtime.inference import DualCTCPosteriorRuntime
from semantic_asr.phonetic_runtime.manifest import load_phonetic_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _outside_checkout(path: Path) -> Path:
    destination = path.resolve()
    repository = Path(__file__).resolve().parents[1]
    if destination == repository or repository in destination.parents:
        raise ValueError("calibration artifacts must be written outside the checkout")
    return destination


def _strict_string(value: Any, *, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"line {line_number}: {name} must be a non-empty string")
    return value


def _symbols(value: Any, *, name: str, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"line {line_number}: {name} must be a non-empty array")
    if any(not isinstance(row, str) or not row for row in value):
        raise TypeError(f"line {line_number}: {name} must contain strings")
    return tuple(value)


def _load_candidates(path: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    output: dict[str, tuple[dict[str, Any], ...]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"line {line_number}: candidate row must be an object")
            if set(payload) != {"utteranceId", "candidates"}:
                raise ValueError(f"line {line_number}: candidate row keys are not exact")
            utterance_id = _strict_string(
                payload["utteranceId"],
                name="utteranceId",
                line_number=line_number,
            )
            if utterance_id in output:
                raise ValueError(f"line {line_number}: duplicate utteranceId")
            rows = payload["candidates"]
            if not isinstance(rows, list) or len(rows) < 2:
                raise ValueError(f"line {line_number}: at least two candidates are required")
            parsed: list[dict[str, Any]] = []
            seen: set[str] = set()
            for candidate in rows:
                if not isinstance(candidate, dict):
                    raise TypeError(f"line {line_number}: candidate must be an object")
                expected = {"candidateId", "text", "phoneSymbols", "moraSymbols", "correct"}
                if set(candidate) != expected:
                    raise ValueError(f"line {line_number}: candidate keys are not exact")
                candidate_id = _strict_string(
                    candidate["candidateId"],
                    name="candidateId",
                    line_number=line_number,
                )
                if candidate_id in seen:
                    raise ValueError(f"line {line_number}: duplicate candidateId")
                seen.add(candidate_id)
                if not isinstance(candidate["correct"], bool):
                    raise TypeError(f"line {line_number}: correct must be a boolean")
                parsed.append(
                    {
                        "candidate_id": candidate_id,
                        "text": _strict_string(
                            candidate["text"],
                            name="text",
                            line_number=line_number,
                        ),
                        "phone_symbols": _symbols(
                            candidate["phoneSymbols"],
                            name="phoneSymbols",
                            line_number=line_number,
                        ),
                        "mora_symbols": _symbols(
                            candidate["moraSymbols"],
                            name="moraSymbols",
                            line_number=line_number,
                        ),
                        "correct": candidate["correct"],
                    }
                )
            if sum(row["correct"] for row in parsed) != 1:
                raise ValueError(f"line {line_number}: exactly one candidate must be correct")
            output[utterance_id] = tuple(parsed)
    if not output:
        raise ValueError("candidate calibration file contains no rows")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = _outside_checkout(args.output)
    manifest = load_phonetic_manifest(args.manifest)
    calibration_rows = manifest.rows_for("calibration")
    if not calibration_rows:
        raise ValueError("phonetic manifest has no calibration split")
    candidates_path = args.candidates.resolve(strict=True)
    candidate_rows = _load_candidates(candidates_path)
    expected_ids = {row.utterance_id for row in calibration_rows}
    if set(candidate_rows) != expected_ids:
        raise ValueError("candidate calibration rows must exactly match calibration utterances")
    runtime = DualCTCPosteriorRuntime.from_artifact(args.artifact_dir, device=args.device)
    phone_examples: list[PhoneticCalibrationExample] = []
    mora_examples: list[PhoneticCalibrationExample] = []
    for row in calibration_rows:
        phone, mora = runtime.infer(
            row.audio_path,
            expected_source_audio_sha256=row.source_audio_sha256,
        )
        entries = candidate_rows[row.utterance_id]
        phone_examples.append(
            PhoneticCalibrationExample(
                example_id=row.utterance_id,
                posterior=phone,
                candidates=tuple(
                    PhoneticCalibrationCandidate(
                        candidate_id=entry["candidate_id"],
                        text=entry["text"],
                        symbols=entry["phone_symbols"],
                        correct=entry["correct"],
                    )
                    for entry in entries
                ),
            )
        )
        mora_examples.append(
            PhoneticCalibrationExample(
                example_id=row.utterance_id,
                posterior=mora,
                candidates=tuple(
                    PhoneticCalibrationCandidate(
                        candidate_id=entry["candidate_id"],
                        text=entry["text"],
                        symbols=entry["mora_symbols"],
                        correct=entry["correct"],
                    )
                    for entry in entries
                ),
            )
        )
    held_out_digest = sha256_json(
        {
            "manifestDigest": manifest.digest,
            "candidateFileSha256": _sha256_file(candidates_path),
            "calibrationUtteranceIds": tuple(row.utterance_id for row in calibration_rows),
        }
    )
    phone_report = fit_ctc_utility_calibration(
        tuple(phone_examples),
        held_out_manifest_sha256=held_out_digest,
        revision=f"{args.revision}:phone",
    )
    mora_report = fit_ctc_utility_calibration(
        tuple(mora_examples),
        held_out_manifest_sha256=held_out_digest,
        revision=f"{args.revision}:mora",
    )
    artifact = DualCTCUtilityArtifact.from_reports(
        phone_report,
        mora_report,
        name=args.name,
        revision=args.revision,
        runtime_profile_digest=runtime.profile_digest,
    )
    artifact.write(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "artifactDigest": artifact.digest,
                "runtimeProfileDigest": runtime.profile_digest,
                "heldOutDigest": held_out_digest,
                "phonePairwiseAccuracy": phone_report.pairwise_accuracy,
                "moraPairwiseAccuracy": mora_report.pairwise_accuracy,
                "examples": len(calibration_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
