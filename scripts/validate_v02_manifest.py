#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_asr.experiment import DatasetManifest, UtteranceRecord


def _record(row: dict[str, object]) -> UtteranceRecord:
    return UtteranceRecord(
        sample_id=str(row.get("sampleId") or row.get("sample_id") or ""),
        split=str(row.get("split") or ""),  # type: ignore[arg-type]
        audio_sha256=str(row.get("audioSha256") or row.get("audio_sha256") or ""),
        reference=str(row.get("reference") or ""),
        speaker_id=(
            None
            if row.get("speakerId", row.get("speaker_id")) is None
            else str(row.get("speakerId", row.get("speaker_id")))
        ),
        source_recording_id=(
            None
            if row.get("sourceRecordingId", row.get("source_recording_id")) is None
            else str(row.get("sourceRecordingId", row.get("source_recording_id")))
        ),
        duration_seconds=(
            None
            if row.get("durationSeconds", row.get("duration_seconds")) is None
            else float(row.get("durationSeconds", row.get("duration_seconds")))
        ),
        domain=(None if row.get("domain") is None else str(row.get("domain"))),
        metadata=dict(row.get("metadata") or {}),  # type: ignore[arg-type]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a Semantic ASR v0.2 experiment manifest and split isolation."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-reference-duplicates",
        action="store_true",
        help="Do not fail when exact normalized references occur across splits.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("manifest records must be an array")
    records = tuple(_record(dict(row)) for row in raw_records)
    manifest = DatasetManifest(
        records=records,
        dataset_name=str(payload.get("datasetName") or payload.get("dataset_name") or ""),
        dataset_revision=str(
            payload.get("datasetRevision") or payload.get("dataset_revision") or ""
        ),
        rights_registry_digest=(
            None
            if payload.get("rightsRegistryDigest", payload.get("rights_registry_digest")) is None
            else str(payload.get("rightsRegistryDigest", payload.get("rights_registry_digest")))
        ),
    )
    manifest.assert_leakage_free(reference_near_duplicate=not args.allow_reference_duplicates)
    print(
        json.dumps(
            {
                "manifestDigest": manifest.digest,
                "datasetName": manifest.dataset_name,
                "datasetRevision": manifest.dataset_revision,
                "records": len(manifest.records),
                "train": len(manifest.split("train")),
                "calibration": len(manifest.split("calibration")),
                "test": len(manifest.split("test")),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
