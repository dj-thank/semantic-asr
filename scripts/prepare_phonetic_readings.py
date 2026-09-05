#!/usr/bin/env python3
"""Prepare explicit-kana phonetic source manifests from reviewed reading evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from semantic_asr.japanese_phonetic_targets import JapanesePronunciationPolicy
from semantic_asr.reading_manifest_prepare import (
    load_machine_reading_proposals,
    load_reading_preparation_manifest,
    load_reading_review_ledger,
    prepare_phonetic_source_manifest,
)
from semantic_asr.reading_provenance import ReadingResolutionPolicy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--machine-proposals", type=Path)
    parser.add_argument("--review-ledger", type=Path)
    parser.add_argument("--review-ledger-revision")
    parser.add_argument("--review-protocol-revision")
    parser.add_argument("--review-batch-manifest-sha256")
    parser.add_argument("--allow-unreviewed-machine-train", action="store_true")
    parser.add_argument("--disallow-human-explicit", action="store_true")
    parser.add_argument("--allow-output", action="store_true")
    parser.add_argument("--maximum-items", type=int, default=2_000_000)
    parser.add_argument("--maximum-transcript-characters", type=int, default=100_000)
    args = parser.parse_args()

    if not args.allow_output:
        parser.error("--allow-output is required because this writes derived reading data")
    review_values = (
        args.review_ledger_revision,
        args.review_protocol_revision,
        args.review_batch_manifest_sha256,
    )
    if args.review_ledger is None and any(value is not None for value in review_values):
        parser.error("review metadata requires --review-ledger")
    if args.review_ledger is not None and any(value is None for value in review_values):
        parser.error(
            "--review-ledger requires --review-ledger-revision, "
            "--review-protocol-revision, and --review-batch-manifest-sha256"
        )

    pronunciation_policy = JapanesePronunciationPolicy()
    source = load_reading_preparation_manifest(
        args.input,
        split=args.split,
        maximum_items=args.maximum_items,
        maximum_transcript_characters=args.maximum_transcript_characters,
    )
    machine_proposals = (
        {}
        if args.machine_proposals is None
        else load_machine_reading_proposals(
            args.machine_proposals,
            pronunciation_policy=pronunciation_policy,
        )
    )
    review_ledger = (
        None
        if args.review_ledger is None
        else load_reading_review_ledger(
            args.review_ledger,
            revision=args.review_ledger_revision,
            review_protocol_revision=args.review_protocol_revision,
            review_batch_manifest_sha256=args.review_batch_manifest_sha256,
        )
    )
    resolution_policy = ReadingResolutionPolicy(
        allow_human_explicit=not args.disallow_human_explicit,
        allow_unreviewed_machine_train=args.allow_unreviewed_machine_train,
        require_review_for_calibration=True,
        require_review_for_test=True,
    )
    result = prepare_phonetic_source_manifest(
        source,
        args.output,
        machine_proposals=machine_proposals,
        review_ledger=review_ledger,
        pronunciation_policy=pronunciation_policy,
        resolution_policy=resolution_policy,
        allow_output=True,
    )
    print(
        json.dumps(
            {
                "schemaVersion": result.schema_version,
                "outputManifest": str(result.output_manifest),
                "receiptManifest": str(result.receipt_manifest),
                "outputManifestSha256": result.output_manifest_sha256,
                "receiptManifestSha256": result.receipt_manifest_sha256,
                "inputManifestDigest": result.input_manifest_digest,
                "pronunciationPolicy": asdict(pronunciation_policy),
                "pronunciationPolicyDigest": result.pronunciation_policy_digest,
                "resolutionPolicy": asdict(resolution_policy),
                "resolutionPolicyDigest": result.resolution_policy_digest,
                "itemCount": result.item_count,
                "originCounts": dict(result.origin_counts),
                "resultDigest": result.digest,
                "claimBoundary": (
                    "reading provenance preparation only; no pronunciation or ASR quality claim"
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
