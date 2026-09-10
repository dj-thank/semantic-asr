#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from semantic_asr.experiment import DatasetManifest, SplitName, UtteranceRecord

_ROOT_FIELDS = frozenset(
    (
        "schemaVersion",
        "datasetName",
        "datasetRevision",
        "rightsRegistryDigest",
        "splitPolicy",
        "splitSeed",
        "exclusions",
        "records",
    )
)
_EXCLUSION_FIELDS = frozenset(("sampleIds", "audioSha256", "referenceSha256"))
_RECORD_FIELDS = frozenset(
    (
        "sampleId",
        "split",
        "audioSha256",
        "pcmSha256",
        "reference",
        "speakerId",
        "sessionId",
        "sourceRecordingId",
        "referenceLineageId",
        "derivationGroupId",
        "nearDuplicateId",
        "rightsAssetId",
        "sourceDatasetRevision",
        "parentSampleIds",
        "durationSeconds",
        "domain",
        "metadata",
    )
)
_REQUIRED_RECORD_FIELDS = frozenset(
    (
        "sampleId",
        "split",
        "audioSha256",
        "pcmSha256",
        "reference",
        "speakerId",
        "sessionId",
        "sourceRecordingId",
        "referenceLineageId",
        "derivationGroupId",
        "nearDuplicateId",
        "rightsAssetId",
        "sourceDatasetRevision",
        "parentSampleIds",
        "metadata",
    )
)


def _reject_unknown_fields(
    row: dict[str, object],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} field(s): {', '.join(unknown)}")


def _required_value(row: dict[str, object], key: str, *, context: str) -> object:
    if key not in row:
        raise ValueError(f"{context} field {key} is required")
    return row[key]


def _required_string(row: dict[str, object], key: str, *, context: str) -> str:
    value = _required_value(row, key, context=context)
    if not isinstance(value, str):
        raise ValueError(f"{context} field {key} must be a string")
    return value


def _optional_string(row: dict[str, object], key: str, *, context: str) -> str | None:
    value = _required_value(row, key, context=context)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} field {key} must be a string or null")
    return value


def _string_array(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be an array of strings")
    return tuple(value)


def _record(row: dict[str, object], *, index: int) -> UtteranceRecord:
    context = f"record {index}"
    _reject_unknown_fields(row, _RECORD_FIELDS, context=context)
    missing = sorted(_REQUIRED_RECORD_FIELDS - set(row))
    if missing:
        raise ValueError(f"{context} is missing required field(s): {', '.join(missing)}")

    metadata = row["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} field metadata must be an object")
    duration = row.get("durationSeconds")
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, (int, float))
    ):
        raise ValueError(f"{context} field durationSeconds must be a number or null")
    domain = row.get("domain")
    if domain is not None and not isinstance(domain, str):
        raise ValueError(f"{context} field domain must be a string or null")

    return UtteranceRecord(
        sample_id=_required_string(row, "sampleId", context=context),
        split=cast(SplitName, _required_string(row, "split", context=context)),
        audio_sha256=_required_string(row, "audioSha256", context=context),
        pcm_sha256=_required_string(row, "pcmSha256", context=context),
        reference=_required_string(row, "reference", context=context),
        speaker_id=_optional_string(row, "speakerId", context=context),
        session_id=_optional_string(row, "sessionId", context=context),
        source_recording_id=_optional_string(row, "sourceRecordingId", context=context),
        reference_lineage_id=_optional_string(row, "referenceLineageId", context=context),
        derivation_group_id=_optional_string(row, "derivationGroupId", context=context),
        near_duplicate_id=_optional_string(row, "nearDuplicateId", context=context),
        rights_asset_id=_required_string(row, "rightsAssetId", context=context),
        source_dataset_revision=_required_string(row, "sourceDatasetRevision", context=context),
        parent_sample_ids=_string_array(
            row["parentSampleIds"],
            name=f"{context} field parentSampleIds",
        ),
        duration_seconds=None if duration is None else float(duration),
        domain=domain,
        metadata=dict(metadata),
    )


def manifest_from_payload(payload: object) -> DatasetManifest:
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain a JSON object")
    _reject_unknown_fields(payload, _ROOT_FIELDS, context="manifest")
    if payload.get("schemaVersion") != "2.0.0":
        raise ValueError("schemaVersion must be exactly 2.0.0")

    raw_records = _required_value(payload, "records", context="manifest")
    if not isinstance(raw_records, list) or any(not isinstance(row, dict) for row in raw_records):
        raise ValueError("manifest field records must be an array of objects")
    records = tuple(_record(row, index=index) for index, row in enumerate(raw_records))

    raw_exclusions = _required_value(payload, "exclusions", context="manifest")
    if not isinstance(raw_exclusions, dict):
        raise ValueError("manifest field exclusions must be an object")
    exclusions: dict[str, object] = raw_exclusions
    _reject_unknown_fields(exclusions, _EXCLUSION_FIELDS, context="exclusions")
    missing_exclusions = sorted(_EXCLUSION_FIELDS - set(exclusions))
    if missing_exclusions:
        raise ValueError(
            "exclusions is missing required field(s): " + ", ".join(missing_exclusions)
        )

    split_seed = _required_value(payload, "splitSeed", context="manifest")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("manifest field splitSeed must be an integer")

    return DatasetManifest(
        records=records,
        dataset_name=_required_string(payload, "datasetName", context="manifest"),
        dataset_revision=_required_string(payload, "datasetRevision", context="manifest"),
        rights_registry_digest=_required_string(
            payload, "rightsRegistryDigest", context="manifest"
        ),
        split_policy=_required_string(payload, "splitPolicy", context="manifest"),
        split_seed=split_seed,
        excluded_sample_ids=_string_array(
            exclusions["sampleIds"],
            name="exclusions field sampleIds",
        ),
        excluded_audio_sha256=_string_array(
            exclusions["audioSha256"],
            name="exclusions field audioSha256",
        ),
        excluded_reference_sha256=_string_array(
            exclusions["referenceSha256"],
            name="exclusions field referenceSha256",
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the five-role Semantic ASR experiment manifest, lineage, "
            "cumulative exclusions, and split isolation."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--allow-reference-duplicates",
        action="store_true",
        help="Do not fail when normalized reference duplicates occur across splits.",
    )
    parser.add_argument(
        "--require-known-speakers",
        action="store_true",
        help="Fail instead of reporting that speaker-disjointness is unproven.",
    )
    parser.add_argument(
        "--require-known-rights",
        action="store_true",
        help="Fail when any record lacks a rightsAssetId.",
    )
    parser.add_argument(
        "--public-redacted-output",
        type=Path,
        help="Write a public-safe manifest without references, speaker IDs, or metadata.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = manifest_from_payload(payload)
    manifest.assert_lineage_valid(
        reference_near_duplicate=not args.allow_reference_duplicates,
        require_known_speakers=args.require_known_speakers,
        require_known_rights=args.require_known_rights,
    )
    report = manifest.integrity_report(reference_near_duplicate=not args.allow_reference_duplicates)
    report.update(
        {
            "schemaVersion": payload.get("schemaVersion"),
            "datasetName": manifest.dataset_name,
            "datasetRevision": manifest.dataset_revision,
            "records": len(manifest.records),
        }
    )
    if args.public_redacted_output is not None:
        args.public_redacted_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_redacted_output.write_text(
            json.dumps(
                manifest.public_redacted_payload(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report["publicRedactedOutput"] = str(args.public_redacted_output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
