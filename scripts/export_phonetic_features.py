#!/usr/bin/env python3
"""Export frozen audio features and deterministic Japanese phone/mora targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_asr.japanese_phonetic_targets import JapanesePronunciationPolicy
from semantic_asr.joint_phonetic_runtime_optional import (
    FrozenAudioFeatureConfig,
    TransformersAudioFeatureBackend,
)
from semantic_asr.phonetic_feature_export import (
    PhoneticFeatureExportConfig,
    PhoneticFeatureExporter,
    PhoneticSourceResourcePolicy,
    load_phonetic_source_manifest,
)


def _exact_object(value: object, expected: set[str], *, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    if set(value) != expected:
        raise ValueError(
            f"{name} schema is not exact; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    return value


def load_config(path: Path):
    payload = _exact_object(
        json.loads(path.read_text(encoding="utf-8")),
        {"schemaVersion", "encoder", "pronunciationPolicy", "export", "resources"},
        name="feature export config",
    )
    if payload["schemaVersion"] != "1":
        raise ValueError("feature export config schemaVersion must be '1'")
    encoder = _exact_object(
        payload["encoder"],
        {
            "modelId",
            "modelRevision",
            "modelArtifactSha256",
            "revisionPolicy",
            "layerIndex",
            "sampleRate",
            "featureDimension",
            "frameStrideMs",
            "device",
            "localFilesOnly",
        },
        name="encoder config",
    )
    pronunciation = _exact_object(
        payload["pronunciationPolicy"],
        {
            "schemaVersion",
            "blankSymbol",
            "nasalSymbol",
            "sokuonSymbol",
            "longVowelSymbol",
            "ignorePunctuation",
            "mappingRevision",
        },
        name="pronunciation policy",
    )
    export = _exact_object(
        payload["export"],
        {
            "schemaVersion",
            "featureDtype",
            "featureSubdirectory",
            "maximumCachedRecordings",
            "fsyncEachRow",
        },
        name="export config",
    )
    resources = _exact_object(
        payload["resources"],
        {
            "maximumItems",
            "maximumReadingCharacters",
            "maximumSegmentDurationMs",
            "maximumTotalAudioSamples",
            "maximumRecordingSamples",
        },
        name="source resource policy",
    )
    feature_config = FrozenAudioFeatureConfig(
        model_id=str(encoder["modelId"]),
        model_revision=str(encoder["modelRevision"]),
        model_artifact_sha256=(
            None
            if encoder["modelArtifactSha256"] is None
            else str(encoder["modelArtifactSha256"])
        ),
        revision_policy=str(encoder["revisionPolicy"]),
        layer_index=int(encoder["layerIndex"]),
        sample_rate=int(encoder["sampleRate"]),
        feature_dimension=int(encoder["featureDimension"]),
        frame_stride_ms=float(encoder["frameStrideMs"]),
    )
    pronunciation_policy = JapanesePronunciationPolicy(
        blank_symbol=str(pronunciation["blankSymbol"]),
        nasal_symbol=str(pronunciation["nasalSymbol"]),
        sokuon_symbol=str(pronunciation["sokuonSymbol"]),
        long_vowel_symbol=str(pronunciation["longVowelSymbol"]),
        ignore_punctuation=bool(pronunciation["ignorePunctuation"]),
        mapping_revision=str(pronunciation["mappingRevision"]),
        schema_version=str(pronunciation["schemaVersion"]),
    )
    export_config = PhoneticFeatureExportConfig(
        feature_dtype=str(export["featureDtype"]),
        feature_subdirectory=str(export["featureSubdirectory"]),
        maximum_cached_recordings=int(export["maximumCachedRecordings"]),
        fsync_each_row=bool(export["fsyncEachRow"]),
        schema_version=str(export["schemaVersion"]),
    )
    resource_policy = PhoneticSourceResourcePolicy(
        maximum_items=int(resources["maximumItems"]),
        maximum_reading_characters=int(resources["maximumReadingCharacters"]),
        maximum_segment_duration_ms=int(resources["maximumSegmentDurationMs"]),
        maximum_total_audio_samples=int(resources["maximumTotalAudioSamples"]),
        maximum_recording_samples=int(resources["maximumRecordingSamples"]),
    )
    runtime = {
        "device": str(encoder["device"]),
        "local_files_only": bool(encoder["localFilesOnly"]),
    }
    return feature_config, pronunciation_policy, export_config, resource_policy, runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-derived-export", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not args.allow_derived_export:
        parser.error("--allow-derived-export is required because this writes derived training data")
    feature_config, pronunciation, export, resources, runtime = load_config(args.config)
    source = load_phonetic_source_manifest(
        args.source,
        split=args.split,
        resources=resources,
    )
    backend = TransformersAudioFeatureBackend(
        feature_config,
        device=runtime["device"],
        local_files_only=runtime["local_files_only"],
    )
    exporter = PhoneticFeatureExporter(
        feature_backend=backend,
        pronunciation_policy=pronunciation,
        config=export,
        source_resources=resources,
    )
    result = exporter.export(
        source,
        args.output,
        allow_derived_export=True,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "schemaVersion": "1",
                "outputManifest": str(result.output_manifest),
                "outputManifestSha256": result.output_manifest_sha256,
                "itemCount": result.item_count,
                "sourceManifestDigest": result.source_manifest_digest,
                "featureBackendConfigDigest": result.feature_backend_config_digest,
                "pronunciationPolicyDigest": result.pronunciation_policy_digest,
                "phoneInventoryDigest": result.phone_inventory_digest,
                "moraInventoryDigest": result.mora_inventory_digest,
                "featureRevision": result.feature_revision,
                "exportConfigDigest": result.export_config_digest,
                "runDigest": result.run_digest,
                "resultDigest": result.digest,
                "claimBoundary": "derived training data only; no model quality claim",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
