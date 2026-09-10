from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from semantic_asr.experiment import DatasetManifest, UtteranceRecord
from semantic_asr.fusion_io import fusion_example_from_row
from semantic_asr.ranker_dataset import ranker_example_from_row
from semantic_asr.ranker_training import example_from_row, load_jsonl_examples
from semantic_asr.rights import RightsRecord, RightsRegistry


def _identifier_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utterance(
    sample_id: str,
    split: str,
    marker: str,
    **changes: Any,
) -> UtteranceRecord:
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "audio_sha256": marker * 64,
        "reference": f"参照文-{sample_id}",
        "speaker_id": f"speaker-{sample_id}",
        "source_recording_id": f"recording-{sample_id}",
        "pcm_sha256": chr(ord(marker) + 1) * 64,
        "session_id": f"session-{sample_id}",
        "reference_lineage_id": f"reference-{sample_id}",
        "derivation_group_id": f"derivation-{sample_id}",
        "near_duplicate_id": f"near-{sample_id}",
        "rights_asset_id": "fixture-rights",
        "source_dataset_revision": "revision-1",
        "parent_sample_ids": (),
        "metadata": {},
    }
    payload.update(changes)
    return UtteranceRecord(**payload)  # type: ignore[arg-type]


def manifest(*records: UtteranceRecord, **changes: Any) -> DatasetManifest:
    payload: dict[str, Any] = {
        "records": tuple(records),
        "dataset_name": "fixture",
        "dataset_revision": "revision-1",
        "split_policy": "speaker-source-lineage-v1",
        "split_seed": 17,
    }
    payload.update(changes)
    return DatasetManifest(**payload)


def rights_record(asset_id: str = "fixture-rights", **changes: Any) -> RightsRecord:
    payload: dict[str, Any] = {
        "asset_id": asset_id,
        "source_name": "Fixture",
        "source_url": "https://example.invalid/source",
        "version": "revision-1",
        "license_name": "Fixture licence",
        "license_url": "https://example.invalid/license",
        "train": "allow",
        "derive_features": "allow",
        "redistribute_raw": "deny",
        "export_speaker_id": "deny",
        "attribution": "Fixture",
        "reviewed_at": "2026-09-11",
        "evaluate": "allow",
        "publish_text": "allow",
        "publish_features": "allow",
        "publish_weights": "review",
        "redistribute_audio": "deny",
        "acquired_at": "2026-09-10",
        "license_text_sha256": "f" * 64,
        "consent_record": "consent-fixture-v1",
    }
    payload.update(changes)
    return RightsRecord(**payload)


def ranker_row(split: object = "train") -> dict[str, object]:
    return {
        "exampleId": "example",
        "groupId": "group",
        "split": split,
        "reference": "料金は3000円です",
        "candidates": [
            {"candidateId": "a", "text": "料金は3000円です", "acoustic": 0.8},
            {"candidateId": "b", "text": "料金は30000円です", "acoustic": 0.2},
        ],
        "targetDistribution": {"a": 1.0, "b": 0.0},
    }


def test_five_roles_are_canonical_and_model_visible_paths_fail_closed() -> None:
    rows = (
        utterance("train", "train", "1"),
        utterance("dev", "dev", "2"),
        utterance("calibration", "calibration", "3"),
        utterance("test", "test", "4"),
        utterance("exposed", "regression-exposed", "5"),
    )
    dataset = manifest(*rows)
    assert [row.sample_id for row in dataset.training_records()] == ["train"]
    assert [row.sample_id for row in dataset.context_records()] == ["train", "dev"]
    assert [row.sample_id for row in dataset.calibration_records()] == ["calibration"]
    with pytest.raises(PermissionError, match="forbids split"):
        dataset.model_visible_records(("test",), purpose="context")  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="forbids split"):
        dataset.model_visible_records(
            ("regression-exposed",),
            purpose="training",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "kind", "value"),
    (
        ("pcm_sha256", "pcm-sha256", "a" * 64),
        ("speaker_id", "speaker-id", "same-speaker"),
        ("session_id", "session-id", "same-session"),
        ("source_recording_id", "source-recording-id", "same-recording"),
        ("reference_lineage_id", "reference-lineage-id", "same-reference"),
        ("derivation_group_id", "derivation-group-id", "same-derivation"),
        ("near_duplicate_id", "near-duplicate-id", "same-near-duplicate"),
    ),
)
def test_lineage_dimensions_cannot_cross_roles(field: str, kind: str, value: str) -> None:
    first = utterance("train", "train", "1", **{field: value})
    second = utterance("test", "test", "2", **{field: value})
    findings = manifest(first, second).leakage_findings(reference_near_duplicate=False)
    assert any(finding.kind == kind and finding.value == value for finding in findings)


def test_normalized_reference_and_parent_lineage_cannot_cross_roles() -> None:
    parent = utterance("parent", "train", "1", reference="東 京です")
    child = utterance(
        "child",
        "test",
        "2",
        reference="東京です",
        parent_sample_ids=("parent",),
    )
    findings = manifest(parent, child).leakage_findings()
    assert {finding.kind for finding in findings} >= {"reference-digest", "parent-sample-id"}


def test_missing_and_cyclic_parents_are_reported_not_silently_skipped() -> None:
    missing = manifest(utterance("child", "train", "1", parent_sample_ids=("not-present",)))
    assert missing.integrity_report()["missingParentRelations"] == [
        {"sampleId": "child", "parentSampleId": "not-present"}
    ]
    with pytest.raises(ValueError, match="lineage integrity"):
        missing.assert_lineage_valid()

    left = utterance("left", "train", "1", parent_sample_ids=("right",))
    right = utterance("right", "train", "2", parent_sample_ids=("left",))
    cyclic = manifest(left, right)
    assert cyclic.integrity_report()["cyclicSampleIds"] == ["left", "right"]
    with pytest.raises(ValueError, match="lineage integrity"):
        cyclic.assert_lineage_valid()


@pytest.mark.parametrize("dimension", ("sample", "audio", "reference"))
def test_cumulative_exclusions_keep_exposed_material_out_of_test(dimension: str) -> None:
    row = utterance("test", "test", "1")
    kwargs: dict[str, object] = {}
    if dimension == "sample":
        kwargs["excluded_sample_ids"] = (row.sample_id,)
    elif dimension == "audio":
        kwargs["excluded_audio_sha256"] = (row.audio_sha256,)
    else:
        kwargs["excluded_reference_sha256"] = (row.reference_digest,)
    dataset = manifest(row, **kwargs)
    assert any(finding.kind.startswith("excluded-") for finding in dataset.leakage_findings())
    with pytest.raises(ValueError, match="leakage"):
        dataset.assert_leakage_free()


def test_test_isolation_covers_every_cross_role_identity_dimension() -> None:
    train = utterance("train", "train", "1", pcm_sha256="a" * 64)
    test = utterance("test", "test", "2", pcm_sha256="a" * 64)
    report = manifest(train, test).integrity_report(reference_near_duplicate=False)
    assert report["testIsolationPassed"] is False
    assert any(row["kind"] == "pcm-sha256" for row in report["leakageFindings"])


def test_required_pcm_and_source_revision_are_reported_and_rejected() -> None:
    row = utterance(
        "incomplete",
        "train",
        "1",
        pcm_sha256=None,
        source_dataset_revision=None,
    )
    dataset = manifest(row)
    report = dataset.integrity_report()
    assert report["missingPcmSha256SampleIds"] == ["incomplete"]
    assert report["unknownSourceDatasetRevisionSampleIds"] == ["incomplete"]
    assert report["lineagePassed"] is False
    with pytest.raises(ValueError, match="lineage integrity"):
        dataset.assert_lineage_valid()


def test_unknown_speaker_rights_and_revision_have_distinct_audit_states() -> None:
    row = utterance(
        "legacy",
        "regression-exposed",
        "1",
        speaker_id=None,
        rights_asset_id=None,
        source_dataset_revision="old-revision",
    )
    report = manifest(row).integrity_report()
    assert report["unknownSpeakerSampleIds"] == ["legacy"]
    assert report["unknownRightsSampleIds"] == ["legacy"]
    assert report["datasetRevisionMismatchSampleIds"] == ["legacy"]
    assert report["speakerDisjointGuaranteed"] is False
    with pytest.raises(ValueError, match="lineage integrity"):
        manifest(row).assert_lineage_valid(require_known_speakers=True, require_known_rights=True)


def test_manifest_digest_binds_split_policy_seed_and_exclusions() -> None:
    row = utterance("train", "train", "1")
    base = manifest(row)
    reordered = manifest(replace(row, metadata={"b": 2, "a": 1}))
    reordered_again = manifest(replace(row, metadata={"a": 1, "b": 2}))
    assert reordered.digest == reordered_again.digest
    assert replace(base, split_seed=18).digest != base.digest
    assert replace(base, split_policy="different-policy").digest != base.digest
    assert replace(base, excluded_sample_ids=("old-public-sample",)).digest != base.digest


def test_public_redaction_omits_reference_speaker_and_metadata_values() -> None:
    row = utterance(
        "train",
        "train",
        "1",
        reference="絶対に公開しない参照文",
        speaker_id="real-person-name",
        metadata={"localPath": "/home/private/audio.wav"},
    )
    payload = manifest(row).public_redacted_payload()
    rendered = json.dumps(payload, ensure_ascii=False)
    assert "絶対に公開しない参照文" not in rendered
    assert "real-person-name" not in rendered
    assert "/home/private/audio.wav" not in rendered
    assert "sampleId" not in payload["records"][0]
    assert payload["records"][0]["sampleIdSha256"] == _identifier_sha256("train")
    assert payload["records"][0]["referenceSha256"] == row.reference_digest
    assert payload["records"][0]["speakerKnown"] is True


def test_rights_registry_is_operation_specific_and_digest_stable() -> None:
    primary = rights_record()
    secondary = rights_record("secondary", publish_weights="allow")
    first = RightsRegistry([primary, secondary])
    second = RightsRegistry([secondary, primary])
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.require("fixture-rights", "evaluate") == primary
    with pytest.raises(PermissionError, match="review"):
        first.require("fixture-rights", "publish_weights")
    with pytest.raises(PermissionError, match="deny"):
        first.require("fixture-rights", "redistribute_audio")


def test_declared_rights_registry_digest_is_verified(tmp_path: Path) -> None:
    registry = RightsRegistry([rights_record()])
    source = tmp_path / "rights-v2.json"
    payload = registry.as_dict()
    payload["registryDigest"] = "0" * 64
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        RightsRegistry.load(source)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["assets"][0].__setitem__("assetId", 7), "assetId"),
        (lambda payload: payload["assets"][0].pop("evaluate"), "evaluate is required"),
        (
            lambda payload: payload["assets"][0].__setitem__("unexpected", True),
            "unknown rights asset field",
        ),
        (lambda payload: payload.__setitem__("schemaVersion", "3.0.0"), "unsupported"),
    ),
)
def test_v2_rights_loader_rejects_coercion_missing_fields_and_unknowns(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    payload = RightsRegistry([rights_record()]).as_dict()
    mutation(payload)
    source = tmp_path / "rights-v2.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        RightsRegistry.load(source)


def test_rights_registry_rejects_empty_asset_set() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        RightsRegistry([])


def test_legacy_rights_registry_migrates_missing_operations_to_review(tmp_path: Path) -> None:
    source = tmp_path / "rights-v1.json"
    source.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0.0",
                "assets": [
                    {
                        "assetId": "legacy",
                        "sourceName": "Legacy",
                        "sourceUrl": "https://example.invalid/source",
                        "version": "1",
                        "licenseName": "Legacy",
                        "licenseUrl": "https://example.invalid/license",
                        "train": "allow",
                        "deriveFeatures": "allow",
                        "redistributeRaw": "deny",
                        "exportSpeakerId": "deny",
                        "attribution": "Legacy",
                        "reviewedAt": "2026-08-29",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = RightsRegistry.load(source)
    record = registry.records["legacy"]
    assert record.permission("evaluate") == "review"
    assert record.permission("redistribute_audio") == "deny"
    assert record.acquired_at == "2026-08-29"
    with pytest.raises(PermissionError, match="review"):
        registry.require("legacy", "publish_text")


def test_manifest_requires_matching_registry_and_known_assets() -> None:
    registry = RightsRegistry([rights_record()])
    dataset = manifest(
        utterance("train", "train", "1"),
        rights_registry_digest=registry.digest,
    )
    assert dataset.require_operation_rights(registry, "evaluate") == dataset.records
    with pytest.raises(ValueError, match="rights_registry_digest is required"):
        replace(dataset, rights_registry_digest=None).require_operation_rights(registry, "evaluate")
    with pytest.raises(PermissionError, match="review"):
        dataset.require_operation_rights(registry, "publish_weights")
    with pytest.raises(ValueError, match="digest"):
        replace(dataset, rights_registry_digest="0" * 64).require_operation_rights(
            registry, "evaluate"
        )
    unknown = manifest(
        replace(dataset.records[0], rights_asset_id=None),
        rights_registry_digest=registry.digest,
    )
    with pytest.raises(PermissionError, match="unknown rights asset"):
        unknown.require_operation_rights(registry, "evaluate")


@pytest.mark.parametrize("split", (None, "dev", "calibration", "test", "regression-exposed"))
def test_every_reference_bearing_training_loader_requires_explicit_train(split: object) -> None:
    row = ranker_row(split)
    with pytest.raises(ValueError, match="forbidden split"):
        ranker_example_from_row(row, line_number=1)
    with pytest.raises(ValueError, match="forbidden split"):
        example_from_row(row, line_number=1)
    with pytest.raises(ValueError, match="forbidden split"):
        fusion_example_from_row(row, line_number=1)


def test_ranker_split_isolation_bypass_is_rejected_at_runtime() -> None:
    with pytest.raises(ValueError, match="cannot be disabled"):
        ranker_example_from_row(
            ranker_row("train"),
            line_number=1,
            require_train_split=False,  # type: ignore[arg-type]
        )


def test_single_candidate_non_train_row_cannot_bypass_loader_guard(tmp_path: Path) -> None:
    row = ranker_row("test")
    row["candidates"] = [row["candidates"][0]]  # type: ignore[index]
    source = tmp_path / "ranker.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden split"):
        load_jsonl_examples(source)
