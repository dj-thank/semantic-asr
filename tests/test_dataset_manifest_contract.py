from __future__ import annotations

import pytest

from semantic_asr.experiment import DatasetManifest, UtteranceRecord


def record(
    sample_id: str,
    split: str = "train",
    audio_sha256: str = "a" * 64,
) -> UtteranceRecord:
    return UtteranceRecord(
        sample_id=sample_id,
        split=split,  # type: ignore[arg-type]
        audio_sha256=audio_sha256,
        reference="参照文",
    )


@pytest.mark.parametrize(
    "invalid",
    ("", "train ", " train", "TRAIN", "development", "regression-exposed", None),
)
def test_record_rejects_unknown_or_noncanonical_split(invalid: object) -> None:
    with pytest.raises(ValueError, match="split must be exactly one of"):
        record("sample", invalid)  # type: ignore[arg-type]


def test_manifest_split_rejects_invalid_query_instead_of_silently_returning_empty() -> None:
    manifest = DatasetManifest((record("sample"),), "fixture", "v1")
    with pytest.raises(ValueError, match="requested split must be exactly one of"):
        manifest.split("train ")  # type: ignore[arg-type]


def test_record_rejects_noncanonical_uppercase_sha_without_rewriting_digest() -> None:
    with pytest.raises(ValueError, match="lowercase hexadecimal SHA-256"):
        record("sample", audio_sha256="AB" * 32)


def test_same_audio_in_different_splits_is_always_reported() -> None:
    digest = "ab" * 32
    manifest = DatasetManifest(
        (
            record("train", "train", digest),
            record("test", "test", digest),
        ),
        "fixture",
        "v1",
    )
    findings = manifest.leakage_findings(reference_near_duplicate=False)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == "audio-sha256"
    assert finding.value == digest
    assert finding.splits == ("test", "train")
    assert finding.sample_ids == ("test", "train")
