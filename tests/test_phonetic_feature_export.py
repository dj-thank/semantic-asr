from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

numpy = pytest.importorskip("numpy")

from semantic_asr.audio_posterior_adapters import canonical_audio_sha256  # noqa: E402
from semantic_asr.joint_phonetic_runtime_optional import (  # noqa: E402
    FrozenAudioFeatureConfig,
    FrozenFeatureMatrix,
)
from semantic_asr.phonetic_dataset import (  # noqa: E402
    file_sha256,
    load_phonetic_feature_manifest,
)
from semantic_asr.phonetic_feature_export import (  # noqa: E402
    LoadedSourceRecording,
    PhoneticFeatureExporter,
    PhoneticSourceResourcePolicy,
    load_phonetic_source_manifest,
)

REVISION = "1" * 40


def backend_config() -> FrozenAudioFeatureConfig:
    return FrozenAudioFeatureConfig(
        model_id="frozen-test-encoder",
        model_revision=REVISION,
        layer_index=4,
        sample_rate=1_000,
        feature_dimension=4,
        frame_stride_ms=5.0,
    )


class FakeLoader:
    def __init__(self, file_sha256_value: str, samples=None) -> None:
        self.file_sha256_value = file_sha256_value
        self.samples = tuple(samples or (index / 100.0 for index in range(100)))
        self.calls = 0

    def load(self, path):
        self.calls += 1
        return LoadedSourceRecording(
            samples=self.samples,
            sample_rate=1_000,
            file_sha256=self.file_sha256_value,
            source_name=Path(path).name,
        )


class FakeBackend:
    def __init__(self, *, fail_after: int | None = None, wrong_audio: bool = False) -> None:
        self.config = backend_config()
        self.fail_after = fail_after
        self.wrong_audio = wrong_audio
        self.calls = 0

    def extract_features(self, samples, *, sample_rate, source_audio_sha256):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated exporter interruption")
        assert sample_rate == self.config.sample_rate
        assert samples
        values = tuple(
            tuple(float(frame + column) / 10.0 for column in range(4))
            for frame in range(12)
        )
        return FrozenFeatureMatrix(
            values=values,
            source_audio_sha256=("f" * 64 if self.wrong_audio else source_audio_sha256),
            feature_config_digest=self.config.digest,
        )


def source_row(
    split: str,
    identifier: str,
    audio_sha256: str,
    *,
    start_ms: int = 0,
    end_ms: int = 20,
    reading: str = "まだ",
    rights: str = "allow",
):
    return {
        "schemaVersion": "1",
        "utteranceId": identifier,
        "split": split,
        "audioPath": "recording.wav",
        "audioSha256": audio_sha256,
        "sampleRate": 1_000,
        "segmentStartMs": start_ms,
        "segmentEndMs": end_ms,
        "reading": reading,
        "speakerId": "speaker-1",
        "sourceId": "recording-1",
        "rightsDecision": rights,
        "licenseId": "fixture-license",
    }


def write_source(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def source_manifest(tmp_path: Path, rows):
    path = tmp_path / "source.jsonl"
    write_source(path, rows)
    return load_phonetic_source_manifest(
        path,
        split="train",
        resources=PhoneticSourceResourcePolicy(maximum_items=10),
    )


def test_export_writes_trainer_compatible_manifest_and_dual_provenance(
    tmp_path: Path,
) -> None:
    audio_file_sha = hashlib.sha256(b"recording-file").hexdigest()
    source = source_manifest(
        tmp_path,
        (source_row("train", "utt-1", audio_file_sha),),
    )
    loader = FakeLoader(audio_file_sha)
    backend = FakeBackend()
    exporter = PhoneticFeatureExporter(
        feature_backend=backend,
        audio_loader=loader,
    )
    output = tmp_path / "derived" / "train.jsonl"

    result = exporter.export(source, output, allow_derived_export=True)

    assert result.item_count == 1
    assert backend.calls == 1
    loaded = load_phonetic_feature_manifest(
        output,
        split="train",
        phone_inventory=exporter.phone_inventory,
        mora_inventory=exporter.mora_inventory,
    )
    item = loaded.items[0]
    expected_clip_sha = canonical_audio_sha256(loader.samples[:20], 1_000)
    assert item.source_audio_sha256 == expected_clip_sha
    assert item.source_audio_sha256 != audio_file_sha
    assert item.feature_revision == exporter.feature_revision
    feature = output.parent / item.feature_path
    values = numpy.load(feature, allow_pickle=False)
    assert values.shape == (12, 4)
    assert file_sha256(feature) == item.feature_sha256
    sidecar = json.loads(feature.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    assert sidecar["source_recording_file_sha256"] == audio_file_sha
    assert sidecar["source_clip_sha256"] == expected_clip_sha
    assert sidecar["sample_start"] == 0
    assert sidecar["sample_end"] == 20
    assert sidecar["receiptDigest"] == result.receipt_digests[0]


def test_export_requires_explicit_derived_data_authorization(tmp_path: Path) -> None:
    audio_sha = hashlib.sha256(b"audio").hexdigest()
    source = source_manifest(tmp_path, (source_row("train", "utt-1", audio_sha),))
    exporter = PhoneticFeatureExporter(
        feature_backend=FakeBackend(),
        audio_loader=FakeLoader(audio_sha),
    )

    with pytest.raises(PermissionError, match="allow_derived_export"):
        exporter.export(source, tmp_path / "out.jsonl", allow_derived_export=False)


def test_resume_uses_verified_prefix_without_recomputing_it(tmp_path: Path) -> None:
    audio_sha = hashlib.sha256(b"audio").hexdigest()
    source = source_manifest(
        tmp_path,
        (
            source_row("train", "utt-1", audio_sha, start_ms=0, end_ms=20),
            source_row("train", "utt-2", audio_sha, start_ms=20, end_ms=40, reading="また"),
        ),
    )
    output = tmp_path / "derived" / "train.jsonl"
    first_backend = FakeBackend(fail_after=1)
    first = PhoneticFeatureExporter(
        feature_backend=first_backend,
        audio_loader=FakeLoader(audio_sha),
    )

    with pytest.raises(RuntimeError, match="interruption"):
        first.export(source, output, allow_derived_export=True)

    partial = output.with_suffix(".jsonl.partial")
    assert partial.exists()
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 1

    second_backend = FakeBackend()
    second = PhoneticFeatureExporter(
        feature_backend=second_backend,
        audio_loader=FakeLoader(audio_sha),
    )
    result = second.export(source, output, allow_derived_export=True, resume=True)

    assert result.item_count == 2
    assert second_backend.calls == 1
    assert output.exists()
    assert not partial.exists()


def test_completed_export_is_a_verified_noop(tmp_path: Path) -> None:
    audio_sha = hashlib.sha256(b"audio").hexdigest()
    source = source_manifest(tmp_path, (source_row("train", "utt-1", audio_sha),))
    output = tmp_path / "derived" / "train.jsonl"
    first = PhoneticFeatureExporter(
        feature_backend=FakeBackend(),
        audio_loader=FakeLoader(audio_sha),
    )
    first_result = first.export(source, output, allow_derived_export=True)
    second_backend = FakeBackend()
    second = PhoneticFeatureExporter(
        feature_backend=second_backend,
        audio_loader=FakeLoader(audio_sha),
    )

    second_result = second.export(source, output, allow_derived_export=True)

    assert second_result.output_manifest_sha256 == first_result.output_manifest_sha256
    assert second_backend.calls == 0


def test_recording_and_feature_matrix_hash_mismatches_fail_closed(tmp_path: Path) -> None:
    audio_sha = hashlib.sha256(b"expected").hexdigest()
    source = source_manifest(tmp_path, (source_row("train", "utt-1", audio_sha),))

    with pytest.raises(ValueError, match="recording SHA-256 mismatch"):
        PhoneticFeatureExporter(
            feature_backend=FakeBackend(),
            audio_loader=FakeLoader(hashlib.sha256(b"other").hexdigest()),
        ).export(source, tmp_path / "bad-recording.jsonl", allow_derived_export=True)

    with pytest.raises(ValueError, match="different audio clip"):
        PhoneticFeatureExporter(
            feature_backend=FakeBackend(wrong_audio=True),
            audio_loader=FakeLoader(audio_sha),
        ).export(source, tmp_path / "bad-feature.jsonl", allow_derived_export=True)


def test_source_manifest_rejects_rights_schema_and_traversal(tmp_path: Path) -> None:
    audio_sha = hashlib.sha256(b"audio").hexdigest()
    denied = tmp_path / "denied.jsonl"
    write_source(denied, (source_row("train", "utt", audio_sha, rights="review"),))
    with pytest.raises(ValueError, match="rights_decision='allow'"):
        load_phonetic_source_manifest(denied, split="train")

    unknown = source_row("train", "utt", audio_sha)
    unknown["unknown"] = True
    path = tmp_path / "unknown.jsonl"
    write_source(path, (unknown,))
    with pytest.raises(ValueError, match="non-exact schema"):
        load_phonetic_source_manifest(path, split="train")

    traversal = source_row("train", "utt", audio_sha)
    traversal["audioPath"] = "../recording.wav"
    path = tmp_path / "traversal.jsonl"
    write_source(path, (traversal,))
    with pytest.raises(ValueError, match="non-traversing"):
        load_phonetic_source_manifest(path, split="train")
