"""Prepare explicit-kana phonetic source manifests with separate reading receipts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .contracts import sha256_json
from .japanese_phonetic_targets import JapanesePronunciationPolicy
from .phonetic_dataset import file_sha256
from .phonetic_feature_export import PhoneticSourceItem
from .reading_provenance import (
    JapaneseReadingProposal,
    JapaneseReadingReview,
    JapaneseReadingReviewLedger,
    ReadingProviderIdentity,
    ReadingResolutionPolicy,
    ResolvedJapaneseReading,
    resolve_japanese_reading,
)

SplitName = Literal["train", "calibration", "test"]


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class ReadingPreparationInputItem:
    utterance_id: str
    split: SplitName
    audio_path: str
    audio_sha256: str
    sample_rate: int
    segment_start_ms: int
    segment_end_ms: int
    transcript: str
    explicit_reading: str | None
    speaker_id: str
    source_id: str
    rights_decision: str
    license_id: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reading input schema_version must be '1'")
        if not self.utterance_id or not self.transcript:
            raise ValueError("reading input requires utterance_id and transcript")
        path = Path(self.audio_path)
        if path.is_absolute() or ".." in path.parts or not self.audio_path:
            raise ValueError("audio_path must be a non-traversing relative path")
        if not _is_sha256(self.audio_sha256):
            raise ValueError("audio_sha256 must be a SHA-256 value")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("reading input split is invalid")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if (
            isinstance(self.segment_start_ms, bool)
            or isinstance(self.segment_end_ms, bool)
            or self.segment_start_ms < 0
            or self.segment_end_ms <= self.segment_start_ms
        ):
            raise ValueError("reading input segment range is invalid")
        if self.explicit_reading is not None and not self.explicit_reading:
            raise ValueError("explicit_reading must be non-empty when supplied")
        if not self.speaker_id or not self.source_id:
            raise ValueError("speaker_id and source_id are required")
        if self.rights_decision != "allow":
            raise ValueError("reading preparation requires rights_decision='allow'")
        if not self.license_id:
            raise ValueError("license_id is required")

    @property
    def source_text_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(self.transcript.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ReadingPreparationManifest:
    path: Path
    split: SplitName
    items: tuple[ReadingPreparationInputItem, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("reading preparation manifest must not be empty")
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA-256 value")
        if any(item.split != self.split for item in self.items):
            raise ValueError("reading preparation manifest mixes splits")
        if len({item.utterance_id for item in self.items}) != len(self.items):
            raise ValueError("utterance IDs must be unique")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "path": self.path.name,
                "split": self.split,
                "itemDigests": [item.digest for item in self.items],
                "manifestSha256": self.manifest_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedReadingReceipt:
    utterance_id: str
    input_item_digest: str
    input_manifest_digest: str
    source_text_sha256: str
    output_source_item_digest: str
    resolved_reading: ResolvedJapaneseReading
    pronunciation_policy_digest: str
    resolution_policy_digest: str
    machine_proposal_digest: str | None
    review_ledger_digest: str | None
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1" or not self.utterance_id:
            raise ValueError("prepared reading receipt schema/utterance is invalid")
        for name in (
            "input_item_digest",
            "input_manifest_digest",
            "source_text_sha256",
            "output_source_item_digest",
            "pronunciation_policy_digest",
            "resolution_policy_digest",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 value")
        for name in ("machine_proposal_digest", "review_ledger_digest"):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 value")
        if self.resolved_reading.source_text_sha256 != self.source_text_sha256:
            raise ValueError("resolved reading is bound to different source text")
        if self.resolved_reading.pronunciation_policy_digest != (
            self.pronunciation_policy_digest
        ):
            raise ValueError("resolved reading uses a different pronunciation policy")
        if self.resolved_reading.resolution_policy_digest != self.resolution_policy_digest:
            raise ValueError("resolved reading uses a different resolution policy")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "utteranceId": self.utterance_id,
                "inputItemDigest": self.input_item_digest,
                "inputManifestDigest": self.input_manifest_digest,
                "sourceTextSha256": self.source_text_sha256,
                "outputSourceItemDigest": self.output_source_item_digest,
                "resolvedReadingDigest": self.resolved_reading.digest,
                "pronunciationPolicyDigest": self.pronunciation_policy_digest,
                "resolutionPolicyDigest": self.resolution_policy_digest,
                "machineProposalDigest": self.machine_proposal_digest,
                "reviewLedgerDigest": self.review_ledger_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class ReadingPreparationResult:
    output_manifest: Path
    receipt_manifest: Path
    output_manifest_sha256: str
    receipt_manifest_sha256: str
    input_manifest_digest: str
    pronunciation_policy_digest: str
    resolution_policy_digest: str
    item_count: int
    origin_counts: tuple[tuple[str, int], ...]
    receipt_digests: tuple[str, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1" or self.item_count < 1:
            raise ValueError("reading preparation result is invalid")
        for name in (
            "output_manifest_sha256",
            "receipt_manifest_sha256",
            "input_manifest_digest",
            "pronunciation_policy_digest",
            "resolution_policy_digest",
            *(),
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 value")
        if any(not _is_sha256(value) for value in self.receipt_digests):
            raise ValueError("receipt_digests contain an invalid SHA-256")
        if sum(count for _, count in self.origin_counts) != self.item_count:
            raise ValueError("origin_counts do not sum to item_count")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "output_manifest": self.output_manifest.name,
                "receipt_manifest": self.receipt_manifest.name,
            }
        )


def _input_row(value: dict[str, object], line_number: int) -> ReadingPreparationInputItem:
    expected = {
        "schemaVersion",
        "utteranceId",
        "split",
        "audioPath",
        "audioSha256",
        "sampleRate",
        "segmentStartMs",
        "segmentEndMs",
        "transcript",
        "explicitReading",
        "speakerId",
        "sourceId",
        "rightsDecision",
        "licenseId",
    }
    if set(value) != expected:
        raise ValueError(
            f"reading input row {line_number} has non-exact schema; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    explicit = value["explicitReading"]
    if explicit is not None and not isinstance(explicit, str):
        raise ValueError("explicitReading must be a string or null")
    return ReadingPreparationInputItem(
        schema_version=str(value["schemaVersion"]),
        utterance_id=str(value["utteranceId"]),
        split=str(value["split"]),  # type: ignore[arg-type]
        audio_path=str(value["audioPath"]),
        audio_sha256=str(value["audioSha256"]),
        sample_rate=_exact_integer(value["sampleRate"], name="sampleRate"),
        segment_start_ms=_exact_integer(value["segmentStartMs"], name="segmentStartMs"),
        segment_end_ms=_exact_integer(value["segmentEndMs"], name="segmentEndMs"),
        transcript=str(value["transcript"]),
        explicit_reading=explicit,
        speaker_id=str(value["speakerId"]),
        source_id=str(value["sourceId"]),
        rights_decision=str(value["rightsDecision"]),
        license_id=str(value["licenseId"]),
    )


def load_reading_preparation_manifest(
    path: str | Path,
    *,
    split: SplitName,
    maximum_items: int = 2_000_000,
    maximum_transcript_characters: int = 100_000,
) -> ReadingPreparationManifest:
    if isinstance(maximum_items, bool) or maximum_items < 1:
        raise ValueError("maximum_items must be positive")
    if (
        isinstance(maximum_transcript_characters, bool)
        or maximum_transcript_characters < 1
    ):
        raise ValueError("maximum_transcript_characters must be positive")
    source = Path(path)
    items = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"reading input row {line_number} must be an object")
            item = _input_row(value, line_number)
            if item.split != split:
                raise ValueError(
                    f"reading input row {line_number} declares {item.split!r}, "
                    f"expected {split!r}"
                )
            if len(item.transcript) > maximum_transcript_characters:
                raise ValueError("transcript exceeds maximum_transcript_characters")
            items.append(item)
            if len(items) > maximum_items:
                raise ValueError("reading input exceeds maximum_items")
    return ReadingPreparationManifest(
        path=source,
        split=split,
        items=tuple(items),
        manifest_sha256=file_sha256(source),
    )


def load_machine_reading_proposals(
    path: str | Path,
    *,
    pronunciation_policy: JapanesePronunciationPolicy | None = None,
) -> dict[str, JapaneseReadingProposal]:
    policy = pronunciation_policy or JapanesePronunciationPolicy()
    expected = {
        "schemaVersion",
        "utteranceId",
        "sourceTextSha256",
        "normalizedReading",
        "readingSha256",
        "pronunciationPolicyDigest",
        "provider",
        "metadata",
    }
    provider_expected = {
        "schemaVersion",
        "providerId",
        "providerRevision",
        "providerConfigDigest",
        "providerArtifactSha256",
        "resourceArtifactSha256",
    }
    output: dict[str, JapaneseReadingProposal] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError(f"machine proposal row {line_number} schema is not exact")
            provider = value["provider"]
            if not isinstance(provider, dict) or set(provider) != provider_expected:
                raise ValueError(f"machine proposal row {line_number} provider schema is not exact")
            utterance_id = str(value["utteranceId"])
            proposal = JapaneseReadingProposal(
                schema_version=str(value["schemaVersion"]),
                source_text_sha256=str(value["sourceTextSha256"]),
                normalized_reading=str(value["normalizedReading"]),
                reading_sha256=str(value["readingSha256"]),
                origin="machine-proposed",
                pronunciation_policy_digest=str(value["pronunciationPolicyDigest"]),
                provider=ReadingProviderIdentity(
                    schema_version=str(provider["schemaVersion"]),
                    provider_id=str(provider["providerId"]),
                    provider_revision=str(provider["providerRevision"]),
                    provider_config_digest=str(provider["providerConfigDigest"]),
                    provider_artifact_sha256=str(provider["providerArtifactSha256"]),
                    resource_artifact_sha256=str(provider["resourceArtifactSha256"]),
                ),
                utterance_id=utterance_id,
                metadata=dict(value["metadata"]),  # type: ignore[arg-type]
            )
            if proposal.pronunciation_policy_digest != policy.digest:
                raise ValueError("machine proposal uses a different pronunciation policy")
            if utterance_id in output:
                raise ValueError("machine proposal utterance IDs must be unique")
            output[utterance_id] = proposal
    return output


def load_reading_review_ledger(
    path: str | Path,
    *,
    revision: str,
    review_protocol_revision: str,
    review_batch_manifest_sha256: str,
) -> JapaneseReadingReviewLedger:
    expected = {
        "schemaVersion",
        "proposalDigest",
        "sourceTextSha256",
        "proposedReadingSha256",
        "disposition",
        "approvedReading",
        "approvedReadingSha256",
        "reviewerIdHash",
        "reviewProtocolRevision",
        "reviewManifestSha256",
        "pronunciationPolicyDigest",
        "metadata",
    }
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError(f"reading review row {line_number} schema is not exact")
            approved = value["approvedReading"]
            approved_hash = value["approvedReadingSha256"]
            if approved is not None and not isinstance(approved, str):
                raise ValueError("approvedReading must be a string or null")
            if approved_hash is not None and not isinstance(approved_hash, str):
                raise ValueError("approvedReadingSha256 must be a string or null")
            metadata = value["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("reading review metadata must be an object")
            records.append(
                JapaneseReadingReview(
                    schema_version=str(value["schemaVersion"]),
                    proposal_digest=str(value["proposalDigest"]),
                    source_text_sha256=str(value["sourceTextSha256"]),
                    proposed_reading_sha256=str(value["proposedReadingSha256"]),
                    disposition=str(value["disposition"]),  # type: ignore[arg-type]
                    approved_reading=approved,
                    approved_reading_sha256=approved_hash,
                    reviewer_id_hash=str(value["reviewerIdHash"]),
                    review_protocol_revision=str(value["reviewProtocolRevision"]),
                    review_manifest_sha256=str(value["reviewManifestSha256"]),
                    pronunciation_policy_digest=str(value["pronunciationPolicyDigest"]),
                    metadata=metadata,
                )
            )
    return JapaneseReadingReviewLedger(
        revision=revision,
        source_manifest_sha256=review_batch_manifest_sha256,
        records=tuple(records),
        review_protocol_revision=review_protocol_revision,
    )


def _source_row(item: PhoneticSourceItem) -> dict[str, object]:
    return {
        "schemaVersion": item.schema_version,
        "utteranceId": item.utterance_id,
        "split": item.split,
        "audioPath": item.audio_path,
        "audioSha256": item.audio_sha256,
        "sampleRate": item.sample_rate,
        "segmentStartMs": item.segment_start_ms,
        "segmentEndMs": item.segment_end_ms,
        "reading": item.reading,
        "speakerId": item.speaker_id,
        "sourceId": item.source_id,
        "rightsDecision": item.rights_decision,
        "licenseId": item.license_id,
    }


def _receipt_row(receipt: PreparedReadingReceipt) -> dict[str, object]:
    return {
        "schemaVersion": receipt.schema_version,
        "utteranceId": receipt.utterance_id,
        "inputItemDigest": receipt.input_item_digest,
        "inputManifestDigest": receipt.input_manifest_digest,
        "sourceTextSha256": receipt.source_text_sha256,
        "outputSourceItemDigest": receipt.output_source_item_digest,
        "resolvedReading": asdict(receipt.resolved_reading),
        "resolvedReadingDigest": receipt.resolved_reading.digest,
        "pronunciationPolicyDigest": receipt.pronunciation_policy_digest,
        "resolutionPolicyDigest": receipt.resolution_policy_digest,
        "machineProposalDigest": receipt.machine_proposal_digest,
        "reviewLedgerDigest": receipt.review_ledger_digest,
        "receiptDigest": receipt.digest,
    }


def prepare_phonetic_source_manifest(
    manifest: ReadingPreparationManifest,
    output_path: str | Path,
    *,
    machine_proposals: Mapping[str, JapaneseReadingProposal] | None = None,
    review_ledger: JapaneseReadingReviewLedger | None = None,
    pronunciation_policy: JapanesePronunciationPolicy | None = None,
    resolution_policy: ReadingResolutionPolicy | None = None,
    allow_output: bool,
) -> ReadingPreparationResult:
    if not allow_output:
        raise PermissionError("reading preparation requires allow_output=True")
    pronunciation_policy = pronunciation_policy or JapanesePronunciationPolicy()
    resolution_policy = resolution_policy or ReadingResolutionPolicy()
    machine_proposals = dict(machine_proposals or {})
    expected_machine_ids = {
        item.utterance_id for item in manifest.items if item.explicit_reading is None
    }
    unknown = set(machine_proposals) - expected_machine_ids
    if unknown:
        raise ValueError(f"machine proposals contain unknown utterance IDs: {sorted(unknown)}")
    output = Path(output_path)
    if output.suffix != ".jsonl":
        raise ValueError("prepared source manifest must use the .jsonl suffix")
    receipt_path = output.with_suffix(output.suffix + ".reading-receipts.jsonl")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("prepared output or reading receipt manifest already exists")

    source_items = []
    receipts = []
    origin_counts: dict[str, int] = {}
    for item in manifest.items:
        proposal = machine_proposals.get(item.utterance_id)
        resolved = resolve_japanese_reading(
            item.transcript,
            split=item.split,
            explicit_reading=item.explicit_reading,
            machine_proposal=proposal,
            review_ledger=review_ledger,
            pronunciation_policy=pronunciation_policy,
            resolution_policy=resolution_policy,
            utterance_id=item.utterance_id,
        )
        source_item = PhoneticSourceItem(
            utterance_id=item.utterance_id,
            split=item.split,
            audio_path=item.audio_path,
            audio_sha256=item.audio_sha256,
            sample_rate=item.sample_rate,
            segment_start_ms=item.segment_start_ms,
            segment_end_ms=item.segment_end_ms,
            reading=resolved.normalized_reading,
            speaker_id=item.speaker_id,
            source_id=item.source_id,
            rights_decision=item.rights_decision,
            license_id=item.license_id,
        )
        receipt = PreparedReadingReceipt(
            utterance_id=item.utterance_id,
            input_item_digest=item.digest,
            input_manifest_digest=manifest.digest,
            source_text_sha256=item.source_text_sha256,
            output_source_item_digest=source_item.digest,
            resolved_reading=resolved,
            pronunciation_policy_digest=pronunciation_policy.digest,
            resolution_policy_digest=resolution_policy.digest,
            machine_proposal_digest=(None if proposal is None else proposal.digest),
            review_ledger_digest=(None if review_ledger is None else review_ledger.digest),
        )
        source_items.append(source_item)
        receipts.append(receipt)
        origin_counts[resolved.origin] = origin_counts.get(resolved.origin, 0) + 1

    _atomic_write(
        output,
        "".join(
            json.dumps(_source_row(item), ensure_ascii=False, sort_keys=True) + "\n"
            for item in source_items
        ),
    )
    _atomic_write(
        receipt_path,
        "".join(
            json.dumps(_receipt_row(receipt), ensure_ascii=False, sort_keys=True) + "\n"
            for receipt in receipts
        ),
    )
    return ReadingPreparationResult(
        output_manifest=output,
        receipt_manifest=receipt_path,
        output_manifest_sha256=file_sha256(output),
        receipt_manifest_sha256=file_sha256(receipt_path),
        input_manifest_digest=manifest.digest,
        pronunciation_policy_digest=pronunciation_policy.digest,
        resolution_policy_digest=resolution_policy.digest,
        item_count=len(source_items),
        origin_counts=tuple(sorted(origin_counts.items())),
        receipt_digests=tuple(receipt.digest for receipt in receipts),
    )
