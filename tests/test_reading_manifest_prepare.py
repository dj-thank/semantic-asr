from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semantic_asr.japanese_phonetic_targets import JapanesePronunciationPolicy
from semantic_asr.phonetic_dataset import file_sha256
from semantic_asr.reading_manifest_prepare import (
    load_machine_reading_proposals,
    load_reading_preparation_manifest,
    load_reading_review_ledger,
    prepare_phonetic_source_manifest,
)
from semantic_asr.reading_provenance import (
    JapaneseReadingProposal,
    JapaneseReadingReview,
    ReadingProviderIdentity,
    ReadingResolutionPolicy,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def input_row(
    identifier: str,
    *,
    split: str = "train",
    transcript: str = "学校へ行く",
    explicit_reading: str | None = "ガッコウヘイク",
):
    return {
        "schemaVersion": "1",
        "utteranceId": identifier,
        "split": split,
        "audioPath": f"audio/{identifier}.wav",
        "audioSha256": sha(f"audio-{identifier}"),
        "sampleRate": 16000,
        "segmentStartMs": 100,
        "segmentEndMs": 1200,
        "transcript": transcript,
        "explicitReading": explicit_reading,
        "speakerId": f"speaker-{identifier}",
        "sourceId": f"source-{identifier}",
        "rightsDecision": "allow",
        "licenseId": "fixture-license",
    }


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def identity() -> ReadingProviderIdentity:
    return ReadingProviderIdentity(
        provider_id="fixture-g2p",
        provider_revision="r1",
        provider_config_digest=sha("config"),
        provider_artifact_sha256=sha("provider"),
        resource_artifact_sha256=sha("dictionary"),
    )


def proposal_row(identifier: str, transcript: str, reading: str):
    proposal = JapaneseReadingProposal.machine(
        transcript,
        reading,
        provider=identity(),
        utterance_id=identifier,
    )
    return {
        "schemaVersion": proposal.schema_version,
        "utteranceId": identifier,
        "sourceTextSha256": proposal.source_text_sha256,
        "normalizedReading": proposal.normalized_reading,
        "readingSha256": proposal.reading_sha256,
        "pronunciationPolicyDigest": proposal.pronunciation_policy_digest,
        "provider": {
            "schemaVersion": proposal.provider.schema_version,
            "providerId": proposal.provider.provider_id,
            "providerRevision": proposal.provider.provider_revision,
            "providerConfigDigest": proposal.provider.provider_config_digest,
            "providerArtifactSha256": proposal.provider.provider_artifact_sha256,
            "resourceArtifactSha256": proposal.provider.resource_artifact_sha256,
        },
        "metadata": {},
    }, proposal


def review_row(proposal: JapaneseReadingProposal, batch_digest: str):
    review = JapaneseReadingReview.create(
        proposal,
        disposition="approved",
        approved_reading=proposal.normalized_reading,
        reviewer_id_hash=sha("reviewer"),
        review_protocol_revision="r1",
        review_manifest_sha256=batch_digest,
    )
    return {
        "schemaVersion": review.schema_version,
        "proposalDigest": review.proposal_digest,
        "sourceTextSha256": review.source_text_sha256,
        "proposedReadingSha256": review.proposed_reading_sha256,
        "disposition": review.disposition,
        "approvedReading": review.approved_reading,
        "approvedReadingSha256": review.approved_reading_sha256,
        "reviewerIdHash": review.reviewer_id_hash,
        "reviewProtocolRevision": review.review_protocol_revision,
        "reviewManifestSha256": review.review_manifest_sha256,
        "pronunciationPolicyDigest": review.pronunciation_policy_digest,
        "metadata": {},
    }


def test_human_readings_prepare_exact_exporter_manifest_and_receipts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, (input_row("utt-1"),))
    source = load_reading_preparation_manifest(input_path, split="train")
    output = tmp_path / "prepared" / "train.jsonl"

    result = prepare_phonetic_source_manifest(
        source,
        output,
        allow_output=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    receipts = [
        json.loads(line)
        for line in result.receipt_manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [
        {
            "schemaVersion": "1",
            "utteranceId": "utt-1",
            "split": "train",
            "audioPath": "audio/utt-1.wav",
            "audioSha256": sha("audio-utt-1"),
            "sampleRate": 16000,
            "segmentStartMs": 100,
            "segmentEndMs": 1200,
            "reading": "ガッコウヘイク",
            "speakerId": "speaker-utt-1",
            "sourceId": "source-utt-1",
            "rightsDecision": "allow",
            "licenseId": "fixture-license",
        }
    ]
    assert result.output_manifest_sha256 == file_sha256(output)
    assert result.receipt_manifest_sha256 == file_sha256(result.receipt_manifest)
    assert result.origin_counts == (("human-explicit", 1),)
    assert receipts[0]["sourceTextSha256"] == sha("学校へ行く")
    assert "学校へ行く" not in result.receipt_manifest.read_text(encoding="utf-8")
    assert receipts[0]["receiptDigest"] == result.receipt_digests[0]


def test_unreviewed_machine_train_requires_explicit_policy(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(
        input_path,
        (input_row("utt-1", explicit_reading=None),),
    )
    source = load_reading_preparation_manifest(input_path, split="train")
    row, _ = proposal_row("utt-1", "学校へ行く", "ガッコウヘイク")
    proposal_path = tmp_path / "proposals.jsonl"
    write_jsonl(proposal_path, (row,))
    proposals = load_machine_reading_proposals(proposal_path)

    with pytest.raises(ValueError, match="disabled by policy"):
        prepare_phonetic_source_manifest(
            source,
            tmp_path / "rejected.jsonl",
            machine_proposals=proposals,
            allow_output=True,
        )

    result = prepare_phonetic_source_manifest(
        source,
        tmp_path / "accepted.jsonl",
        machine_proposals=proposals,
        resolution_policy=ReadingResolutionPolicy(
            allow_unreviewed_machine_train=True,
        ),
        allow_output=True,
    )

    assert result.origin_counts == (("machine-proposed", 1),)


def test_locked_test_requires_review_and_emits_reviewed_origin(tmp_path: Path) -> None:
    input_path = tmp_path / "test-input.jsonl"
    write_jsonl(
        input_path,
        (input_row("utt-1", split="test", explicit_reading=None),),
    )
    source = load_reading_preparation_manifest(input_path, split="test")
    proposal_payload, proposal = proposal_row(
        "utt-1", "学校へ行く", "ガッコウヘイク"
    )
    proposal_path = tmp_path / "proposals.jsonl"
    write_jsonl(proposal_path, (proposal_payload,))
    proposals = load_machine_reading_proposals(proposal_path)

    with pytest.raises(ValueError, match="requires a reviewed machine reading"):
        prepare_phonetic_source_manifest(
            source,
            tmp_path / "unreviewed.jsonl",
            machine_proposals=proposals,
            allow_output=True,
        )

    batch_digest = sha("review-batch")
    review_path = tmp_path / "reviews.jsonl"
    write_jsonl(review_path, (review_row(proposal, batch_digest),))
    reviews = load_reading_review_ledger(
        review_path,
        revision="ledger-r1",
        review_protocol_revision="r1",
        review_batch_manifest_sha256=batch_digest,
    )
    result = prepare_phonetic_source_manifest(
        source,
        tmp_path / "reviewed.jsonl",
        machine_proposals=proposals,
        review_ledger=reviews,
        allow_output=True,
    )

    assert result.origin_counts == (("machine-reviewed", 1),)
    receipt = json.loads(
        result.receipt_manifest.read_text(encoding="utf-8").splitlines()[0]
    )
    assert receipt["reviewLedgerDigest"] == reviews.digest
    assert receipt["resolvedReading"]["eligible_for_locked_test"]


def test_unknown_machine_proposal_and_overwrite_are_rejected(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    write_jsonl(input_path, (input_row("utt-1"),))
    source = load_reading_preparation_manifest(input_path, split="train")
    _, unknown = proposal_row("other", "学校へ行く", "ガッコウヘイク")

    with pytest.raises(ValueError, match="unknown utterance IDs"):
        prepare_phonetic_source_manifest(
            source,
            tmp_path / "out.jsonl",
            machine_proposals={"other": unknown},
            allow_output=True,
        )

    output = tmp_path / "existing.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_phonetic_source_manifest(source, output, allow_output=True)


def test_input_schema_rights_and_split_are_strict(tmp_path: Path) -> None:
    unknown = input_row("utt-1")
    unknown["extra"] = True
    path = tmp_path / "unknown.jsonl"
    write_jsonl(path, (unknown,))
    with pytest.raises(ValueError, match="non-exact schema"):
        load_reading_preparation_manifest(path, split="train")

    denied = input_row("utt-1")
    denied["rightsDecision"] = "review"
    path = tmp_path / "denied.jsonl"
    write_jsonl(path, (denied,))
    with pytest.raises(ValueError, match="rights_decision='allow'"):
        load_reading_preparation_manifest(path, split="train")

    mismatch = input_row("utt-1", split="test")
    path = tmp_path / "mismatch.jsonl"
    write_jsonl(path, (mismatch,))
    with pytest.raises(ValueError, match="expected 'train'"):
        load_reading_preparation_manifest(path, split="train")


def test_machine_proposal_policy_digest_is_exact(tmp_path: Path) -> None:
    payload, _ = proposal_row("utt-1", "学校へ行く", "ガッコウヘイク")
    payload["pronunciationPolicyDigest"] = "f" * 64
    path = tmp_path / "proposal.jsonl"
    write_jsonl(path, (payload,))

    with pytest.raises(ValueError, match="different pronunciation policy"):
        load_machine_reading_proposals(
            path,
            pronunciation_policy=JapanesePronunciationPolicy(),
        )
