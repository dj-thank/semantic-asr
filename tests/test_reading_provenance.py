from __future__ import annotations

import hashlib

import pytest

from semantic_asr.japanese_phonetic_targets import JapanesePronunciationPolicy
from semantic_asr.reading_provenance import (
    CallableJapaneseReadingProvider,
    JapaneseReadingProposal,
    JapaneseReadingReview,
    JapaneseReadingReviewLedger,
    ReadingProviderIdentity,
    ReadingResolutionPolicy,
    resolve_japanese_reading,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def identity() -> ReadingProviderIdentity:
    return ReadingProviderIdentity(
        provider_id="frozen-g2p-fixture",
        provider_revision="fixture-r1",
        provider_config_digest=sha("config"),
        provider_artifact_sha256=sha("provider"),
        resource_artifact_sha256=sha("dictionary"),
    )


def proposal(text: str = "学校へ行く", reading: str = "ガッコウヘイク"):
    return JapaneseReadingProposal.machine(
        text,
        reading,
        provider=identity(),
        utterance_id="utt-1",
    )


def ledger(
    value: JapaneseReadingProposal,
    *,
    disposition: str = "approved",
    approved_reading: str = "ガッコウヘイク",
):
    review_batch = sha("review-batch")
    review = JapaneseReadingReview.create(
        value,
        disposition=disposition,  # type: ignore[arg-type]
        approved_reading=approved_reading,
        reviewer_id_hash=sha("reviewer-opaque-1"),
        review_protocol_revision="double-check-r1",
        review_manifest_sha256=review_batch,
    )
    return JapaneseReadingReviewLedger(
        revision="ledger-r1",
        source_manifest_sha256=review_batch,
        records=(review,),
        review_protocol_revision="double-check-r1",
    )


def test_human_explicit_reading_is_locked_test_eligible() -> None:
    resolved = resolve_japanese_reading(
        "学校へ行く",
        split="test",
        explicit_reading="がっこうへいく。",
        utterance_id="utt-1",
    )

    assert resolved.normalized_reading == "ガッコウヘイク"
    assert resolved.origin == "human-explicit"
    assert resolved.eligible_for_locked_test
    assert resolved.provider_digest is None
    assert resolved.review_digest is None


def test_unreviewed_machine_reading_requires_explicit_train_policy() -> None:
    value = proposal()

    with pytest.raises(ValueError, match="disabled by policy"):
        resolve_japanese_reading(
            "学校へ行く",
            split="train",
            machine_proposal=value,
            utterance_id="utt-1",
        )

    resolved = resolve_japanese_reading(
        "学校へ行く",
        split="train",
        machine_proposal=value,
        resolution_policy=ReadingResolutionPolicy(
            allow_unreviewed_machine_train=True,
        ),
        utterance_id="utt-1",
    )

    assert resolved.origin == "machine-proposed"
    assert not resolved.eligible_for_locked_test
    assert resolved.provider_digest == identity().digest


def test_calibration_and_test_reject_unreviewed_machine_readings() -> None:
    value = proposal()

    for split in ("calibration", "test"):
        with pytest.raises(ValueError, match="requires a reviewed machine reading"):
            resolve_japanese_reading(
                "学校へ行く",
                split=split,  # type: ignore[arg-type]
                machine_proposal=value,
                utterance_id="utt-1",
            )


def test_reviewed_machine_reading_is_bound_to_exact_proposal() -> None:
    value = proposal()
    resolved = resolve_japanese_reading(
        "学校へ行く",
        split="test",
        machine_proposal=value,
        review_ledger=ledger(value),
        utterance_id="utt-1",
    )

    assert resolved.origin == "machine-reviewed"
    assert resolved.eligible_for_locked_test
    assert resolved.review_digest is not None
    assert resolved.normalized_reading == "ガッコウヘイク"

    changed = proposal(reading="ガッコーヘイク")
    with pytest.raises(ValueError, match="requires a reviewed machine reading"):
        resolve_japanese_reading(
            "学校へ行く",
            split="test",
            machine_proposal=changed,
            review_ledger=ledger(value),
            utterance_id="utt-1",
        )


def test_corrected_review_records_the_changed_reading() -> None:
    value = proposal(reading="ガッコーヘイク")
    resolved = resolve_japanese_reading(
        "学校へ行く",
        split="test",
        machine_proposal=value,
        review_ledger=ledger(
            value,
            disposition="corrected",
            approved_reading="ガッコウヘイク",
        ),
        utterance_id="utt-1",
    )

    assert resolved.origin == "machine-reviewed"
    assert resolved.normalized_reading == "ガッコウヘイク"
    assert resolved.metadata["reviewDisposition"] == "corrected"


def test_rejected_review_fails_closed() -> None:
    value = proposal()
    review_batch = sha("review-batch")
    review = JapaneseReadingReview.create(
        value,
        disposition="rejected",
        approved_reading=None,
        reviewer_id_hash=sha("reviewer"),
        review_protocol_revision="r1",
        review_manifest_sha256=review_batch,
    )
    rejected = JapaneseReadingReviewLedger(
        revision="ledger-r1",
        source_manifest_sha256=review_batch,
        records=(review,),
        review_protocol_revision="r1",
    )

    with pytest.raises(ValueError, match="rejected"):
        resolve_japanese_reading(
            "学校へ行く",
            split="test",
            machine_proposal=value,
            review_ledger=rejected,
            utterance_id="utt-1",
        )


def test_provider_output_carries_frozen_identity() -> None:
    provider = CallableJapaneseReadingProvider(
        lambda text: "ガッコウヘイク" if text == "学校へ行く" else "",
        identity=identity(),
    )

    resolved = resolve_japanese_reading(
        "学校へ行く",
        split="train",
        provider=provider,
        resolution_policy=ReadingResolutionPolicy(
            allow_unreviewed_machine_train=True,
        ),
        utterance_id="utt-1",
    )

    assert resolved.provider_digest == provider.identity.digest
    assert resolved.origin == "machine-proposed"


def test_machine_proposal_cannot_be_reused_for_different_source_text() -> None:
    value = proposal()

    with pytest.raises(ValueError, match="different source text"):
        resolve_japanese_reading(
            "学校へ行った",
            split="train",
            machine_proposal=value,
            resolution_policy=ReadingResolutionPolicy(
                allow_unreviewed_machine_train=True,
            ),
            utterance_id="utt-1",
        )


def test_reading_policy_digest_is_part_of_proposal_identity() -> None:
    strict = JapanesePronunciationPolicy(ignore_punctuation=False)
    value = proposal()

    with pytest.raises(ValueError, match="different pronunciation policy"):
        resolve_japanese_reading(
            "学校へ行く",
            split="train",
            machine_proposal=value,
            pronunciation_policy=strict,
            resolution_policy=ReadingResolutionPolicy(
                allow_unreviewed_machine_train=True,
            ),
            utterance_id="utt-1",
        )
