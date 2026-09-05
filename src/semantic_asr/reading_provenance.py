"""Provenance-safe Japanese reading proposals, reviews, and split policies.

This module does not infer pronunciation from text by itself. A reading is either supplied
explicitly by a human or proposed by a separately frozen provider whose model/config/resource
identity is digest-bound. Locked evaluation can require an exact human review of that proposal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

from .contracts import sha256_json
from .japanese_phonetic_targets import (
    JapanesePronunciationPolicy,
    japanese_pronunciation_target,
)

ReadingOrigin = Literal["human-explicit", "machine-proposed", "machine-reviewed"]
ReviewDisposition = Literal["approved", "corrected", "rejected"]
SplitName = Literal["train", "calibration", "test"]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _required_sha256(value: str, *, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a SHA-256 value")
    return value


def _normalized_reading(
    reading: str,
    policy: JapanesePronunciationPolicy,
) -> str:
    return japanese_pronunciation_target(reading, policy=policy).normalized_reading


@dataclass(frozen=True, slots=True)
class ReadingProviderIdentity:
    provider_id: str
    provider_revision: str
    provider_config_digest: str
    provider_artifact_sha256: str
    resource_artifact_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reading provider identity schema_version must be '1'")
        if not self.provider_id or not self.provider_revision:
            raise ValueError("reading provider ID and revision are required")
        for name in (
            "provider_config_digest",
            "provider_artifact_sha256",
            "resource_artifact_sha256",
        ):
            _required_sha256(getattr(self, name), name=name)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class JapaneseReadingProposal:
    source_text_sha256: str
    normalized_reading: str
    reading_sha256: str
    origin: Literal["human-explicit", "machine-proposed"]
    pronunciation_policy_digest: str
    provider: ReadingProviderIdentity | None = None
    utterance_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reading proposal schema_version must be '1'")
        _required_sha256(self.source_text_sha256, name="source_text_sha256")
        _required_sha256(self.reading_sha256, name="reading_sha256")
        _required_sha256(
            self.pronunciation_policy_digest,
            name="pronunciation_policy_digest",
        )
        if not self.normalized_reading:
            raise ValueError("normalized_reading is required")
        if self.reading_sha256 != _text_sha256(self.normalized_reading):
            raise ValueError("reading_sha256 does not match normalized_reading")
        if self.origin == "human-explicit":
            if self.provider is not None:
                raise ValueError("human-explicit readings must not claim a machine provider")
        elif self.origin == "machine-proposed":
            if self.provider is None:
                raise ValueError("machine-proposed readings require a provider identity")
        else:
            raise ValueError("unknown reading proposal origin")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def human(
        cls,
        source_text: str,
        reading: str,
        *,
        pronunciation_policy: JapanesePronunciationPolicy | None = None,
        utterance_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> JapaneseReadingProposal:
        policy = pronunciation_policy or JapanesePronunciationPolicy()
        normalized = _normalized_reading(reading, policy)
        return cls(
            source_text_sha256=_text_sha256(source_text),
            normalized_reading=normalized,
            reading_sha256=_text_sha256(normalized),
            origin="human-explicit",
            pronunciation_policy_digest=policy.digest,
            utterance_id=utterance_id,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def machine(
        cls,
        source_text: str,
        reading: str,
        *,
        provider: ReadingProviderIdentity,
        pronunciation_policy: JapanesePronunciationPolicy | None = None,
        utterance_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> JapaneseReadingProposal:
        policy = pronunciation_policy or JapanesePronunciationPolicy()
        normalized = _normalized_reading(reading, policy)
        return cls(
            source_text_sha256=_text_sha256(source_text),
            normalized_reading=normalized,
            reading_sha256=_text_sha256(normalized),
            origin="machine-proposed",
            pronunciation_policy_digest=policy.digest,
            provider=provider,
            utterance_id=utterance_id,
            metadata=dict(metadata or {}),
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "sourceTextSha256": self.source_text_sha256,
                "normalizedReading": self.normalized_reading,
                "readingSha256": self.reading_sha256,
                "origin": self.origin,
                "pronunciationPolicyDigest": self.pronunciation_policy_digest,
                "providerDigest": None if self.provider is None else self.provider.digest,
                "utteranceId": self.utterance_id,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class JapaneseReadingReview:
    proposal_digest: str
    source_text_sha256: str
    proposed_reading_sha256: str
    disposition: ReviewDisposition
    approved_reading: str | None
    approved_reading_sha256: str | None
    reviewer_id_hash: str
    review_protocol_revision: str
    review_manifest_sha256: str
    pronunciation_policy_digest: str
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reading review schema_version must be '1'")
        for name in (
            "proposal_digest",
            "source_text_sha256",
            "proposed_reading_sha256",
            "reviewer_id_hash",
            "review_manifest_sha256",
            "pronunciation_policy_digest",
        ):
            _required_sha256(getattr(self, name), name=name)
        if self.disposition not in {"approved", "corrected", "rejected"}:
            raise ValueError("unknown reading review disposition")
        if not self.review_protocol_revision:
            raise ValueError("review_protocol_revision is required")
        if self.disposition == "rejected":
            if self.approved_reading is not None or self.approved_reading_sha256 is not None:
                raise ValueError("rejected reviews must not contain an approved reading")
        else:
            if not self.approved_reading or not self.approved_reading_sha256:
                raise ValueError("approved/corrected reviews require an approved reading")
            if self.approved_reading_sha256 != _text_sha256(self.approved_reading):
                raise ValueError("approved_reading_sha256 does not match approved_reading")
            if (
                self.disposition == "approved"
                and self.approved_reading_sha256 != self.proposed_reading_sha256
            ):
                raise ValueError("approved review must retain the proposed reading")
            if (
                self.disposition == "corrected"
                and self.approved_reading_sha256 == self.proposed_reading_sha256
            ):
                raise ValueError("corrected review must change the proposed reading")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def create(
        cls,
        proposal: JapaneseReadingProposal,
        *,
        disposition: ReviewDisposition,
        approved_reading: str | None,
        reviewer_id_hash: str,
        review_protocol_revision: str,
        review_manifest_sha256: str,
        pronunciation_policy: JapanesePronunciationPolicy | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> JapaneseReadingReview:
        if proposal.origin != "machine-proposed":
            raise ValueError("only machine proposals require a reading review")
        policy = pronunciation_policy or JapanesePronunciationPolicy()
        if policy.digest != proposal.pronunciation_policy_digest:
            raise ValueError("review policy differs from proposal pronunciation policy")
        normalized = (
            None
            if approved_reading is None
            else _normalized_reading(approved_reading, policy)
        )
        return cls(
            proposal_digest=proposal.digest,
            source_text_sha256=proposal.source_text_sha256,
            proposed_reading_sha256=proposal.reading_sha256,
            disposition=disposition,
            approved_reading=normalized,
            approved_reading_sha256=(None if normalized is None else _text_sha256(normalized)),
            reviewer_id_hash=reviewer_id_hash,
            review_protocol_revision=review_protocol_revision,
            review_manifest_sha256=review_manifest_sha256,
            pronunciation_policy_digest=policy.digest,
            metadata=dict(metadata or {}),
        )

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def verify_proposal(self, proposal: JapaneseReadingProposal) -> None:
        if proposal.digest != self.proposal_digest:
            raise ValueError("review is bound to a different proposal")
        if proposal.source_text_sha256 != self.source_text_sha256:
            raise ValueError("review is bound to different source text")
        if proposal.reading_sha256 != self.proposed_reading_sha256:
            raise ValueError("review is bound to a different proposed reading")
        if proposal.pronunciation_policy_digest != self.pronunciation_policy_digest:
            raise ValueError("review is bound to a different pronunciation policy")


@dataclass(frozen=True, slots=True)
class JapaneseReadingReviewLedger:
    revision: str
    source_manifest_sha256: str
    records: tuple[JapaneseReadingReview, ...]
    review_protocol_revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reading review ledger schema_version must be '1'")
        if not self.revision or not self.review_protocol_revision:
            raise ValueError("reading review ledger revisions are required")
        _required_sha256(self.source_manifest_sha256, name="source_manifest_sha256")
        if len({record.proposal_digest for record in self.records}) != len(self.records):
            raise ValueError("review ledger proposal digests must be unique")
        if any(
            record.review_protocol_revision != self.review_protocol_revision
            for record in self.records
        ):
            raise ValueError("review ledger mixes review protocol revisions")
        if any(
            record.review_manifest_sha256 != self.source_manifest_sha256
            for record in self.records
        ):
            raise ValueError("review records are bound to a different review manifest")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "revision": self.revision,
                "sourceManifestSha256": self.source_manifest_sha256,
                "reviewProtocolRevision": self.review_protocol_revision,
                "recordDigests": [record.digest for record in self.records],
            }
        )

    def review_for(self, proposal: JapaneseReadingProposal) -> JapaneseReadingReview | None:
        for record in self.records:
            if record.proposal_digest == proposal.digest:
                record.verify_proposal(proposal)
                return record
        return None


@dataclass(frozen=True, slots=True)
class ReadingResolutionPolicy:
    allow_human_explicit: bool = True
    allow_unreviewed_machine_train: bool = False
    require_review_for_calibration: bool = True
    require_review_for_test: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("reading resolution policy schema_version must be '1'")
        for name in (
            "allow_human_explicit",
            "allow_unreviewed_machine_train",
            "require_review_for_calibration",
            "require_review_for_test",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ResolvedJapaneseReading:
    source_text_sha256: str
    normalized_reading: str
    reading_sha256: str
    origin: ReadingOrigin
    proposal_digest: str
    review_digest: str | None
    provider_digest: str | None
    pronunciation_policy_digest: str
    resolution_policy_digest: str
    eligible_for_locked_test: bool
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("resolved reading schema_version must be '1'")
        for name in (
            "source_text_sha256",
            "reading_sha256",
            "proposal_digest",
            "pronunciation_policy_digest",
            "resolution_policy_digest",
        ):
            _required_sha256(getattr(self, name), name=name)
        for name in ("review_digest", "provider_digest"):
            value = getattr(self, name)
            if value is not None:
                _required_sha256(value, name=name)
        if self.reading_sha256 != _text_sha256(self.normalized_reading):
            raise ValueError("resolved reading hash mismatch")
        if self.origin == "human-explicit":
            if self.review_digest is not None or self.provider_digest is not None:
                raise ValueError("human-explicit resolution must not claim machine provenance")
        elif self.origin == "machine-proposed":
            if self.provider_digest is None or self.review_digest is not None:
                raise ValueError("machine-proposed resolution provenance is invalid")
            if self.eligible_for_locked_test:
                raise ValueError("unreviewed machine reading cannot be locked-test eligible")
        elif self.origin == "machine-reviewed":
            if self.provider_digest is None or self.review_digest is None:
                raise ValueError("machine-reviewed resolution requires provider and review")
            if not self.eligible_for_locked_test:
                raise ValueError("reviewed machine reading must be locked-test eligible")
        else:
            raise ValueError("unknown resolved reading origin")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


class JapaneseReadingProvider(Protocol):
    identity: ReadingProviderIdentity

    def propose(self, source_text: str, *, utterance_id: str | None = None) -> str: ...


class CallableJapaneseReadingProvider:
    """Adapter for a separately frozen G2P implementation.

    The callable is deliberately not serialized. Its exact implementation, model, configuration,
    and dictionary/resource artifacts must already be represented by ``identity``.
    """

    def __init__(
        self,
        function: Callable[[str], str],
        *,
        identity: ReadingProviderIdentity,
    ) -> None:
        self.function = function
        self.identity = identity

    def propose(self, source_text: str, *, utterance_id: str | None = None) -> str:
        del utterance_id
        value = self.function(source_text)
        if not isinstance(value, str) or not value:
            raise ValueError("reading provider must return a non-empty string")
        return value


def resolve_japanese_reading(
    source_text: str,
    *,
    split: SplitName,
    explicit_reading: str | None = None,
    provider: JapaneseReadingProvider | None = None,
    machine_proposal: JapaneseReadingProposal | None = None,
    review_ledger: JapaneseReadingReviewLedger | None = None,
    pronunciation_policy: JapanesePronunciationPolicy | None = None,
    resolution_policy: ReadingResolutionPolicy | None = None,
    utterance_id: str | None = None,
) -> ResolvedJapaneseReading:
    if split not in {"train", "calibration", "test"}:
        raise ValueError("split must be train, calibration, or test")
    if not source_text:
        raise ValueError("source_text is required")
    pronunciation_policy = pronunciation_policy or JapanesePronunciationPolicy()
    resolution_policy = resolution_policy or ReadingResolutionPolicy()
    if explicit_reading is not None:
        if provider is not None or machine_proposal is not None:
            raise ValueError("explicit and machine readings are mutually exclusive")
        if not resolution_policy.allow_human_explicit:
            raise ValueError("human-explicit readings are disabled by policy")
        proposal = JapaneseReadingProposal.human(
            source_text,
            explicit_reading,
            pronunciation_policy=pronunciation_policy,
            utterance_id=utterance_id,
        )
        return ResolvedJapaneseReading(
            source_text_sha256=proposal.source_text_sha256,
            normalized_reading=proposal.normalized_reading,
            reading_sha256=proposal.reading_sha256,
            origin="human-explicit",
            proposal_digest=proposal.digest,
            review_digest=None,
            provider_digest=None,
            pronunciation_policy_digest=pronunciation_policy.digest,
            resolution_policy_digest=resolution_policy.digest,
            eligible_for_locked_test=True,
        )

    if machine_proposal is not None and provider is not None:
        raise ValueError("supply provider or precomputed machine_proposal, not both")
    if machine_proposal is None:
        if provider is None:
            raise ValueError("a reading, provider, or machine proposal is required")
        machine_proposal = JapaneseReadingProposal.machine(
            source_text,
            provider.propose(source_text, utterance_id=utterance_id),
            provider=provider.identity,
            pronunciation_policy=pronunciation_policy,
            utterance_id=utterance_id,
        )
    if machine_proposal.origin != "machine-proposed" or machine_proposal.provider is None:
        raise ValueError("machine_proposal must contain frozen machine provenance")
    if machine_proposal.source_text_sha256 != _text_sha256(source_text):
        raise ValueError("machine proposal is bound to different source text")
    if machine_proposal.pronunciation_policy_digest != pronunciation_policy.digest:
        raise ValueError("machine proposal uses a different pronunciation policy")
    if utterance_id is not None and machine_proposal.utterance_id not in {None, utterance_id}:
        raise ValueError("machine proposal is bound to a different utterance")

    review = None if review_ledger is None else review_ledger.review_for(machine_proposal)
    if review is not None:
        if review.disposition == "rejected":
            raise ValueError("machine reading was rejected by the review ledger")
        assert review.approved_reading is not None
        return ResolvedJapaneseReading(
            source_text_sha256=machine_proposal.source_text_sha256,
            normalized_reading=review.approved_reading,
            reading_sha256=review.approved_reading_sha256 or "",
            origin="machine-reviewed",
            proposal_digest=machine_proposal.digest,
            review_digest=review.digest,
            provider_digest=machine_proposal.provider.digest,
            pronunciation_policy_digest=pronunciation_policy.digest,
            resolution_policy_digest=resolution_policy.digest,
            eligible_for_locked_test=True,
            metadata={"reviewDisposition": review.disposition},
        )

    review_required = (
        split == "test" and resolution_policy.require_review_for_test
    ) or (
        split == "calibration" and resolution_policy.require_review_for_calibration
    )
    if review_required:
        raise ValueError(f"{split} split requires a reviewed machine reading")
    if split != "train" or not resolution_policy.allow_unreviewed_machine_train:
        raise ValueError("unreviewed machine reading is disabled by policy")
    return ResolvedJapaneseReading(
        source_text_sha256=machine_proposal.source_text_sha256,
        normalized_reading=machine_proposal.normalized_reading,
        reading_sha256=machine_proposal.reading_sha256,
        origin="machine-proposed",
        proposal_digest=machine_proposal.digest,
        review_digest=None,
        provider_digest=machine_proposal.provider.digest,
        pronunciation_policy_digest=pronunciation_policy.digest,
        resolution_policy_digest=resolution_policy.digest,
        eligible_for_locked_test=False,
    )
