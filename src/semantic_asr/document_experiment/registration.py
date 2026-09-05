"""Immutable scorer registry and preregistration binding for context experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .protocol import DocumentExperimentManifest, DocumentExperimentProtocol
from .runner import (
    DocumentArmScorer,
    DocumentContextExperimentReport,
    PreparedDocumentCase,
    run_document_context_experiment,
)


@dataclass(frozen=True, slots=True)
class FrozenScorerRegistry:
    profiles: tuple[tuple[str, str], ...]
    revision: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.revision:
            raise ValueError("scorer registry revision is required")
        keys = [key for key, _ in self.profiles]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("scorer registry keys must be non-empty and unique")
        if keys != sorted(keys):
            raise ValueError("scorer registry profiles must be sorted by key")
        for key, digest in self.profiles:
            if not key or not _is_sha256(digest):
                raise ValueError("scorer registry entries require key and SHA-256 profile")

    @classmethod
    def from_scorers(
        cls,
        scorers: Mapping[str, DocumentArmScorer],
        *,
        revision: str,
    ) -> FrozenScorerRegistry:
        return cls(
            profiles=tuple(
                sorted((key, scorer.profile_digest) for key, scorer in scorers.items())
            ),
            revision=revision,
        )

    @property
    def profile_map(self) -> dict[str, str]:
        return dict(self.profiles)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))

    def validate(
        self,
        protocol: DocumentExperimentProtocol,
        scorers: Mapping[str, DocumentArmScorer],
    ) -> None:
        registered = self.profile_map
        required = {arm.scorer_key for arm in protocol.arms if arm.scorer_key is not None}
        if required != set(registered):
            raise ValueError("registered scorer keys do not exactly match protocol arms")
        if set(scorers) != set(registered):
            raise ValueError("runtime scorer keys do not exactly match frozen registry")
        for key, scorer in scorers.items():
            if scorer.profile_digest != registered[key]:
                raise ValueError(f"runtime scorer profile changed for key {key!r}")


@dataclass(frozen=True, slots=True)
class DocumentExperimentRegistration:
    protocol_digest: str
    manifest_digest: str
    scorer_registry_digest: str
    registration_id: str
    frozen_before_evaluation: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for digest in (
            self.protocol_digest,
            self.manifest_digest,
            self.scorer_registry_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("registration digests must be SHA-256 values")
        if not self.registration_id:
            raise ValueError("registration_id is required")
        if not isinstance(self.frozen_before_evaluation, bool):
            raise TypeError("frozen_before_evaluation must be a boolean")
        if not self.frozen_before_evaluation:
            raise ValueError("experiment registration must be frozen before evaluation")

    @classmethod
    def create(
        cls,
        protocol: DocumentExperimentProtocol,
        manifest: DocumentExperimentManifest,
        registry: FrozenScorerRegistry,
        *,
        registration_id: str,
    ) -> DocumentExperimentRegistration:
        return cls(
            protocol_digest=protocol.digest,
            manifest_digest=manifest.digest,
            scorer_registry_digest=registry.digest,
            registration_id=registration_id,
        )

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class RegisteredDocumentContextReport:
    registration: DocumentExperimentRegistration
    scorer_registry: FrozenScorerRegistry
    report: DocumentContextExperimentReport

    def __post_init__(self) -> None:
        if self.registration.protocol_digest != self.report.protocol_digest:
            raise ValueError("report protocol does not match preregistration")
        if self.registration.manifest_digest != self.report.manifest_digest:
            raise ValueError("report manifest does not match preregistration")
        if self.registration.scorer_registry_digest != self.scorer_registry.digest:
            raise ValueError("report scorer registry does not match preregistration")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "registrationDigest": self.registration.digest,
                "scorerRegistryDigest": self.scorer_registry.digest,
                "reportDigest": self.report.digest,
            }
        )

    def as_dict(self, *, include_text: bool = False) -> dict[str, object]:
        return {
            "registration": asdict(self.registration),
            "registrationDigest": self.registration.digest,
            "scorerRegistry": asdict(self.scorer_registry),
            "scorerRegistryDigest": self.scorer_registry.digest,
            "report": self.report.as_dict(include_text=include_text),
            "registeredReportDigest": self.digest,
        }


def run_registered_document_context_experiment(
    prepared_cases: tuple[PreparedDocumentCase, ...],
    manifest: DocumentExperimentManifest,
    protocol: DocumentExperimentProtocol,
    *,
    registry: FrozenScorerRegistry,
    registration: DocumentExperimentRegistration,
    scorers: Mapping[str, DocumentArmScorer],
) -> RegisteredDocumentContextReport:
    registry.validate(protocol, scorers)
    if registration.protocol_digest != protocol.digest:
        raise ValueError("runtime protocol differs from preregistration")
    if registration.manifest_digest != manifest.digest:
        raise ValueError("runtime manifest differs from preregistration")
    if registration.scorer_registry_digest != registry.digest:
        raise ValueError("runtime scorer registry differs from preregistration")
    report = run_document_context_experiment(
        prepared_cases,
        manifest,
        protocol,
        scorers=scorers,
    )
    return RegisteredDocumentContextReport(
        registration=registration,
        scorer_registry=registry,
        report=report,
    )
