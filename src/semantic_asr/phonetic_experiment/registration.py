"""Pre-registration binding for phonetic proposal ablations and promotion thresholds."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from .planner import FrozenPhoneticCandidatePlanner
from .promotion import (
    PhoneticPromotionDecision,
    PhoneticPromotionPolicy,
    evaluate_phonetic_promotion,
)
from .protocol import PhoneticAblationManifest, PhoneticAblationProtocol
from .runner import PhoneticAblationReport, run_phonetic_ablation


@dataclass(frozen=True, slots=True)
class PhoneticExperimentRegistration:
    name: str
    revision: str
    manifest_digest: str
    protocol_digest: str
    planner_profile_digest: str
    promotion_policy_digest: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("phonetic experiment registration requires name and revision")
        for value in (
            self.manifest_digest,
            self.protocol_digest,
            self.planner_profile_digest,
            self.promotion_policy_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("registration digests must be SHA-256 values")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        revision: str,
        manifest: PhoneticAblationManifest,
        protocol: PhoneticAblationProtocol,
        planner: FrozenPhoneticCandidatePlanner,
        promotion_policy: PhoneticPromotionPolicy,
    ) -> PhoneticExperimentRegistration:
        return cls(
            name=name,
            revision=revision,
            manifest_digest=manifest.digest,
            protocol_digest=protocol.digest,
            planner_profile_digest=planner.profile_digest,
            promotion_policy_digest=promotion_policy.digest,
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "manifestDigest": self.manifest_digest,
                "protocolDigest": self.protocol_digest,
                "plannerProfileDigest": self.planner_profile_digest,
                "promotionPolicyDigest": self.promotion_policy_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class RegisteredPhoneticExperimentResult:
    registration_digest: str
    report: PhoneticAblationReport
    promotion: PhoneticPromotionDecision

    def __post_init__(self) -> None:
        if not _is_sha256(self.registration_digest):
            raise ValueError("registration_digest must be a SHA-256 value")
        if self.promotion.report_digest != self.report.digest:
            raise ValueError("promotion decision belongs to a different report")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "registrationDigest": self.registration_digest,
                "reportDigest": self.report.digest,
                "promotionDigest": self.promotion.digest,
            }
        )


def run_registered_phonetic_experiment(
    registration: PhoneticExperimentRegistration,
    *,
    manifest: PhoneticAblationManifest,
    protocol: PhoneticAblationProtocol,
    planner: FrozenPhoneticCandidatePlanner,
    promotion_policy: PhoneticPromotionPolicy,
) -> RegisteredPhoneticExperimentResult:
    observed = {
        "manifest_digest": manifest.digest,
        "protocol_digest": protocol.digest,
        "planner_profile_digest": planner.profile_digest,
        "promotion_policy_digest": promotion_policy.digest,
    }
    expected = {
        "manifest_digest": registration.manifest_digest,
        "protocol_digest": registration.protocol_digest,
        "planner_profile_digest": registration.planner_profile_digest,
        "promotion_policy_digest": registration.promotion_policy_digest,
    }
    mismatches = [name for name in expected if expected[name] != observed[name]]
    if mismatches:
        raise ValueError(
            "registered phonetic experiment identity mismatch: " + ", ".join(mismatches)
        )
    report = run_phonetic_ablation(manifest, protocol, planner)
    promotion = evaluate_phonetic_promotion(report, promotion_policy)
    return RegisteredPhoneticExperimentResult(
        registration_digest=registration.digest,
        report=report,
        promotion=promotion,
    )
