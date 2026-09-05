"""Pre-registration for context × phonetic factorial experiments."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_experiment.planner import FrozenPhoneticCandidatePlanner
from .context_scorer import CandidateContextScorer
from .promotion import (
    ContextPhoneticPromotionDecision,
    ContextPhoneticPromotionPolicy,
    evaluate_context_phonetic_promotion,
)
from .protocol import ContextPhoneticManifest, ContextPhoneticProtocol
from .runner import (
    ContextPhoneticFactorialReport,
    run_context_phonetic_experiment,
)


@dataclass(frozen=True, slots=True)
class ContextPhoneticExperimentRegistration:
    name: str
    revision: str
    manifest_digest: str
    manifest_planning_digest: str
    protocol_digest: str
    phonetic_planner_profile_digest: str
    context_scorer_source: str
    context_scorer_profile_digest: str
    promotion_policy_digest: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.context_scorer_source:
            raise ValueError("factorial registration requires name, revision, and scorer source")
        for value in (
            self.manifest_digest,
            self.manifest_planning_digest,
            self.protocol_digest,
            self.phonetic_planner_profile_digest,
            self.context_scorer_profile_digest,
            self.promotion_policy_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("factorial registration digests must be SHA-256 values")

    @classmethod
    def create(
        cls,
        *,
        name: str,
        revision: str,
        manifest: ContextPhoneticManifest,
        protocol: ContextPhoneticProtocol,
        phonetic_planner: FrozenPhoneticCandidatePlanner,
        context_scorer: CandidateContextScorer,
        promotion_policy: ContextPhoneticPromotionPolicy,
    ) -> ContextPhoneticExperimentRegistration:
        return cls(
            name=name,
            revision=revision,
            manifest_digest=manifest.digest,
            manifest_planning_digest=manifest.planning_digest,
            protocol_digest=protocol.digest,
            phonetic_planner_profile_digest=phonetic_planner.profile_digest,
            context_scorer_source=context_scorer.source,
            context_scorer_profile_digest=context_scorer.profile_digest,
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
                "manifestPlanningDigest": self.manifest_planning_digest,
                "protocolDigest": self.protocol_digest,
                "phoneticPlannerProfileDigest": self.phonetic_planner_profile_digest,
                "contextScorerSource": self.context_scorer_source,
                "contextScorerProfileDigest": self.context_scorer_profile_digest,
                "promotionPolicyDigest": self.promotion_policy_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class RegisteredContextPhoneticResult:
    registration_digest: str
    report: ContextPhoneticFactorialReport
    promotion: ContextPhoneticPromotionDecision

    def __post_init__(self) -> None:
        if not _is_sha256(self.registration_digest):
            raise ValueError("registration_digest must be a SHA-256 value")
        if self.promotion.report_digest != self.report.digest:
            raise ValueError("promotion decision belongs to a different factorial report")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "registrationDigest": self.registration_digest,
                "reportDigest": self.report.digest,
                "promotionDigest": self.promotion.digest,
            }
        )


def run_registered_context_phonetic_experiment(
    registration: ContextPhoneticExperimentRegistration,
    *,
    manifest: ContextPhoneticManifest,
    protocol: ContextPhoneticProtocol,
    phonetic_planner: FrozenPhoneticCandidatePlanner,
    context_scorer: CandidateContextScorer,
    promotion_policy: ContextPhoneticPromotionPolicy,
) -> RegisteredContextPhoneticResult:
    observed = {
        "manifest_digest": manifest.digest,
        "manifest_planning_digest": manifest.planning_digest,
        "protocol_digest": protocol.digest,
        "phonetic_planner_profile_digest": phonetic_planner.profile_digest,
        "context_scorer_source": context_scorer.source,
        "context_scorer_profile_digest": context_scorer.profile_digest,
        "promotion_policy_digest": promotion_policy.digest,
    }
    expected = {
        "manifest_digest": registration.manifest_digest,
        "manifest_planning_digest": registration.manifest_planning_digest,
        "protocol_digest": registration.protocol_digest,
        "phonetic_planner_profile_digest": (
            registration.phonetic_planner_profile_digest
        ),
        "context_scorer_source": registration.context_scorer_source,
        "context_scorer_profile_digest": registration.context_scorer_profile_digest,
        "promotion_policy_digest": registration.promotion_policy_digest,
    }
    mismatches = tuple(name for name in expected if expected[name] != observed[name])
    if mismatches:
        raise ValueError(
            "registered context-phonetic experiment identity mismatch: "
            + ", ".join(mismatches)
        )
    report = run_context_phonetic_experiment(
        manifest,
        protocol,
        phonetic_planner,
        context_scorer,
    )
    promotion = evaluate_context_phonetic_promotion(report, promotion_policy)
    return RegisteredContextPhoneticResult(
        registration_digest=registration.digest,
        report=report,
        promotion=promotion,
    )
