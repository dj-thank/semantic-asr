"""Document decision and plan receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from .overlap_types import OverlapCompatibility
from .path_types import DocumentPathHypothesis
from .window_types import WindowPathOption, WindowPathSet


@dataclass(frozen=True, slots=True)
class DocumentDeliberationDecision:
    selected: DocumentPathHypothesis
    retained: DocumentPathHypothesis
    alternatives: tuple[DocumentPathHypothesis, ...]
    status: Literal["accepted", "provisional"]
    margin: float
    reasons: tuple[str, ...]
    first_pass_evidence_sha256: str
    config_digest: str
    local_policy_digest: str
    context_digest: str
    scorer_source: str | None = None
    scorer_profile_digest: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"accepted", "provisional"}:
            raise ValueError("document decision status must be accepted or provisional")
        margin = _strict_float(self.margin, name="margin")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        object.__setattr__(self, "margin", margin)
        for digest in (
            self.first_pass_evidence_sha256,
            self.config_digest,
            self.local_policy_digest,
            self.context_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("document decision contains an invalid SHA-256 value")
        if (self.scorer_source is None) != (self.scorer_profile_digest is None):
            raise ValueError("document scorer source and profile must be supplied together")
        if self.scorer_profile_digest is not None and not _is_sha256(self.scorer_profile_digest):
            raise ValueError("document scorer profile must be a SHA-256 value")
        if self.selected.digest not in {row.digest for row in self.alternatives}:
            raise ValueError("selected document path must be in alternatives")
        if self.retained.digest not in {row.digest for row in self.alternatives}:
            raise ValueError("retained document path must be in alternatives")

    @property
    def changed(self) -> bool:
        return self.selected.digest != self.retained.digest

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "selectedDigest": self.selected.digest,
                "selectedScoreDigest": self.selected.score_digest,
                "retainedDigest": self.retained.digest,
                "retainedScoreDigest": self.retained.score_digest,
                "alternativeScoreDigests": [row.score_digest for row in self.alternatives],
                "status": self.status,
                "margin": self.margin,
                "reasons": self.reasons,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "configDigest": self.config_digest,
                "localPolicyDigest": self.local_policy_digest,
                "contextDigest": self.context_digest,
                "scorerSource": self.scorer_source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentDeliberationPlan:
    first_pass_evidence_sha256: str
    window_sets: tuple[WindowPathSet, ...]
    decision: DocumentDeliberationDecision
    config_digest: str
    local_policy_digest: str
    context_plan_digest: str

    def __post_init__(self) -> None:
        for digest in (
            self.first_pass_evidence_sha256,
            self.config_digest,
            self.local_policy_digest,
            self.context_plan_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("document plan contains an invalid SHA-256 value")
        if tuple(row.window_index for row in self.window_sets) != tuple(
            range(len(self.window_sets))
        ):
            raise ValueError("document window sets must be contiguous")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "windowSetDigests": [row.digest for row in self.window_sets],
                "decisionDigest": self.decision.digest,
                "configDigest": self.config_digest,
                "localPolicyDigest": self.local_policy_digest,
                "contextPlanDigest": self.context_plan_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class _BeamState:
    options: tuple[WindowPathOption, ...]
    overlap_receipts: tuple[OverlapCompatibility, ...]
    base_score: float
    changed_windows: int
    generated_windows: int
