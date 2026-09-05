"""Frozen, reference-separated protocol for phonetic proposal ablations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_bridge import FrozenPronunciationLexicon

EvidenceChannel = Literal["first_pass", "phone", "mora", "discrete_unit"]
BootstrapGroup = Literal["speaker", "session", "source"]


def _strict_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class FrozenSpanReference:
    reference_id: str
    text: str
    semantic_kind: str = "general"
    critical: bool = False
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.reference_id or not self.text or not self.semantic_kind:
            raise ValueError("span reference requires ID, text, and semantic kind")
        if not isinstance(self.critical, bool):
            raise TypeError("critical must be a boolean")

    @property
    def text_sha256(self) -> str:
        return sha256_json({"text": self.text})

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "referenceId": self.reference_id,
                "textSha256": self.text_sha256,
                "semanticKind": self.semantic_kind,
                "critical": self.critical,
            }
        )


@dataclass(frozen=True, slots=True)
class FirstPassSpanCandidate:
    candidate_id: str
    text: str
    posterior: float
    selected: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.text:
            raise ValueError("first-pass span candidate requires ID and text")
        posterior = _strict_float(self.posterior, name="first-pass posterior")
        if not 0.0 <= posterior <= 1.0:
            raise ValueError("first-pass posterior must be in [0, 1]")
        if not isinstance(self.selected, bool):
            raise TypeError("selected must be a boolean")
        object.__setattr__(self, "posterior", posterior)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "candidateId": self.candidate_id,
                "textSha256": sha256_json({"text": self.text}),
                "posterior": self.posterior,
                "selected": self.selected,
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticAblationCase:
    case_id: str
    audio_path: Path
    source_audio_sha256: str
    start_ms: int
    end_ms: int
    first_pass_candidates: tuple[FirstPassSpanCandidate, ...]
    lexicon: FrozenPronunciationLexicon
    reference: FrozenSpanReference
    speaker_id: str
    session_id: str
    source_id: str
    license_id: str
    rights_decision: str
    dataset_revision: str
    split_manifest_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.case_id or not self.audio_path.is_absolute():
            raise ValueError("ablation case requires ID and absolute audio path")
        for value in (
            self.source_audio_sha256,
            self.split_manifest_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("ablation case digests must be SHA-256 values")
        if isinstance(self.start_ms, bool) or isinstance(self.end_ms, bool):
            raise TypeError("span times must be integers")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("ablation span requires 0 <= start_ms < end_ms")
        if not self.first_pass_candidates:
            raise ValueError("ablation case requires first-pass candidates")
        if len({row.candidate_id for row in self.first_pass_candidates}) != len(
            self.first_pass_candidates
        ):
            raise ValueError("first-pass candidate IDs must be unique")
        if len({row.text for row in self.first_pass_candidates}) != len(self.first_pass_candidates):
            raise ValueError("first-pass candidate surfaces must be unique")
        if sum(row.selected for row in self.first_pass_candidates) != 1:
            raise ValueError("exactly one first-pass candidate must be selected")
        total = sum(row.posterior for row in self.first_pass_candidates)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("first-pass posteriors must sum to one")
        lexicon_surfaces = {entry.text for entry in self.lexicon.entries}
        missing = {row.text for row in self.first_pass_candidates} - lexicon_surfaces
        if missing:
            raise ValueError(
                f"frozen pronunciation lexicon misses first-pass surfaces: {sorted(missing)}"
            )
        if self.rights_decision != "allow":
            raise ValueError("ablation cases require rights_decision='allow'")
        for name in (
            "speaker_id",
            "session_id",
            "source_id",
            "license_id",
            "dataset_revision",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")

    @property
    def selected_first_pass(self) -> FirstPassSpanCandidate:
        return next(row for row in self.first_pass_candidates if row.selected)

    @property
    def planning_digest(self) -> str:
        """Reference-free identity passed to candidate planning."""

        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "caseId": self.case_id,
                "sourceAudioSha256": self.source_audio_sha256,
                "startMs": self.start_ms,
                "endMs": self.end_ms,
                "firstPassDigests": [row.digest for row in self.first_pass_candidates],
                "lexiconDigest": self.lexicon.digest,
                "speakerId": self.speaker_id,
                "sessionId": self.session_id,
                "sourceId": self.source_id,
                "licenseId": self.license_id,
                "rightsDecision": self.rights_decision,
                "datasetRevision": self.dataset_revision,
                "splitManifestSha256": self.split_manifest_sha256,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "planningDigest": self.planning_digest,
                "referenceDigest": self.reference.digest,
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticAblationManifest:
    name: str
    revision: str
    cases: tuple[PhoneticAblationCase, ...]
    runtime_profile_digest: str
    utility_artifact_digest: str
    rights_registry_sha256: str
    split_manifest_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.cases:
            raise ValueError("ablation manifest requires name, revision, and cases")
        for value in (
            self.runtime_profile_digest,
            self.utility_artifact_digest,
            self.rights_registry_sha256,
            self.split_manifest_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("ablation manifest digests must be SHA-256 values")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("ablation case IDs must be unique")
        if len({case.source_audio_sha256 for case in self.cases}) != len(self.cases):
            raise ValueError("ablation source-audio hashes must be unique")
        for case in self.cases:
            if case.split_manifest_sha256 != self.split_manifest_sha256:
                raise ValueError("case belongs to a different split manifest")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "caseDigests": [case.digest for case in self.cases],
                "runtimeProfileDigest": self.runtime_profile_digest,
                "utilityArtifactDigest": self.utility_artifact_digest,
                "rightsRegistrySha256": self.rights_registry_sha256,
                "splitManifestSha256": self.split_manifest_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class PhoneticAblationArm:
    name: str
    channel_weights: tuple[tuple[EvidenceChannel, float], ...]
    allow_outside_first_pass: bool = True
    retention_bonus: float = 0.0
    minimum_margin: float = 0.0
    apply_provisional: bool = False
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.channel_weights:
            raise ValueError("ablation arm requires name and channel weights")
        channels = [channel for channel, _ in self.channel_weights]
        if len(channels) != len(set(channels)):
            raise ValueError("ablation arm channels must be unique")
        allowed = {"first_pass", "phone", "mora", "discrete_unit"}
        if set(channels) - allowed:
            raise ValueError("ablation arm contains an unknown evidence channel")
        normalized: list[tuple[EvidenceChannel, float]] = []
        for channel, value in self.channel_weights:
            weight = _strict_float(value, name=f"weight for {channel}")
            if weight < 0.0:
                raise ValueError("ablation arm weights must be non-negative")
            normalized.append((channel, weight))
        if not any(weight > 0.0 for _, weight in normalized):
            raise ValueError("ablation arm requires a positive weight")
        retention = _strict_float(self.retention_bonus, name="retention_bonus")
        margin = _strict_float(self.minimum_margin, name="minimum_margin")
        if retention < 0.0 or margin < 0.0:
            raise ValueError("retention_bonus and minimum_margin must be non-negative")
        if not isinstance(self.allow_outside_first_pass, bool):
            raise TypeError("allow_outside_first_pass must be a boolean")
        if not isinstance(self.apply_provisional, bool):
            raise TypeError("apply_provisional must be a boolean")
        object.__setattr__(self, "channel_weights", tuple(sorted(normalized)))
        object.__setattr__(self, "retention_bonus", retention)
        object.__setattr__(self, "minimum_margin", margin)

    @property
    def weights(self) -> dict[EvidenceChannel, float]:
        return dict(self.channel_weights)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticAblationProtocol:
    name: str
    revision: str
    arms: tuple[PhoneticAblationArm, ...]
    baseline_arm: str
    maximum_candidates: int = 256
    maximum_crop_ms: int = 4_000
    bootstrap_resamples: int = 2_000
    bootstrap_seed: str = "semantic-asr-phonetic-ablation-v1"
    bootstrap_group: BootstrapGroup = "speaker"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.arms:
            raise ValueError("ablation protocol requires name, revision, and arms")
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("ablation arm names must be unique")
        if self.baseline_arm not in names:
            raise ValueError("baseline_arm is absent from the protocol")
        for name in ("maximum_candidates", "maximum_crop_ms", "bootstrap_resamples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.bootstrap_seed:
            raise ValueError("bootstrap_seed is required")
        if self.bootstrap_group not in {"speaker", "session", "source"}:
            raise ValueError("bootstrap_group must be speaker, session, or source")

    def arm(self, name: str) -> PhoneticAblationArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(name)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "armDigests": [arm.digest for arm in self.arms],
                "baselineArm": self.baseline_arm,
                "maximumCandidates": self.maximum_candidates,
                "maximumCropMs": self.maximum_crop_ms,
                "bootstrapResamples": self.bootstrap_resamples,
                "bootstrapSeed": self.bootstrap_seed,
                "bootstrapGroup": self.bootstrap_group,
            }
        )
