"""Frozen protocol contracts for context × phonetic factorial experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_experiment.protocol import (
    PhoneticAblationArm,
    PhoneticAblationCase,
    PhoneticAblationProtocol,
)

ContextCondition = Literal["none", "ordered", "shuffled"]
BootstrapGroup = Literal["speaker", "session", "source"]


def _strict_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return numeric


@dataclass(frozen=True, slots=True)
class FrozenContextSnapshot:
    """Reference-free context supplied before any evaluation reference is opened."""

    context_id: str
    left_context: str = ""
    right_context: str = ""
    topic_summary: str = ""
    entity_ids: tuple[str, ...] = ()
    source_case_id: str = ""
    revision: str = "context-snapshot-v1"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.context_id or not self.source_case_id or not self.revision:
            raise ValueError("context snapshot requires context_id, source_case_id, and revision")
        if not any((self.left_context, self.right_context, self.topic_summary, self.entity_ids)):
            raise ValueError("context snapshot must contain declared context evidence")
        if len(self.entity_ids) != len(set(self.entity_ids)) or any(
            not value for value in self.entity_ids
        ):
            raise ValueError("context entity IDs must be unique and non-empty")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ContextPhoneticCase:
    phonetic_case: PhoneticAblationCase
    ordered_context: FrozenContextSnapshot
    context_group_id: str

    def __post_init__(self) -> None:
        if self.ordered_context.source_case_id != self.phonetic_case.case_id:
            raise ValueError("ordered context must originate from its own case")
        if not self.context_group_id:
            raise ValueError("context_group_id is required")

    @property
    def case_id(self) -> str:
        return self.phonetic_case.case_id

    @property
    def planning_digest(self) -> str:
        return sha256_json(
            {
                "phoneticPlanningDigest": self.phonetic_case.planning_digest,
                "orderedContextDigest": self.ordered_context.digest,
                "contextGroupId": self.context_group_id,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "planningDigest": self.planning_digest,
                "phoneticCaseDigest": self.phonetic_case.digest,
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPhoneticManifest:
    name: str
    revision: str
    cases: tuple[ContextPhoneticCase, ...]
    phonetic_manifest_digest: str
    context_source_digest: str
    rights_registry_sha256: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.cases:
            raise ValueError("factorial manifest requires name, revision, and cases")
        for value in (
            self.phonetic_manifest_digest,
            self.context_source_digest,
            self.rights_registry_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("factorial manifest digests must be SHA-256 values")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("factorial case IDs must be unique")
        split_digests = {case.phonetic_case.split_manifest_sha256 for case in self.cases}
        if len(split_digests) != 1:
            raise ValueError("factorial cases must share one frozen split manifest")

    @property
    def planning_digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "casePlanningDigests": [case.planning_digest for case in self.cases],
                "phoneticManifestDigest": self.phonetic_manifest_digest,
                "contextSourceDigest": self.context_source_digest,
                "rightsRegistrySha256": self.rights_registry_sha256,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "planningDigest": self.planning_digest,
                "caseDigests": [case.digest for case in self.cases],
            }
        )


@dataclass(frozen=True, slots=True)
class ContextPhoneticArm:
    name: str
    phonetic_arm_name: str
    context_condition: ContextCondition
    context_weight: float = 0.0
    minimum_margin: float | None = None
    apply_provisional: bool | None = None
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.phonetic_arm_name:
            raise ValueError("factorial arm requires name and phonetic_arm_name")
        if self.context_condition not in {"none", "ordered", "shuffled"}:
            raise ValueError("context condition must be none, ordered, or shuffled")
        weight = _strict_float(self.context_weight, name="context_weight", minimum=0.0)
        if self.context_condition == "none" and weight != 0.0:
            raise ValueError("none-context arm must have zero context_weight")
        if self.context_condition != "none" and weight <= 0.0:
            raise ValueError("context arms require positive context_weight")
        if self.minimum_margin is not None:
            margin = _strict_float(self.minimum_margin, name="minimum_margin", minimum=0.0)
            object.__setattr__(self, "minimum_margin", margin)
        if self.apply_provisional is not None and not isinstance(self.apply_provisional, bool):
            raise TypeError("apply_provisional must be a boolean when supplied")
        object.__setattr__(self, "context_weight", weight)

    def resolve_phonetic_arm(self, protocol: PhoneticAblationProtocol) -> PhoneticAblationArm:
        return protocol.arm(self.phonetic_arm_name)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class ContextPhoneticProtocol:
    name: str
    revision: str
    phonetic_protocol: PhoneticAblationProtocol
    arms: tuple[ContextPhoneticArm, ...]
    baseline_arm: str
    target_arm: str
    shuffled_control_arm: str
    shuffle_seed: str
    bootstrap_group: BootstrapGroup = "speaker"
    bootstrap_resamples: int = 2_000
    require_different_speaker_for_shuffle: bool = True
    require_different_session_for_shuffle: bool = True
    require_different_source_for_shuffle: bool = True
    require_different_context_group_for_shuffle: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.revision or not self.arms or not self.shuffle_seed:
            raise ValueError("factorial protocol requires name, revision, arms, and shuffle seed")
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("factorial arm names must be unique")
        for required in (self.baseline_arm, self.target_arm, self.shuffled_control_arm):
            if required not in names:
                raise ValueError(f"required factorial arm is absent: {required}")
        phonetic_names = {arm.name for arm in self.phonetic_protocol.arms}
        missing = {arm.phonetic_arm_name for arm in self.arms} - phonetic_names
        if missing:
            raise ValueError(f"factorial arms reference unknown phonetic arms: {sorted(missing)}")
        if self.bootstrap_group not in {"speaker", "session", "source"}:
            raise ValueError("bootstrap_group must be speaker, session, or source")
        if (
            isinstance(self.bootstrap_resamples, bool)
            or not isinstance(self.bootstrap_resamples, int)
            or self.bootstrap_resamples < 1
        ):
            raise ValueError("bootstrap_resamples must be a positive integer")
        for name in (
            "require_different_speaker_for_shuffle",
            "require_different_session_for_shuffle",
            "require_different_source_for_shuffle",
            "require_different_context_group_for_shuffle",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        baseline = self.arm(self.baseline_arm)
        target = self.arm(self.target_arm)
        shuffled = self.arm(self.shuffled_control_arm)
        if baseline.context_condition != "none":
            raise ValueError("baseline arm must use no context")
        if target.context_condition != "ordered":
            raise ValueError("target arm must use ordered context")
        if shuffled.context_condition != "shuffled":
            raise ValueError("shuffled_control_arm must use shuffled context")
        if target.phonetic_arm_name != shuffled.phonetic_arm_name:
            raise ValueError("ordered target and shuffled control must use the same phonetic arm")

    def arm(self, name: str) -> ContextPhoneticArm:
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
                "phoneticProtocolDigest": self.phonetic_protocol.digest,
                "armDigests": [arm.digest for arm in self.arms],
                "baselineArm": self.baseline_arm,
                "targetArm": self.target_arm,
                "shuffledControlArm": self.shuffled_control_arm,
                "shuffleSeed": self.shuffle_seed,
                "bootstrapGroup": self.bootstrap_group,
                "bootstrapResamples": self.bootstrap_resamples,
                "requireDifferentSpeakerForShuffle": (self.require_different_speaker_for_shuffle),
                "requireDifferentSessionForShuffle": (self.require_different_session_for_shuffle),
                "requireDifferentSourceForShuffle": (self.require_different_source_for_shuffle),
                "requireDifferentContextGroupForShuffle": (
                    self.require_different_context_group_for_shuffle
                ),
            }
        )
