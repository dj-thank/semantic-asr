"""Frozen, leakage-resistant protocol types for document-context ASR experiments."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..audio import require_integer
from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from ..deliberation_lattice import DocumentContext
from ..longform import LongformResult

RightsDecision = Literal["allow", "review", "deny"]
CandidateView = Literal["acoustic-only", "ordered-document", "shuffled-document"]
ScoringDirection = Literal["none", "forward", "bidirectional"]
CriticalKind = Literal[
    "number",
    "date",
    "time",
    "currency",
    "percentage",
    "negation",
    "modality",
    "entity",
    "repair",
    "filler",
    "other",
]


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _reference_surface(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace() and not unicodedata.category(character).startswith(("P", "S"))
    )


@dataclass(frozen=True, slots=True)
class CriticalReferenceToken:
    kind: CriticalKind
    text: str
    count: int = 1
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {
            "number",
            "date",
            "time",
            "currency",
            "percentage",
            "negation",
            "modality",
            "entity",
            "repair",
            "filler",
            "other",
        }:
            raise ValueError("unknown critical-token kind")
        _nonempty(self.text, name="critical token text")
        require_integer(self.count, name="critical token count", minimum=1)
        aliases = tuple(dict.fromkeys(self.aliases))
        if any(not alias for alias in aliases):
            raise ValueError("critical-token aliases must not be empty")
        object.__setattr__(self, "aliases", aliases)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class FrozenReference:
    """Reference-bearing object that is never passed into candidate planning or scoring."""

    reference_id: str
    source_audio_sha256: str
    text: str
    window_texts: tuple[str, ...]
    critical_tokens: tuple[CriticalReferenceToken, ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, name="reference_id")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("reference source_audio_sha256 must be a SHA-256 value")
        _nonempty(self.text, name="reference text")
        if not self.window_texts or any(not isinstance(row, str) for row in self.window_texts):
            raise ValueError("reference requires one string per first-pass window")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "referenceId": self.reference_id,
                "sourceAudioSha256": self.source_audio_sha256,
                "textSha256": sha256_json({"text": self.text}),
                "windowTextSha256": [sha256_json({"text": row}) for row in self.window_texts],
                "criticalTokenDigests": [row.digest for row in self.critical_tokens],
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenExternalContext:
    name: str
    context: DocumentContext
    provenance_sha256: str
    frozen_before_evaluation: bool = True

    def __post_init__(self) -> None:
        _nonempty(self.name, name="context name")
        if not _is_sha256(self.provenance_sha256):
            raise ValueError("context provenance_sha256 must be a SHA-256 value")
        if not isinstance(self.frozen_before_evaluation, bool):
            raise TypeError("frozen_before_evaluation must be a boolean")
        if not self.frozen_before_evaluation:
            raise ValueError("evaluation context must be frozen before evaluation audio is scored")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "name": self.name,
                "contextDigest": self.context.digest,
                "provenanceSha256": self.provenance_sha256,
                "frozenBeforeEvaluation": self.frozen_before_evaluation,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentExperimentCase:
    case_id: str
    first_pass: LongformResult = field(repr=False)
    reference: FrozenReference = field(repr=False)
    rights_decision: RightsDecision = "allow"
    license_id: str = ""
    source_id: str = ""
    speaker_id: str = ""
    session_id: str = ""
    dataset_revision: str = ""
    split_manifest_sha256: str = ""
    external_contexts: tuple[FrozenExternalContext, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.case_id, name="case_id")
        if self.rights_decision != "allow":
            raise ValueError("reference-bearing experiment cases require rights_decision='allow'")
        for name in (
            "license_id",
            "source_id",
            "speaker_id",
            "session_id",
            "dataset_revision",
        ):
            _nonempty(getattr(self, name), name=name)
        if not _is_sha256(self.split_manifest_sha256):
            raise ValueError("split_manifest_sha256 must be a SHA-256 value")
        self.first_pass.verify()
        if self.first_pass.source_audio_sha256 != self.reference.source_audio_sha256:
            raise ValueError("reference and first-pass result belong to different audio")
        if len(self.reference.window_texts) != len(self.first_pass.segments):
            raise ValueError("reference window count must match first-pass window count")
        names = [row.name for row in self.external_contexts]
        if len(names) != len(set(names)):
            raise ValueError("external context names must be unique within a case")
        reference_surface = _reference_surface(self.reference.text)
        for row in self.external_contexts:
            combined = _reference_surface(
                "\n".join(
                    (
                        row.context.left_context,
                        row.context.right_context,
                        row.context.topic_summary,
                    )
                )
            )
            if len(reference_surface) >= 24 and reference_surface in combined:
                raise ValueError("external context contains the complete evaluation reference")

    def context(self, name: str | None) -> DocumentContext:
        if name is None:
            return DocumentContext()
        for row in self.external_contexts:
            if row.name == name:
                return row.context
        raise KeyError(name)

    @property
    def planning_digest(self) -> str:
        """Identity available before the evaluator opens the reference text."""

        return sha256_json(
            {
                "caseId": self.case_id,
                "firstPassEvidenceSha256": self.first_pass.evidence_sha256,
                "sourceAudioSha256": self.first_pass.source_audio_sha256,
                "licenseId": self.license_id,
                "sourceId": self.source_id,
                "speakerId": self.speaker_id,
                "sessionId": self.session_id,
                "datasetRevision": self.dataset_revision,
                "splitManifestSha256": self.split_manifest_sha256,
                "externalContextDigests": [row.digest for row in self.external_contexts],
            }
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "planningDigest": self.planning_digest,
                "referenceDigest": self.reference.digest,
                "rightsDecision": self.rights_decision,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentExperimentManifest:
    name: str
    revision: str
    cases: tuple[DocumentExperimentCase, ...]
    rights_registry_sha256: str
    split_manifest_sha256: str
    training_speaker_ids: tuple[str, ...] = ()
    calibration_speaker_ids: tuple[str, ...] = ()
    training_session_ids: tuple[str, ...] = ()
    calibration_session_ids: tuple[str, ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        _nonempty(self.name, name="manifest name")
        _nonempty(self.revision, name="manifest revision")
        if not self.cases:
            raise ValueError("experiment manifest requires at least one case")
        if not _is_sha256(self.rights_registry_sha256):
            raise ValueError("rights_registry_sha256 must be a SHA-256 value")
        if not _is_sha256(self.split_manifest_sha256):
            raise ValueError("split_manifest_sha256 must be a SHA-256 value")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("experiment case IDs must be unique")
        if len({case.first_pass.source_audio_sha256 for case in self.cases}) != len(self.cases):
            raise ValueError("test audio must not be duplicated across experiment cases")
        if any(case.split_manifest_sha256 != self.split_manifest_sha256 for case in self.cases):
            raise ValueError("case split manifest does not match the experiment manifest")
        test_speakers = {case.speaker_id for case in self.cases}
        test_sessions = {case.session_id for case in self.cases}
        training_speakers = set(self.training_speaker_ids)
        calibration_speakers = set(self.calibration_speaker_ids)
        training_sessions = set(self.training_session_ids)
        calibration_sessions = set(self.calibration_session_ids)
        if training_speakers.intersection(calibration_speakers):
            raise ValueError("training and calibration speakers overlap")
        if test_speakers.intersection(training_speakers | calibration_speakers):
            raise ValueError("test speakers overlap training or calibration")
        if training_sessions.intersection(calibration_sessions):
            raise ValueError("training and calibration sessions overlap")
        if test_sessions.intersection(training_sessions | calibration_sessions):
            raise ValueError("test sessions overlap training or calibration")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "caseDigests": [case.digest for case in self.cases],
                "rightsRegistrySha256": self.rights_registry_sha256,
                "splitManifestSha256": self.split_manifest_sha256,
                "trainingSpeakerIds": self.training_speaker_ids,
                "calibrationSpeakerIds": self.calibration_speaker_ids,
                "trainingSessionIds": self.training_session_ids,
                "calibrationSessionIds": self.calibration_session_ids,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentExperimentArm:
    name: str
    candidate_view: CandidateView
    direction: ScoringDirection
    scorer_key: str | None = None
    external_context_name: str | None = None
    linguistic_weight: float = 1.0
    minimum_margin: float = 0.02
    shuffled_seed: str = "semantic-asr-document-shuffle-v1"
    is_control: bool = False

    def __post_init__(self) -> None:
        _nonempty(self.name, name="arm name")
        if self.candidate_view not in {
            "acoustic-only",
            "ordered-document",
            "shuffled-document",
        }:
            raise ValueError("unknown candidate view")
        if self.direction not in {"none", "forward", "bidirectional"}:
            raise ValueError("unknown scoring direction")
        if self.candidate_view == "acoustic-only" or self.direction == "none":
            if self.candidate_view != "acoustic-only" or self.direction != "none":
                raise ValueError("acoustic-only candidate view and none direction must be paired")
            if self.scorer_key is not None:
                raise ValueError("acoustic-only arm cannot name a linguistic scorer")
        elif self.scorer_key is None:
            raise ValueError("linguistic experiment arms require scorer_key")
        weight = _strict_float(self.linguistic_weight, name="linguistic_weight")
        margin = _strict_float(self.minimum_margin, name="minimum_margin")
        if weight < 0.0 or margin < 0.0:
            raise ValueError("arm weight and margin must be non-negative")
        object.__setattr__(self, "linguistic_weight", weight)
        object.__setattr__(self, "minimum_margin", margin)
        if not isinstance(self.is_control, bool):
            raise TypeError("is_control must be a boolean")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentExperimentProtocol:
    name: str
    revision: str
    arms: tuple[DocumentExperimentArm, ...]
    baseline_arm: str
    maximum_candidate_documents: int = 32
    maximum_scored_characters: int = 200_000
    bootstrap_resamples: int = 2_000
    bootstrap_seed: int = 20260905
    fail_on_case_error: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        _nonempty(self.name, name="protocol name")
        _nonempty(self.revision, name="protocol revision")
        if len(self.arms) < 2:
            raise ValueError("experiment protocol requires at least two arms")
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("experiment arm names must be unique")
        if self.baseline_arm not in names:
            raise ValueError("baseline_arm is not present in arms")
        baseline = next(arm for arm in self.arms if arm.name == self.baseline_arm)
        if baseline.candidate_view != "acoustic-only":
            raise ValueError("baseline arm must be acoustic-only")
        for name in (
            "maximum_candidate_documents",
            "maximum_scored_characters",
            "bootstrap_resamples",
            "bootstrap_seed",
        ):
            require_integer(getattr(self, name), name=name)
        if self.maximum_candidate_documents < 2:
            raise ValueError("maximum_candidate_documents must be at least two")
        if self.maximum_scored_characters < 1:
            raise ValueError("maximum_scored_characters must be positive")
        if self.bootstrap_resamples < 200:
            raise ValueError("bootstrap_resamples must be at least 200")
        if not isinstance(self.fail_on_case_error, bool):
            raise TypeError("fail_on_case_error must be a boolean")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "name": self.name,
                "revision": self.revision,
                "armDigests": [arm.digest for arm in self.arms],
                "baselineArm": self.baseline_arm,
                "maximumCandidateDocuments": self.maximum_candidate_documents,
                "maximumScoredCharacters": self.maximum_scored_characters,
                "bootstrapResamples": self.bootstrap_resamples,
                "bootstrapSeed": self.bootstrap_seed,
                "failOnCaseError": self.fail_on_case_error,
                "referenceBoundary": "planning-before-reference-evaluation-v1",
                "metrics": "document-context-metrics-v1",
            }
        )
