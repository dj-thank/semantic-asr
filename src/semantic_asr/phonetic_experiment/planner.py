"""Reference-free planning of one shared phonetic candidate/evidence pool per case."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_bridge import (
    FrozenPronunciationLexicon,
    PhoneticBridgeConfig,
    propose_text_from_pronunciation,
)
from ..phonetic_runtime.calibration_artifact import DualCTCUtilityArtifact
from ..phonetic_runtime.provider import PhoneMoraPosteriorRuntime
from .protocol import FirstPassSpanCandidate, PhoneticAblationCase, PhoneticAblationProtocol


@dataclass(frozen=True, slots=True)
class PlanningCaseView:
    """Reference-free case surface passed to acoustic candidate planning."""

    case_id: str
    audio_path: Path
    source_audio_sha256: str
    start_ms: int
    end_ms: int
    first_pass_candidates: tuple[FirstPassSpanCandidate, ...]
    lexicon: FrozenPronunciationLexicon
    planning_digest: str

    @classmethod
    def from_case(cls, case: PhoneticAblationCase) -> PlanningCaseView:
        return cls(
            case_id=case.case_id,
            audio_path=case.audio_path,
            source_audio_sha256=case.source_audio_sha256,
            start_ms=case.start_ms,
            end_ms=case.end_ms,
            first_pass_candidates=case.first_pass_candidates,
            lexicon=case.lexicon,
            planning_digest=case.planning_digest,
        )


@dataclass(frozen=True, slots=True)
class FrozenPhoneticCandidate:
    candidate_id: str
    lexicon_entry_id: str
    text: str
    pronunciation_key: str
    phone_utility: float
    mora_utility: float
    discrete_unit_utility: float | None
    first_pass_candidate_ids: tuple[str, ...]
    first_pass_posterior: float
    first_pass_selected: bool
    proposal_digest: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.lexicon_entry_id or not self.text:
            raise ValueError("frozen phonetic candidate requires IDs and text")
        if not _is_sha256(self.pronunciation_key):
            raise ValueError("pronunciation_key must be a SHA-256 value")
        if not _is_sha256(self.proposal_digest):
            raise ValueError("proposal_digest must be a SHA-256 value")
        for name in ("phone_utility", "mora_utility"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
            object.__setattr__(self, name, value)
        if self.discrete_unit_utility is not None:
            value = float(self.discrete_unit_utility)
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("discrete_unit_utility must be finite and in [-1, 1]")
            object.__setattr__(self, "discrete_unit_utility", value)
        posterior = float(self.first_pass_posterior)
        if not math.isfinite(posterior) or not 0.0 <= posterior <= 1.0:
            raise ValueError("first_pass_posterior must be in [0, 1]")
        if not isinstance(self.first_pass_selected, bool):
            raise TypeError("first_pass_selected must be a boolean")
        object.__setattr__(self, "first_pass_posterior", posterior)
        object.__setattr__(
            self,
            "first_pass_candidate_ids",
            tuple(dict.fromkeys(self.first_pass_candidate_ids)),
        )

    @property
    def is_first_pass(self) -> bool:
        return bool(self.first_pass_candidate_ids)

    @property
    def text_sha256(self) -> str:
        return sha256_json({"text": self.text})

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "text": None,
                "textSha256": self.text_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenPhoneticCandidatePool:
    case_id: str
    planning_digest: str
    source_audio_sha256: str
    runtime_profile_digest: str
    utility_artifact_digest: str
    lexicon_digest: str
    phone_posterior_digest: str
    mora_posterior_digest: str
    candidates: tuple[FrozenPhoneticCandidate, ...]
    first_pass_selected_candidate_id: str
    generation_latency_ms: float
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.case_id or not self.candidates:
            raise ValueError("frozen candidate pool requires case ID and candidates")
        for value in (
            self.planning_digest,
            self.source_audio_sha256,
            self.runtime_profile_digest,
            self.utility_artifact_digest,
            self.lexicon_digest,
            self.phone_posterior_digest,
            self.mora_posterior_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("candidate pool digests must be SHA-256 values")
        if len({row.candidate_id for row in self.candidates}) != len(self.candidates):
            raise ValueError("candidate pool IDs must be unique")
        if len({row.text for row in self.candidates}) != len(self.candidates):
            raise ValueError("candidate pool surfaces must be unique")
        selected = [row for row in self.candidates if row.first_pass_selected]
        if len(selected) != 1 or selected[0].candidate_id != self.first_pass_selected_candidate_id:
            raise ValueError("candidate pool first-pass selected identity is inconsistent")
        latency = float(self.generation_latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("generation_latency_ms must be finite and non-negative")
        object.__setattr__(self, "generation_latency_ms", latency)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "caseId": self.case_id,
                "planningDigest": self.planning_digest,
                "sourceAudioSha256": self.source_audio_sha256,
                "runtimeProfileDigest": self.runtime_profile_digest,
                "utilityArtifactDigest": self.utility_artifact_digest,
                "lexiconDigest": self.lexicon_digest,
                "phonePosteriorDigest": self.phone_posterior_digest,
                "moraPosteriorDigest": self.mora_posterior_digest,
                "candidateDigests": [row.digest for row in self.candidates],
                "firstPassSelectedCandidateId": self.first_pass_selected_candidate_id,
            }
        )

    def candidate(self, candidate_id: str) -> FrozenPhoneticCandidate:
        for row in self.candidates:
            if row.candidate_id == candidate_id:
                return row
        raise KeyError(candidate_id)


class FrozenPhoneticCandidatePlanner:
    """Generate the shared candidate/evidence pool once, without reference access."""

    def __init__(
        self,
        runtime: PhoneMoraPosteriorRuntime,
        utility_artifact: DualCTCUtilityArtifact,
    ) -> None:
        if runtime.profile_digest != utility_artifact.runtime_profile_digest:
            raise ValueError("utility artifact belongs to a different phonetic runtime")
        self.runtime = runtime
        self.utility_artifact = utility_artifact

    @property
    def profile_digest(self) -> str:
        return sha256_json(
            {
                "runtimeProfileDigest": self.runtime.profile_digest,
                "utilityArtifactDigest": self.utility_artifact.digest,
                "plannerRevision": "frozen-phonetic-pool-v1",
            }
        )

    def plan(
        self,
        case: PlanningCaseView,
        *,
        protocol: PhoneticAblationProtocol,
    ) -> FrozenPhoneticCandidatePool:
        if case.end_ms - case.start_ms > protocol.maximum_crop_ms:
            raise ValueError("phonetic ablation crop exceeds protocol maximum_crop_ms")
        if len(case.lexicon.entries) > protocol.maximum_candidates:
            raise ValueError("frozen lexicon exceeds protocol maximum_candidates")
        started = time.perf_counter_ns()
        phone, mora = self.runtime.infer(
            case.audio_path,
            start_ms=case.start_ms,
            end_ms=case.end_ms,
            expected_source_audio_sha256=case.source_audio_sha256,
        )
        proposals = propose_text_from_pronunciation(
            case.lexicon,
            phone_posterior=phone,
            mora_posterior=mora,
            phone_calibration=self.utility_artifact.phone_profile,
            mora_calibration=self.utility_artifact.mora_profile,
            config=PhoneticBridgeConfig(top_k=len(case.lexicon.entries)),
        )
        if len(proposals) != len(case.lexicon.entries):
            raise ValueError("frozen planner did not score every exogenous lexicon entry")
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        by_text: dict[str, list[FirstPassSpanCandidate]] = {}
        for row in case.first_pass_candidates:
            by_text.setdefault(row.text, []).append(row)
        frozen: list[FrozenPhoneticCandidate] = []
        for proposal in proposals:
            first_pass_rows = tuple(by_text.get(proposal.text, ()))
            utilities = {utility.channel: utility.value for utility in proposal.utilities}
            if set(utilities) != {"phone", "mora"}:
                raise ValueError("phonetic planner requires both phone and mora utilities")
            frozen.append(
                FrozenPhoneticCandidate(
                    candidate_id=f"surface-{proposal.candidate_id}",
                    lexicon_entry_id=proposal.entry_id,
                    text=proposal.text,
                    pronunciation_key=proposal.pronunciation_key,
                    phone_utility=utilities["phone"],
                    mora_utility=utilities["mora"],
                    discrete_unit_utility=None,
                    first_pass_candidate_ids=tuple(row.candidate_id for row in first_pass_rows),
                    first_pass_posterior=sum(row.posterior for row in first_pass_rows),
                    first_pass_selected=any(row.selected for row in first_pass_rows),
                    proposal_digest=sha256_json(
                        {
                            "candidateId": proposal.candidate_id,
                            "entryId": proposal.entry_id,
                            "pronunciationKey": proposal.pronunciation_key,
                            "utilityDigests": [utility.digest for utility in proposal.utilities],
                            "phonePosteriorDigest": proposal.phone_score.posterior_digest,
                            "phonePronunciationDigest": proposal.phone_score.pronunciation_digest,
                            "moraPosteriorDigest": proposal.mora_score.posterior_digest,
                            "moraPronunciationDigest": proposal.mora_score.pronunciation_digest,
                        }
                    ),
                )
            )
        selected = [row for row in frozen if row.first_pass_selected]
        if len(selected) != 1:
            raise ValueError(
                "frozen lexicon/proposal pool did not preserve exactly one selected first-pass surface"
            )
        return FrozenPhoneticCandidatePool(
            case_id=case.case_id,
            planning_digest=case.planning_digest,
            source_audio_sha256=case.source_audio_sha256,
            runtime_profile_digest=self.runtime.profile_digest,
            utility_artifact_digest=self.utility_artifact.digest,
            lexicon_digest=case.lexicon.digest,
            phone_posterior_digest=phone.digest,
            mora_posterior_digest=mora.digest,
            candidates=tuple(sorted(frozen, key=lambda row: row.candidate_id)),
            first_pass_selected_candidate_id=selected[0].candidate_id,
            generation_latency_ms=elapsed_ms,
        )
