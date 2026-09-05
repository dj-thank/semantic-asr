"""Reference-free preparation of shared phonetic pools and context scores."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass

from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256
from ..phonetic_experiment.planner import (
    FrozenPhoneticCandidatePlanner,
    FrozenPhoneticCandidatePool,
    PlanningCaseView,
)
from .context_scorer import (
    CandidateContextScorer,
    ContextCandidate,
    ContextCandidateScore,
)
from .protocol import (
    ContextPhoneticCase,
    ContextPhoneticManifest,
    ContextPhoneticProtocol,
    FrozenContextSnapshot,
)


@dataclass(frozen=True, slots=True)
class ContextScoreSet:
    condition: str
    context: FrozenContextSnapshot
    donor_case_id: str
    scores: tuple[ContextCandidateScore, ...]
    scorer_source: str
    scorer_profile_digest: str
    scoring_latency_ms: float

    def __post_init__(self) -> None:
        if self.condition not in {"ordered", "shuffled"}:
            raise ValueError("context score set must be ordered or shuffled")
        if not self.donor_case_id or not self.scorer_source:
            raise ValueError("context score set requires donor and scorer source")
        if self.context.source_case_id != self.donor_case_id:
            raise ValueError("context score donor does not match context source")
        if not _is_sha256(self.scorer_profile_digest):
            raise ValueError("scorer_profile_digest must be a SHA-256 value")
        if not self.scores:
            raise ValueError("context score set requires candidate scores")
        if len({row.candidate_id for row in self.scores}) != len(self.scores):
            raise ValueError("context score candidate IDs must be unique")
        for row in self.scores:
            if row.context_digest != self.context.digest:
                raise ValueError("context score is bound to a different context")
            if row.source != self.scorer_source:
                raise ValueError("context score source differs within one score set")
            if row.scorer_profile_digest != self.scorer_profile_digest:
                raise ValueError("context scorer profile differs within one score set")
        latency = float(self.scoring_latency_ms)
        if latency < 0.0:
            raise ValueError("scoring_latency_ms must be non-negative")
        object.__setattr__(self, "scoring_latency_ms", latency)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "condition": self.condition,
                "contextDigest": self.context.digest,
                "donorCaseId": self.donor_case_id,
                "scoreDigests": [row.digest for row in self.scores],
                "scorerSource": self.scorer_source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )

    def score(self, candidate_id: str) -> ContextCandidateScore:
        for row in self.scores:
            if row.candidate_id == candidate_id:
                return row
        raise KeyError(candidate_id)


@dataclass(frozen=True, slots=True)
class PreparedContextPhoneticCase:
    case_id: str
    case_planning_digest: str
    pool: FrozenPhoneticCandidatePool
    pool_evidence_digest: str
    ordered: ContextScoreSet
    shuffled: ContextScoreSet

    def __post_init__(self) -> None:
        if self.case_id != self.pool.case_id:
            raise ValueError("prepared case and phonetic pool case IDs differ")
        for value in (self.case_planning_digest, self.pool_evidence_digest):
            if not _is_sha256(value):
                raise ValueError("prepared case digests must be SHA-256 values")
        expected_ids = {row.candidate_id for row in self.pool.candidates}
        if {row.candidate_id for row in self.ordered.scores} != expected_ids:
            raise ValueError("ordered context scores do not cover the frozen pool exactly")
        if {row.candidate_id for row in self.shuffled.scores} != expected_ids:
            raise ValueError("shuffled context scores do not cover the frozen pool exactly")
        if self.ordered.scorer_source != self.shuffled.scorer_source:
            raise ValueError("ordered and shuffled contexts use different scorer sources")
        if self.ordered.scorer_profile_digest != self.shuffled.scorer_profile_digest:
            raise ValueError("ordered and shuffled contexts use different scorer profiles")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "caseId": self.case_id,
                "casePlanningDigest": self.case_planning_digest,
                "poolEvidenceDigest": self.pool_evidence_digest,
                "orderedScoreSetDigest": self.ordered.digest,
                "shuffledScoreSetDigest": self.shuffled.digest,
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedContextPhoneticExperiment:
    manifest_planning_digest: str
    protocol_digest: str
    phonetic_planner_profile_digest: str
    context_scorer_source: str
    context_scorer_profile_digest: str
    shuffle_assignment_digest: str
    cases: tuple[PreparedContextPhoneticCase, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for value in (
            self.manifest_planning_digest,
            self.protocol_digest,
            self.phonetic_planner_profile_digest,
            self.context_scorer_profile_digest,
            self.shuffle_assignment_digest,
        ):
            if not _is_sha256(value):
                raise ValueError("prepared factorial experiment digests must be SHA-256 values")
        if not self.context_scorer_source or not self.cases:
            raise ValueError("prepared factorial experiment requires scorer source and cases")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("prepared factorial case IDs must be unique")
        for case in self.cases:
            if case.ordered.scorer_source != self.context_scorer_source:
                raise ValueError("prepared case uses a different context scorer source")
            if (
                case.ordered.scorer_profile_digest
                != self.context_scorer_profile_digest
            ):
                raise ValueError("prepared case uses a different context scorer profile")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "manifestPlanningDigest": self.manifest_planning_digest,
                "protocolDigest": self.protocol_digest,
                "phoneticPlannerProfileDigest": self.phonetic_planner_profile_digest,
                "contextScorerSource": self.context_scorer_source,
                "contextScorerProfileDigest": self.context_scorer_profile_digest,
                "shuffleAssignmentDigest": self.shuffle_assignment_digest,
                "caseDigests": [case.digest for case in self.cases],
            }
        )

    def case(self, case_id: str) -> PreparedContextPhoneticCase:
        for row in self.cases:
            if row.case_id == case_id:
                return row
        raise KeyError(case_id)


def _pool_evidence_digest(pool: FrozenPhoneticCandidatePool) -> str:
    """Stable pool identity that deliberately excludes measured wall-clock latency."""

    return sha256_json(
        {
            "caseId": pool.case_id,
            "planningDigest": pool.planning_digest,
            "sourceAudioSha256": pool.source_audio_sha256,
            "runtimeProfileDigest": pool.runtime_profile_digest,
            "utilityArtifactDigest": pool.utility_artifact_digest,
            "lexiconDigest": pool.lexicon_digest,
            "phonePosteriorDigest": pool.phone_posterior_digest,
            "moraPosteriorDigest": pool.mora_posterior_digest,
            "candidateDigests": [row.digest for row in pool.candidates],
            "firstPassSelectedCandidateId": pool.first_pass_selected_candidate_id,
        }
    )


def _shuffle_compatible(
    receiver: ContextPhoneticCase,
    donor: ContextPhoneticCase,
    protocol: ContextPhoneticProtocol,
) -> bool:
    if receiver.case_id == donor.case_id:
        return False
    left = receiver.phonetic_case
    right = donor.phonetic_case
    if protocol.require_different_speaker_for_shuffle and left.speaker_id == right.speaker_id:
        return False
    if protocol.require_different_session_for_shuffle and left.session_id == right.session_id:
        return False
    if protocol.require_different_source_for_shuffle and left.source_id == right.source_id:
        return False
    return True


def deterministic_context_derangement(
    manifest: ContextPhoneticManifest,
    protocol: ContextPhoneticProtocol,
) -> dict[str, ContextPhoneticCase]:
    cases = tuple(
        sorted(
            manifest.cases,
            key=lambda case: hashlib.sha256(
                f"{protocol.shuffle_seed}:{case.case_id}".encode("utf-8")
            ).hexdigest(),
        )
    )
    if len(cases) < 2:
        raise ValueError("shuffled-context control requires at least two cases")
    for shift in range(1, len(cases)):
        donors = cases[shift:] + cases[:shift]
        if all(
            _shuffle_compatible(receiver, donor, protocol)
            for receiver, donor in zip(cases, donors, strict=True)
        ):
            return {
                receiver.case_id: donor
                for receiver, donor in zip(cases, donors, strict=True)
            }
    raise ValueError(
        "no deterministic context derangement satisfies the registered speaker/session/source exclusions"
    )


def _context_candidates(pool: FrozenPhoneticCandidatePool) -> tuple[ContextCandidate, ...]:
    return tuple(
        ContextCandidate(candidate_id=row.candidate_id, text=row.text)
        for row in pool.candidates
    )


def _score_context(
    scorer: CandidateContextScorer,
    candidates: tuple[ContextCandidate, ...],
    *,
    context: FrozenContextSnapshot,
    condition: str,
    donor_case_id: str,
) -> ContextScoreSet:
    started = time.perf_counter_ns()
    scores = tuple(scorer.score_many(candidates, context=context))
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    expected = {row.candidate_id: row for row in candidates}
    if len(scores) != len(expected):
        raise ValueError("context scorer returned the wrong number of candidate scores")
    seen: set[str] = set()
    for score in scores:
        candidate = expected.get(score.candidate_id)
        if candidate is None or score.candidate_id in seen:
            raise ValueError("context scorer returned unknown or duplicate candidate IDs")
        seen.add(score.candidate_id)
        if score.candidate_text_sha256 != candidate.text_sha256:
            raise ValueError("context score is bound to different candidate text")
        if score.context_digest != context.digest:
            raise ValueError("context score is bound to a different context snapshot")
    return ContextScoreSet(
        condition=condition,
        context=context,
        donor_case_id=donor_case_id,
        scores=tuple(sorted(scores, key=lambda row: row.candidate_id)),
        scorer_source=scorer.source,
        scorer_profile_digest=scorer.profile_digest,
        scoring_latency_ms=latency_ms,
    )


def prepare_context_phonetic_experiment(
    manifest: ContextPhoneticManifest,
    protocol: ContextPhoneticProtocol,
    phonetic_planner: FrozenPhoneticCandidatePlanner,
    context_scorer: CandidateContextScorer,
) -> PreparedContextPhoneticExperiment:
    """Freeze all acoustic candidates and ordered/shuffled context scores before evaluation."""

    assignments = deterministic_context_derangement(manifest, protocol)
    prepared: list[PreparedContextPhoneticCase] = []
    for case in manifest.cases:
        pool = phonetic_planner.plan(
            PlanningCaseView.from_case(case.phonetic_case),
            protocol=protocol.phonetic_protocol,
        )
        if pool.planning_digest != case.phonetic_case.planning_digest:
            raise ValueError("phonetic pool planning digest differs from the registered case")
        candidates = _context_candidates(pool)
        donor = assignments[case.case_id]
        ordered = _score_context(
            context_scorer,
            candidates,
            context=case.ordered_context,
            condition="ordered",
            donor_case_id=case.case_id,
        )
        shuffled = _score_context(
            context_scorer,
            candidates,
            context=donor.ordered_context,
            condition="shuffled",
            donor_case_id=donor.case_id,
        )
        prepared.append(
            PreparedContextPhoneticCase(
                case_id=case.case_id,
                case_planning_digest=case.planning_digest,
                pool=pool,
                pool_evidence_digest=_pool_evidence_digest(pool),
                ordered=ordered,
                shuffled=shuffled,
            )
        )
    assignment_digest = sha256_json(
        tuple(sorted((case_id, donor.case_id) for case_id, donor in assignments.items()))
    )
    return PreparedContextPhoneticExperiment(
        manifest_planning_digest=manifest.planning_digest,
        protocol_digest=protocol.digest,
        phonetic_planner_profile_digest=phonetic_planner.profile_digest,
        context_scorer_source=context_scorer.source,
        context_scorer_profile_digest=context_scorer.profile_digest,
        shuffle_assignment_digest=assignment_digest,
        cases=tuple(prepared),
    )
