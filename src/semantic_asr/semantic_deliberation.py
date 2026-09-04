"""Project whole-hypothesis ASR evidence into an exact local deliberation lattice.

Every source candidate is aligned to the selected first-pass candidate, divided at the union of all
edit boundaries, and assigned to each resulting span exactly once. Concatenating a candidate's
projected slices must reproduce its original text byte-for-byte. Local utilities share a finite
factor budget, so a whole-hypothesis score is not counted once per character or ambiguity island.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Literal

from .contracts import CandidateEvidence, sha256_json
from .deliberation_evidence import (
    INDEPENDENT_AUDIO_CHANNELS,
    BoundedUtility,
    UtilityChannel,
    _is_sha256,
)
from .deliberation_lattice import (
    DeliberationLattice,
    DeliberationSpan,
    LatticeArc,
    SourcePath,
    TransitionUtility,
)
from .semantic_lattice import _classify, _criticality

ProposalOrigin = Literal["phonetic-proposal", "context-proposal", "guarded-generation"]


def _strict_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _normalize_distribution(
    candidate_ids: Sequence[str],
    values: Mapping[str, float] | None,
) -> dict[str, float]:
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if values is None:
        uniform = 1.0 / len(candidate_ids)
        return {candidate_id: uniform for candidate_id in candidate_ids}
    if set(values) != set(candidate_ids):
        raise ValueError("distribution IDs must match candidate IDs exactly")
    rows = {
        candidate_id: _strict_probability(values[candidate_id], name="candidate probability")
        for candidate_id in candidate_ids
    }
    total = sum(rows.values())
    if total <= 0.0:
        raise ValueError("candidate probabilities must contain positive mass")
    return {candidate_id: value / total for candidate_id, value in rows.items()}


def _softmax(values: Mapping[str, float], *, temperature: float) -> dict[str, float]:
    if not values:
        return {}
    numeric: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            raise TypeError("evidence scores must be real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("evidence scores must be finite")
        numeric[key] = number
    maximum = max(numeric.values())
    exponentials = {
        key: math.exp(max(-80.0, min(80.0, (value - maximum) / temperature)))
        for key, value in numeric.items()
    }
    total = sum(exponentials.values()) or 1.0
    return {key: value / total for key, value in exponentials.items()}


def _candidate_relative_distributions(
    candidates: Sequence[CandidateEvidence],
    *,
    acoustic_temperature: float,
    stream_temperature: float,
) -> dict[UtilityChannel, tuple[str, dict[str, float]]]:
    output: dict[UtilityChannel, tuple[str, dict[str, float]]] = {}
    for field_name in ("acoustic", "avg_logprob", "sequence_score"):
        values = {candidate.candidate_id: getattr(candidate, field_name) for candidate in candidates}
        if all(value is not None for value in values.values()):
            output["asr_acoustic"] = (
                field_name,
                _softmax(
                    {candidate_id: float(value) for candidate_id, value in values.items()},
                    temperature=acoustic_temperature,
                ),
            )
            break
    for channel, field_name in (
        ("mora_shadow", "mora"),
        ("lexical", "lexical"),
        ("preservation", "preservation"),
        ("cross_model", "cross_model"),
    ):
        values = {candidate.candidate_id: getattr(candidate, field_name) for candidate in candidates}
        if all(value is not None for value in values.values()):
            output[channel] = (
                field_name,
                _softmax(
                    {candidate_id: float(value) for candidate_id, value in values.items()},
                    temperature=stream_temperature,
                ),
            )
    return output


def _bounded_mass(mass: float) -> float:
    return min(1.0, max(-1.0, 2.0 * mass - 1.0))


def _profile_digest(kind: str, payload: Mapping[str, object]) -> str:
    return sha256_json({"kind": kind, "revision": "1", "payload": dict(payload)})


def _pronunciation_key(candidates: Sequence[CandidateEvidence]) -> str | None:
    """Return a key only when every supporting full candidate has one identical reading."""

    values: set[str] = set()
    for candidate in candidates:
        if candidate.mora_units:
            units = tuple(unit.kana for unit in candidate.mora_units)
        elif candidate.reading:
            units = tuple(candidate.reading)
        else:
            return None
        values.add(sha256_json({"kind": "candidate-pronunciation-v1", "units": units}))
    return next(iter(values)) if len(values) == 1 else None


@dataclass(frozen=True, slots=True)
class ProjectedCandidate:
    candidate_id: str
    source_text_sha256: str
    span_texts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not _is_sha256(self.source_text_sha256):
            raise ValueError("projected candidate requires an ID and source-text SHA-256")
        if not self.span_texts:
            raise ValueError("projected candidate requires at least one span")
        if len({span_id for span_id, _ in self.span_texts}) != len(self.span_texts):
            raise ValueError("projected candidate span IDs must be unique")

    @property
    def text(self) -> str:
        return "".join(text for _, text in self.span_texts)

    def verify(self) -> None:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if digest != self.source_text_sha256:
            raise ValueError("candidate projection does not exactly reconstruct source text")


@dataclass(frozen=True, slots=True)
class VerifiedSpanProposal:
    proposal_id: str
    text: str
    utilities: tuple[BoundedUtility, ...]
    source_audio_sha256: str
    origin: ProposalOrigin = "phonetic-proposal"
    pronunciation_key: str | None = None
    source_candidate_ids: tuple[str, ...] = ()
    observed_eligible: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("verified span proposal requires proposal_id")
        if self.origin not in {
            "phonetic-proposal",
            "context-proposal",
            "guarded-generation",
        }:
            raise ValueError("verified span proposal has an invalid origin")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("verified span proposal requires source-audio SHA-256")
        channels = [utility.channel for utility in self.utilities]
        if len(channels) != len(set(channels)):
            raise ValueError("proposal utility channels must be unique")
        if self.observed_eligible and not set(channels).intersection(INDEPENDENT_AUDIO_CHANNELS):
            raise ValueError(
                "observed-eligible proposals require phone, mora, or discrete-unit evidence"
            )
        object.__setattr__(
            self,
            "utilities",
            tuple(sorted(self.utilities, key=lambda utility: utility.channel)),
        )
        object.__setattr__(
            self,
            "source_candidate_ids",
            tuple(dict.fromkeys(self.source_candidate_ids)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "proposalId": self.proposal_id,
                "text": self.text,
                "origin": self.origin,
                "utilityDigests": [utility.digest for utility in self.utilities],
                "sourceAudioSha256": self.source_audio_sha256,
                "pronunciationKey": self.pronunciation_key,
                "sourceCandidateIds": self.source_candidate_ids,
                "observedEligible": self.observed_eligible,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class SemanticDeliberationConfig:
    acoustic_temperature: float = 1.0
    stream_temperature: float = 1.0
    transition_temperature: float = 1.0
    transition_epsilon: float = 1e-9
    include_consensus_utilities: bool = False

    def __post_init__(self) -> None:
        for name in (
            "acoustic_temperature",
            "stream_temperature",
            "transition_temperature",
            "transition_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, number)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "acousticTemperature": self.acoustic_temperature,
                "streamTemperature": self.stream_temperature,
                "transitionTemperature": self.transition_temperature,
                "transitionEpsilon": self.transition_epsilon,
                "includeConsensusUtilities": self.include_consensus_utilities,
                "projection": "union-edit-boundaries-with-forward-insertions-v2",
                "factorBudget": "normalized-ambiguity-criticality-width-v1",
                "moraSemantics": "candidate-derived-mora-is-mora-shadow-v1",
            }
        )


@dataclass(frozen=True, slots=True)
class SemanticDeliberationBuild:
    lattice: DeliberationLattice
    projections: tuple[ProjectedCandidate, ...]
    candidate_posteriors: tuple[tuple[str, float], ...]
    pivot_candidate_id: str
    config_digest: str
    proposal_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _is_sha256(self.config_digest):
            raise ValueError("config_digest must be a SHA-256 value")
        if len({candidate_id for candidate_id, _ in self.candidate_posteriors}) != len(
            self.candidate_posteriors
        ):
            raise ValueError("candidate posterior IDs must be unique")
        self.verify()

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "latticeDigest": self.lattice.digest,
                "projectionDigests": [
                    sha256_json(
                        {
                            "candidateId": row.candidate_id,
                            "sourceTextSha256": row.source_text_sha256,
                            "spanTexts": row.span_texts,
                        }
                    )
                    for row in self.projections
                ],
                "candidatePosteriors": self.candidate_posteriors,
                "pivotCandidateId": self.pivot_candidate_id,
                "configDigest": self.config_digest,
                "proposalDigests": self.proposal_digests,
            }
        )

    def projection(self, candidate_id: str) -> ProjectedCandidate:
        for row in self.projections:
            if row.candidate_id == candidate_id:
                return row
        raise KeyError(candidate_id)

    def verify(self) -> None:
        for row in self.projections:
            row.verify()
        pivot = self.projection(self.pivot_candidate_id)
        retained = "".join(span.retained_arc.text for span in self.lattice.spans)
        if retained != pivot.text:
            raise ValueError("retained lattice path does not reconstruct the pivot candidate")
        source_ids = {path.candidate_id for path in self.lattice.source_paths}
        projection_ids = {row.candidate_id for row in self.projections}
        if source_ids != projection_ids:
            raise ValueError("lattice source paths do not match projected candidates")


@dataclass(frozen=True, slots=True)
class _CandidateAlignment:
    candidate_id: str
    text_units: tuple[str, ...]
    opcodes: tuple[tuple[str, int, int, int, int], ...]

    @property
    def insertions(self) -> dict[int, tuple[str, ...]]:
        values: dict[int, list[str]] = defaultdict(list)
        for tag, pivot_start, pivot_end, candidate_start, candidate_end in self.opcodes:
            if tag == "insert":
                if pivot_start != pivot_end:
                    raise ValueError("invalid insertion opcode")
                values[pivot_start].extend(self.text_units[candidate_start:candidate_end])
        return {key: tuple(value) for key, value in values.items()}

    def project_interval(self, start: int, end: int, *, pivot_length: int) -> tuple[str, ...]:
        if not 0 <= start < end <= pivot_length:
            raise ValueError("projection interval is outside the pivot")
        output: list[str] = []
        insertions = self.insertions
        output.extend(insertions.get(start, ()))
        for tag, pivot_start, pivot_end, candidate_start, candidate_end in self.opcodes:
            if pivot_start == pivot_end:
                continue
            left = max(start, pivot_start)
            right = min(end, pivot_end)
            if left >= right or tag == "delete":
                continue
            pivot_width = pivot_end - pivot_start
            candidate_width = candidate_end - candidate_start
            projected_start = candidate_start + (
                (left - pivot_start) * candidate_width // pivot_width
            )
            projected_end = candidate_start + (
                (right - pivot_start) * candidate_width // pivot_width
            )
            if left == pivot_start:
                projected_start = candidate_start
            if right == pivot_end:
                projected_end = candidate_end
            output.extend(self.text_units[projected_start:projected_end])
        if end == pivot_length:
            output.extend(insertions.get(pivot_length, ()))
        return tuple(output)


@dataclass(frozen=True, slots=True)
class _SpanPlan:
    index: int
    unit_start: int
    unit_end: int
    start_ms: int
    end_ms: int
    timing_source: str
    texts: tuple[tuple[str, str], ...]
    semantic_kinds: tuple[str, ...]
    semantic_criticality: float
    posterior_ambiguity: float

    @property
    def text_map(self) -> dict[str, str]:
        return dict(self.texts)

    @property
    def is_contradiction(self) -> bool:
        return len({text for _, text in self.texts}) > 1

    @property
    def factor_signal(self) -> float:
        width = max(1, self.unit_end - self.unit_start)
        return (
            width
            * max(0.05, self.posterior_ambiguity)
            * max(0.25, self.semantic_criticality)
        )


def _alignments(
    pivot: CandidateEvidence,
    candidates: Sequence[CandidateEvidence],
) -> tuple[tuple[tuple[int, int], ...], dict[str, _CandidateAlignment]]:
    pivot_units = tuple(pivot.text)
    boundaries = {0, len(pivot_units)}
    alignments: dict[str, _CandidateAlignment] = {}
    for candidate in candidates:
        units = tuple(candidate.text)
        opcodes = tuple(SequenceMatcher(a=pivot_units, b=units, autojunk=False).get_opcodes())
        for _tag, start, end, _candidate_start, _candidate_end in opcodes:
            boundaries.update((start, end))
        alignments[candidate.candidate_id] = _CandidateAlignment(
            candidate_id=candidate.candidate_id,
            text_units=units,
            opcodes=opcodes,
        )
    ordered = sorted(boundaries)
    intervals = tuple(
        (left, right)
        for left, right in zip(ordered, ordered[1:], strict=False)
        if left < right
    )
    if not intervals:
        raise ValueError("pivot candidate produced no deliberation intervals")
    return intervals, alignments


def _proportional_time_boundaries(
    intervals: Sequence[tuple[int, int]],
    *,
    pivot_length: int,
    start_ms: int,
    end_ms: int,
) -> tuple[tuple[int, int], ...]:
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("deliberation window requires 0 <= start_ms < end_ms")
    if end_ms - start_ms < len(intervals):
        raise ValueError("window is too short to assign monotonic millisecond spans")
    times = [start_ms]
    for index, (_left, right) in enumerate(intervals[:-1], 1):
        target = start_ms + round((end_ms - start_ms) * right / pivot_length)
        minimum = times[-1] + 1
        maximum = end_ms - (len(intervals) - index)
        times.append(min(max(target, minimum), maximum))
    times.append(end_ms)
    return tuple(zip(times[:-1], times[1:], strict=True))


def _timeline_time_boundaries(
    intervals: Sequence[tuple[int, int]],
    *,
    pivot_length: int,
    start_ms: int,
    end_ms: int,
    pivot_timeline: Sequence[tuple[int, int, int, int]],
) -> tuple[tuple[tuple[int, int], ...], str]:
    fallback = (
        _proportional_time_boundaries(
            intervals,
            pivot_length=pivot_length,
            start_ms=start_ms,
            end_ms=end_ms,
        ),
        "proportional-surface",
    )
    rows = tuple(pivot_timeline)
    if not rows:
        return fallback
    previous_char_end = 0
    previous_time_end = start_ms
    boundary_times = {0: start_ms, pivot_length: end_ms}
    for char_start, char_end, time_start, time_end in rows:
        if not (0 <= char_start < char_end <= pivot_length):
            return fallback
        if not (start_ms <= time_start < time_end <= end_ms):
            return fallback
        if char_start != previous_char_end or time_start < previous_time_end:
            return fallback
        width = char_end - char_start
        for offset in range(width + 1):
            boundary_times[char_start + offset] = time_start + round(
                (time_end - time_start) * offset / width
            )
        previous_char_end = char_end
        previous_time_end = time_end
    if previous_char_end != pivot_length:
        return fallback
    times: list[tuple[int, int]] = []
    previous = start_ms
    for index, (left, right) in enumerate(intervals):
        left_time = max(previous, boundary_times[left])
        right_time = boundary_times[right]
        remaining = len(intervals) - index - 1
        right_time = min(max(right_time, left_time + 1), end_ms - remaining)
        times.append((left_time, right_time))
        previous = right_time
    return tuple(times), "exact-pivot-timeline"


def _local_ambiguity(masses: Sequence[float]) -> float:
    positive = [value for value in masses if value > 0.0]
    if len(positive) <= 1:
        return 0.0
    total = sum(positive)
    probabilities = [value / total for value in positive]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return min(1.0, entropy / math.log(len(probabilities)))


def _factor_weights(
    plans: Sequence[_SpanPlan],
    *,
    include_consensus: bool,
) -> dict[int, float]:
    active = [plan for plan in plans if include_consensus or plan.is_contradiction]
    if not active:
        return {plan.index: 0.0 for plan in plans}
    total = sum(plan.factor_signal for plan in active)
    if total <= 0.0:
        uniform = 1.0 / len(active)
        active_weights = {plan.index: uniform for plan in active}
    else:
        active_weights = {plan.index: plan.factor_signal / total for plan in active}
    return {plan.index: active_weights.get(plan.index, 0.0) for plan in plans}


def _utility(
    *,
    channel: UtilityChannel,
    value: float,
    source: str,
    profile_digest: str,
    payload: Mapping[str, object],
    factor_weight: float,
) -> BoundedUtility:
    return BoundedUtility(
        channel=channel,
        value=value,
        source=source,
        profile_digest=profile_digest,
        input_digest=sha256_json(dict(payload)),
        factor_weight=factor_weight,
    )


def _base_arc_groups(
    span_id: str,
    texts: Mapping[str, str],
    candidates_by_id: Mapping[str, CandidateEvidence],
    posterior: Mapping[str, float],
    distributions: Mapping[UtilityChannel, tuple[str, Mapping[str, float]]],
    *,
    retained_candidate_id: str,
    source_audio_sha256: str,
    semantic_kinds: tuple[str, ...],
    semantic_criticality: float,
    ambiguity: float,
    factor_weight: float,
    config: SemanticDeliberationConfig,
) -> tuple[LatticeArc, ...]:
    groups: dict[str, list[str]] = defaultdict(list)
    for candidate_id, text in texts.items():
        groups[text].append(candidate_id)
    posterior_profile = _profile_digest(
        "first-pass-local-surface-marginal",
        {"configDigest": config.digest},
    )
    retained_text = texts[retained_candidate_id]
    arcs: list[LatticeArc] = []
    for text, candidate_ids in sorted(
        groups.items(),
        key=lambda row: (row[0] != retained_text, row[0]),
    ):
        support = tuple(sorted(candidate_ids))
        local_posterior = sum(posterior[candidate_id] for candidate_id in support)
        utilities: list[BoundedUtility] = []
        local_masses: dict[str, float] = {}
        if factor_weight > 0.0:
            utilities.append(
                _utility(
                    channel="first_pass",
                    value=_bounded_mass(local_posterior),
                    source="first-pass-local-posterior-v1",
                    profile_digest=posterior_profile,
                    payload={
                        "spanId": span_id,
                        "surface": text,
                        "support": support,
                        "candidatePosteriors": tuple(
                            (candidate_id, posterior[candidate_id]) for candidate_id in support
                        ),
                    },
                    factor_weight=factor_weight,
                )
            )
            for channel, (field_name, values) in distributions.items():
                local_mass = sum(values[candidate_id] for candidate_id in support)
                local_masses[channel] = local_mass
                profile = _profile_digest(
                    f"candidate-{field_name}-local-surface-marginal",
                    {
                        "configDigest": config.digest,
                        "sourceField": field_name,
                        "channel": channel,
                    },
                )
                utilities.append(
                    _utility(
                        channel=channel,
                        value=_bounded_mass(local_mass),
                        source=f"candidate-{field_name}-local-softmax-v1",
                        profile_digest=profile,
                        payload={
                            "spanId": span_id,
                            "surface": text,
                            "support": support,
                            "candidateMass": tuple(
                                (candidate_id, values[candidate_id])
                                for candidate_id in support
                            ),
                        },
                        factor_weight=factor_weight,
                    )
                )
        supporting_candidates = tuple(candidates_by_id[candidate_id] for candidate_id in support)
        arc_digest = sha256_json({"spanId": span_id, "text": text, "support": support})[:20]
        arcs.append(
            LatticeArc(
                arc_id=f"{span_id}:surface:{arc_digest}",
                span_id=span_id,
                text=text,
                origin="first-pass",
                utilities=tuple(utilities),
                observed_eligible=True,
                pronunciation_key=_pronunciation_key(supporting_candidates),
                source_candidate_ids=support,
                source_audio_sha256=source_audio_sha256,
                is_epsilon=not text,
                metadata={
                    "localPosteriorMass": local_posterior,
                    "localEvidenceMasses": local_masses,
                    "semanticKinds": semantic_kinds,
                    "semanticCriticality": semantic_criticality,
                    "posteriorAmbiguity": ambiguity,
                    "factorWeight": factor_weight,
                    "retainedSurface": text == retained_text,
                },
            )
        )
    return tuple(arcs)


def _merge_proposals(
    span: DeliberationSpan,
    proposals: Sequence[VerifiedSpanProposal],
    *,
    source_audio_sha256: str,
) -> DeliberationSpan:
    if not proposals:
        return span
    factor_weight = float(span.metadata.get("factorWeight", 0.0))
    if factor_weight <= 0.0:
        raise ValueError("verified proposals may only target an active deliberation span")
    arcs = list(span.arcs)
    for proposal in proposals:
        if proposal.source_audio_sha256 != source_audio_sha256:
            raise ValueError("verified proposal is bound to different source audio")
        bound_utilities = tuple(
            utility.with_factor_weight(factor_weight) for utility in proposal.utilities
        )
        matching_index = next(
            (index for index, arc in enumerate(arcs) if arc.text == proposal.text),
            None,
        )
        if matching_index is None:
            arcs.append(
                LatticeArc(
                    arc_id=f"{span.span_id}:proposal:{proposal.proposal_id}",
                    span_id=span.span_id,
                    text=proposal.text,
                    origin=proposal.origin,
                    utilities=bound_utilities,
                    observed_eligible=proposal.observed_eligible,
                    pronunciation_key=proposal.pronunciation_key,
                    source_candidate_ids=proposal.source_candidate_ids,
                    source_audio_sha256=proposal.source_audio_sha256,
                    is_epsilon=not proposal.text,
                    metadata={
                        **proposal.metadata,
                        "proposalDigest": proposal.digest,
                        "factorWeight": factor_weight,
                    },
                )
            )
            continue
        current = arcs[matching_index]
        merged = {utility.channel: utility for utility in current.utilities}
        for utility in bound_utilities:
            existing = merged.get(utility.channel)
            if existing is not None and existing.digest != utility.digest:
                raise ValueError(
                    f"proposal {proposal.proposal_id!r} conflicts on utility channel "
                    f"{utility.channel!r}"
                )
            merged[utility.channel] = utility
        source_ids = tuple(
            dict.fromkeys((*current.source_candidate_ids, *proposal.source_candidate_ids))
        )
        metadata = dict(current.metadata)
        metadata["proposalDigests"] = tuple(
            dict.fromkeys((*metadata.get("proposalDigests", ()), proposal.digest))
        )
        arcs[matching_index] = replace(
            current,
            utilities=tuple(merged.values()),
            pronunciation_key=current.pronunciation_key or proposal.pronunciation_key,
            source_candidate_ids=source_ids,
            source_audio_sha256=proposal.source_audio_sha256,
            metadata=metadata,
        )
    return replace(span, arcs=tuple(arcs))


def _transition_utilities(
    spans: Sequence[DeliberationSpan],
    posterior: Mapping[str, float],
    *,
    config: SemanticDeliberationConfig,
) -> tuple[TransitionUtility, ...]:
    boundary_count = max(1, len(spans) - 1)
    factor_weight = 1.0 / boundary_count
    profile = _profile_digest(
        "first-pass-adjacent-surface-pmi",
        {
            "temperature": config.transition_temperature,
            "epsilon": config.transition_epsilon,
            "configDigest": config.digest,
            "boundaryFactorWeight": factor_weight,
        },
    )
    rows: list[TransitionUtility] = []
    for left_span, right_span in zip(spans, spans[1:], strict=False):
        for left in left_span.arcs:
            if left.origin != "first-pass":
                continue
            left_ids = set(left.source_candidate_ids)
            if not left_ids:
                continue
            left_mass = sum(posterior[candidate_id] for candidate_id in left_ids)
            for right in right_span.arcs:
                if right.origin != "first-pass":
                    continue
                right_ids = set(right.source_candidate_ids)
                if not right_ids:
                    continue
                right_mass = sum(posterior[candidate_id] for candidate_id in right_ids)
                joint_ids = tuple(sorted(left_ids.intersection(right_ids)))
                joint = sum(posterior[candidate_id] for candidate_id in joint_ids)
                epsilon = config.transition_epsilon
                pmi = math.log((joint + epsilon) / (left_mass * right_mass + epsilon))
                value = math.tanh(pmi / config.transition_temperature)
                rows.append(
                    TransitionUtility(
                        left_arc_id=left.arc_id,
                        right_arc_id=right.arc_id,
                        utility=_utility(
                            channel="transition",
                            value=value,
                            source="first-pass-adjacent-pmi-v1",
                            profile_digest=profile,
                            payload={
                                "leftArcId": left.arc_id,
                                "rightArcId": right.arc_id,
                                "leftMass": left_mass,
                                "rightMass": right_mass,
                                "jointMass": joint,
                                "jointCandidateIds": joint_ids,
                            },
                            factor_weight=factor_weight,
                        ),
                    )
                )
    return tuple(rows)


def _span_plans(
    candidates: Sequence[CandidateEvidence],
    pivot: CandidateEvidence,
    posterior: Mapping[str, float],
    *,
    segment_start_ms: int,
    segment_end_ms: int,
    pivot_timeline: Sequence[tuple[int, int, int, int]],
) -> tuple[_SpanPlan, ...]:
    intervals, alignments = _alignments(pivot, candidates)
    timed_intervals, timing_source = _timeline_time_boundaries(
        intervals,
        pivot_length=len(pivot.text),
        start_ms=segment_start_ms,
        end_ms=segment_end_ms,
        pivot_timeline=pivot_timeline,
    )
    output: list[_SpanPlan] = []
    for index, ((unit_start, unit_end), (start_ms, end_ms)) in enumerate(
        zip(intervals, timed_intervals, strict=True)
    ):
        texts = tuple(
            (
                candidate.candidate_id,
                "".join(
                    alignments[candidate.candidate_id].project_interval(
                        unit_start,
                        unit_end,
                        pivot_length=len(pivot.text),
                    )
                ),
            )
            for candidate in candidates
        )
        alternatives = tuple(dict.fromkeys(text for _, text in texts))
        retained_text = dict(texts)[pivot.candidate_id]
        kinds = tuple(
            sorted(
                _classify(
                    tuple(retained_text),
                    tuple(tuple(alternative) for alternative in alternatives),
                )
            )
        )
        local_masses = [
            sum(
                posterior[candidate_id]
                for candidate_id, text in texts
                if text == alternative
            )
            for alternative in alternatives
        ]
        output.append(
            _SpanPlan(
                index=index,
                unit_start=unit_start,
                unit_end=unit_end,
                start_ms=start_ms,
                end_ms=end_ms,
                timing_source=timing_source,
                texts=texts,
                semantic_kinds=kinds,
                semantic_criticality=_criticality(kinds),
                posterior_ambiguity=_local_ambiguity(local_masses),
            )
        )
    return tuple(output)


def build_semantic_deliberation_lattice(
    candidates: Sequence[CandidateEvidence],
    *,
    posterior: Mapping[str, float] | None,
    pivot_candidate_id: str,
    document_id: str,
    source_audio_sha256: str,
    segment_start_ms: int,
    segment_end_ms: int,
    pivot_timeline: Sequence[tuple[int, int, int, int]] = (),
    proposals: Mapping[str, Sequence[VerifiedSpanProposal]] | None = None,
    config: SemanticDeliberationConfig | None = None,
) -> SemanticDeliberationBuild:
    """Build an exact, score-domain-safe confusion network from whole ASR hypotheses."""

    config = config or SemanticDeliberationConfig()
    if not candidates:
        raise ValueError("semantic deliberation requires at least one candidate")
    if not _is_sha256(source_audio_sha256):
        raise ValueError("source_audio_sha256 must be a SHA-256 value")
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    try:
        pivot = candidates_by_id[pivot_candidate_id]
    except KeyError as exc:
        raise ValueError("pivot_candidate_id is absent from candidates") from exc
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    normalized_posterior = _normalize_distribution(candidate_ids, posterior)
    distributions = _candidate_relative_distributions(
        candidates,
        acoustic_temperature=config.acoustic_temperature,
        stream_temperature=config.stream_temperature,
    )
    plans = _span_plans(
        candidates,
        pivot,
        normalized_posterior,
        segment_start_ms=segment_start_ms,
        segment_end_ms=segment_end_ms,
        pivot_timeline=pivot_timeline,
    )
    factor_weights = _factor_weights(
        plans,
        include_consensus=config.include_consensus_utilities,
    )

    projected_texts: dict[str, list[tuple[str, str]]] = {
        candidate_id: [] for candidate_id in candidate_ids
    }
    spans: list[DeliberationSpan] = []
    proposal_digests: list[str] = []
    expected_span_ids = {f"{document_id}:span:{plan.index:04d}" for plan in plans}
    unknown_proposal_spans = set(proposals or {}) - expected_span_ids
    if unknown_proposal_spans:
        raise ValueError(f"proposals reference unknown spans: {sorted(unknown_proposal_spans)}")
    for plan in plans:
        span_id = f"{document_id}:span:{plan.index:04d}"
        texts = plan.text_map
        for candidate_id, text in plan.texts:
            projected_texts[candidate_id].append((span_id, text))
        arcs = _base_arc_groups(
            span_id,
            texts,
            candidates_by_id,
            normalized_posterior,
            distributions,
            retained_candidate_id=pivot_candidate_id,
            source_audio_sha256=source_audio_sha256,
            semantic_kinds=plan.semantic_kinds,
            semantic_criticality=plan.semantic_criticality,
            ambiguity=plan.posterior_ambiguity,
            factor_weight=factor_weights[plan.index],
            config=config,
        )
        retained_text = texts[pivot_candidate_id]
        retained_arc = next(
            arc
            for arc in arcs
            if pivot_candidate_id in arc.source_candidate_ids and arc.text == retained_text
        )
        span = DeliberationSpan(
            span_id=span_id,
            index=plan.index,
            start_ms=plan.start_ms,
            end_ms=plan.end_ms,
            arcs=arcs,
            retained_arc_id=retained_arc.arc_id,
            metadata={
                "pivotUnitStart": plan.unit_start,
                "pivotUnitEnd": plan.unit_end,
                "timingSource": plan.timing_source,
                "semanticKinds": plan.semantic_kinds,
                "semanticCriticality": plan.semantic_criticality,
                "posteriorAmbiguity": plan.posterior_ambiguity,
                "factorWeight": factor_weights[plan.index],
                "isContradiction": plan.is_contradiction,
            },
        )
        span_proposals = tuple((proposals or {}).get(span_id, ()))
        proposal_digests.extend(proposal.digest for proposal in span_proposals)
        spans.append(
            _merge_proposals(
                span,
                span_proposals,
                source_audio_sha256=source_audio_sha256,
            )
        )

    projections = tuple(
        ProjectedCandidate(
            candidate_id=candidate_id,
            source_text_sha256=hashlib.sha256(
                candidates_by_id[candidate_id].text.encode("utf-8")
            ).hexdigest(),
            span_texts=tuple(projected_texts[candidate_id]),
        )
        for candidate_id in candidate_ids
    )
    source_paths: list[SourcePath] = []
    for projection in projections:
        arc_ids = []
        for span, (_span_id, text) in zip(spans, projection.span_texts, strict=True):
            arc = next(
                arc
                for arc in span.arcs
                if projection.candidate_id in arc.source_candidate_ids and arc.text == text
            )
            arc_ids.append(arc.arc_id)
        source_paths.append(
            SourcePath(
                candidate_id=projection.candidate_id,
                arc_ids=tuple(arc_ids),
                text_sha256=projection.source_text_sha256,
                posterior=normalized_posterior[projection.candidate_id],
                metadata={"projection": "exact-surface-v2"},
            )
        )
    lattice = DeliberationLattice(
        document_id=document_id,
        source_audio_sha256=source_audio_sha256,
        spans=tuple(spans),
        transitions=_transition_utilities(spans, normalized_posterior, config=config),
        source_paths=tuple(source_paths),
        metadata={
            "pivotCandidateId": pivot_candidate_id,
            "projection": "exact-surface-v2",
            "configDigest": config.digest,
            "candidatePosteriorDigest": sha256_json(normalized_posterior),
            "proposalDigests": tuple(sorted(proposal_digests)),
            "localFactorWeightSum": sum(factor_weights.values()),
            "evidenceChannels": tuple(sorted(distributions)),
        },
    )
    return SemanticDeliberationBuild(
        lattice=lattice,
        projections=projections,
        candidate_posteriors=tuple(sorted(normalized_posterior.items())),
        pivot_candidate_id=pivot_candidate_id,
        config_digest=config.digest,
        proposal_digests=tuple(sorted(proposal_digests)),
    )


def path_source_candidate_ids(path: Sequence[LatticeArc]) -> tuple[str, ...]:
    """Return first-pass candidates that support every arc of an exact selected path."""

    if not path:
        raise ValueError("path must not be empty")
    support: set[str] | None = None
    for arc in path:
        current = set(arc.source_candidate_ids)
        if not current:
            return ()
        support = current if support is None else support.intersection(current)
        if not support:
            return ()
    return tuple(sorted(support or ()))


def path_is_recombined(path: Sequence[LatticeArc]) -> bool:
    return not path_source_candidate_ids(path)
