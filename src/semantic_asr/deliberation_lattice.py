"""Ordered confusion-network types for multi-level ASR deliberation."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .contracts import sha256_json
from .deliberation_evidence import (
    GENERATED_ORIGINS,
    INDEPENDENT_AUDIO_CHANNELS,
    ArcOrigin,
    BoundedUtility,
    UtilityChannel,
    _is_sha256,
    _strict_float,
)


@dataclass(frozen=True, slots=True)
class LatticeArc:
    arc_id: str
    span_id: str
    text: str
    origin: ArcOrigin
    utilities: tuple[BoundedUtility, ...]
    observed_eligible: bool = True
    pronunciation_key: str | None = None
    source_candidate_ids: tuple[str, ...] = ()
    source_audio_sha256: str | None = None
    is_epsilon: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arc_id or not self.span_id:
            raise ValueError("lattice arc requires arc_id and span_id")
        if self.is_epsilon:
            if self.text:
                raise ValueError("epsilon lattice arcs must have empty text")
        elif not self.text:
            raise ValueError("empty lattice arc text must be explicitly marked epsilon")
        if self.origin not in {
            "first-pass",
            "phonetic-proposal",
            "context-proposal",
            "guarded-generation",
            "human",
        }:
            raise ValueError("unknown lattice arc origin")
        channels = [utility.channel for utility in self.utilities]
        if len(channels) != len(set(channels)):
            raise ValueError("a lattice arc may contain at most one utility per channel")
        if self.pronunciation_key is not None and not self.pronunciation_key:
            raise ValueError("pronunciation_key must be non-empty when supplied")
        if self.source_audio_sha256 is not None and not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
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
    def utility_map(self) -> dict[UtilityChannel, BoundedUtility]:
        return {utility.channel: utility for utility in self.utilities}

    @property
    def independent_audio_channels(self) -> frozenset[str]:
        return frozenset(
            utility.channel
            for utility in self.utilities
            if utility.factor_weight > 0.0
        ).intersection(INDEPENDENT_AUDIO_CHANNELS)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "arcId": self.arc_id,
                "spanId": self.span_id,
                "text": self.text,
                "origin": self.origin,
                "utilities": [utility.digest for utility in self.utilities],
                "observedEligible": self.observed_eligible,
                "pronunciationKey": self.pronunciation_key,
                "sourceCandidateIds": self.source_candidate_ids,
                "sourceAudioSha256": self.source_audio_sha256,
                "isEpsilon": self.is_epsilon,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class DeliberationSpan:
    span_id: str
    index: int
    start_ms: int
    end_ms: int
    arcs: tuple[LatticeArc, ...]
    retained_arc_id: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.span_id or self.index < 0:
            raise ValueError("deliberation span requires a non-negative index and ID")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("deliberation span requires 0 <= start_ms < end_ms")
        if not self.arcs:
            raise ValueError("deliberation span requires at least one arc")
        if len({arc.arc_id for arc in self.arcs}) != len(self.arcs):
            raise ValueError("lattice arc IDs must be unique within a span")
        if any(arc.span_id != self.span_id for arc in self.arcs):
            raise ValueError("every arc must be bound to its deliberation span")
        try:
            retained = self.arc(self.retained_arc_id)
        except KeyError as exc:
            raise ValueError("retained_arc_id is absent from the span") from exc
        if retained.origin != "first-pass":
            raise ValueError("the retained arc must be a first-pass ASR arc")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def arc(self, arc_id: str) -> LatticeArc:
        for arc in self.arcs:
            if arc.arc_id == arc_id:
                return arc
        raise KeyError(arc_id)

    @property
    def retained_arc(self) -> LatticeArc:
        return self.arc(self.retained_arc_id)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "spanId": self.span_id,
                "index": self.index,
                "startMs": self.start_ms,
                "endMs": self.end_ms,
                "retainedArcId": self.retained_arc_id,
                "arcDigests": [arc.digest for arc in self.arcs],
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class TransitionUtility:
    left_arc_id: str
    right_arc_id: str
    utility: BoundedUtility

    def __post_init__(self) -> None:
        if not self.left_arc_id or not self.right_arc_id:
            raise ValueError("transition arc IDs are required")
        if self.utility.channel != "transition":
            raise ValueError("transition evidence must use the transition utility channel")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "leftArcId": self.left_arc_id,
                "rightArcId": self.right_arc_id,
                "utilityDigest": self.utility.digest,
            }
        )


@dataclass(frozen=True, slots=True)
class SourcePath:
    """One exact first-pass hypothesis projected through every lattice span."""

    candidate_id: str
    arc_ids: tuple[str, ...]
    text_sha256: str
    posterior: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.arc_ids:
            raise ValueError("source path requires candidate_id and arc_ids")
        if len(self.arc_ids) != len(set(self.arc_ids)):
            raise ValueError("source path arc IDs must be unique")
        if not _is_sha256(self.text_sha256):
            raise ValueError("source path text_sha256 must be a SHA-256 value")
        posterior = _strict_float(self.posterior, name="source path posterior")
        if not 0.0 <= posterior <= 1.0:
            raise ValueError("source path posterior must be in [0, 1]")
        object.__setattr__(self, "posterior", posterior)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "candidateId": self.candidate_id,
                "arcIds": self.arc_ids,
                "textSha256": self.text_sha256,
                "posterior": self.posterior,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class DeliberationLattice:
    document_id: str
    source_audio_sha256: str
    spans: tuple[DeliberationSpan, ...]
    transitions: tuple[TransitionUtility, ...] = ()
    source_paths: tuple[SourcePath, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: str = "2"

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id is required")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        if not self.spans:
            raise ValueError("deliberation lattice requires at least one span")
        if tuple(span.index for span in self.spans) != tuple(range(len(self.spans))):
            raise ValueError("deliberation span indexes must be contiguous from zero")
        previous_end = -1
        all_arcs: dict[str, tuple[int, LatticeArc]] = {}
        for span in self.spans:
            if span.start_ms < previous_end:
                raise ValueError("deliberation spans must be ordered and non-overlapping")
            previous_end = span.end_ms
            for arc in span.arcs:
                if arc.arc_id in all_arcs:
                    raise ValueError("lattice arc IDs must be globally unique")
                if (
                    arc.source_audio_sha256 is not None
                    and arc.source_audio_sha256 != self.source_audio_sha256
                ):
                    raise ValueError("lattice arc is bound to a different source audio")
                if (
                    arc.origin in GENERATED_ORIGINS
                    and arc.observed_eligible
                    and arc.independent_audio_channels
                    and arc.source_audio_sha256 != self.source_audio_sha256
                ):
                    raise ValueError(
                        "acoustically verified generated arcs must bind the source-audio SHA-256"
                    )
                all_arcs[arc.arc_id] = (span.index, arc)

        seen_transitions: set[tuple[str, str]] = set()
        for transition in self.transitions:
            key = (transition.left_arc_id, transition.right_arc_id)
            if key in seen_transitions:
                raise ValueError("duplicate transition utility")
            seen_transitions.add(key)
            try:
                left_index = all_arcs[transition.left_arc_id][0]
                right_index = all_arcs[transition.right_arc_id][0]
            except KeyError as exc:
                raise ValueError("transition references an unknown arc") from exc
            if right_index != left_index + 1:
                raise ValueError("transition utilities may only connect adjacent spans")

        if len({path.candidate_id for path in self.source_paths}) != len(self.source_paths):
            raise ValueError("source path candidate IDs must be unique")
        for path in self.source_paths:
            if len(path.arc_ids) != len(self.spans):
                raise ValueError("every source path must select one arc from every span")
            text: list[str] = []
            for span_index, arc_id in enumerate(path.arc_ids):
                try:
                    actual_index, arc = all_arcs[arc_id]
                except KeyError as exc:
                    raise ValueError("source path references an unknown arc") from exc
                if actual_index != span_index:
                    raise ValueError("source path arc order does not match lattice span order")
                if path.candidate_id not in arc.source_candidate_ids:
                    raise ValueError("source path uses an arc that does not support its candidate")
                text.append(arc.text)
            digest = hashlib.sha256("".join(text).encode("utf-8")).hexdigest()
            if digest != path.text_sha256:
                raise ValueError("source path does not reconstruct its exact candidate text")
        if self.source_paths:
            total = sum(path.posterior for path in self.source_paths)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("source path posteriors must sum to one")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def arc(self, arc_id: str) -> LatticeArc:
        for span in self.spans:
            for arc in span.arcs:
                if arc.arc_id == arc_id:
                    return arc
        raise KeyError(arc_id)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "documentId": self.document_id,
                "sourceAudioSha256": self.source_audio_sha256,
                "spanDigests": [span.digest for span in self.spans],
                "transitionDigests": [row.digest for row in self.transitions],
                "sourcePathDigests": [path.digest for path in self.source_paths],
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentContext:
    left_context: str = ""
    right_context: str = ""
    topic_summary: str = ""
    entity_ids: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_ids", tuple(dict.fromkeys(self.entity_ids)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "leftContext": self.left_context,
                "rightContext": self.right_context,
                "topicSummary": self.topic_summary,
                "entityIds": self.entity_ids,
                "metadata": self.metadata,
            }
        )


def path_digest(path: Sequence[LatticeArc]) -> str:
    if not path:
        raise ValueError("a deliberation path must not be empty")
    return sha256_json(
        [
            {
                "arcId": arc.arc_id,
                "spanId": arc.span_id,
                "text": arc.text,
                "arcDigest": arc.digest,
            }
            for arc in path
        ]
    )
