"""Ordered confusion-network types for multi-level ASR deliberation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .contracts import sha256_json
from .deliberation_evidence import (
    INDEPENDENT_AUDIO_CHANNELS,
    ArcOrigin,
    BoundedUtility,
    UtilityChannel,
    _is_sha256,
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
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arc_id or not self.span_id or not self.text:
            raise ValueError("lattice arc requires arc_id, span_id and text")
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
        return frozenset(self.utility_map).intersection(INDEPENDENT_AUDIO_CHANNELS)

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

    def arc(self, arc_id: str) -> LatticeArc:
        for arc in self.arcs:
            if arc.arc_id == arc_id:
                return arc
        raise KeyError(arc_id)

    @property
    def retained_arc(self) -> LatticeArc:
        return self.arc(self.retained_arc_id)


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
class DeliberationLattice:
    document_id: str
    source_audio_sha256: str
    spans: tuple[DeliberationSpan, ...]
    transitions: tuple[TransitionUtility, ...] = ()
    schema_version: str = "1"

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
        all_arcs: dict[str, int] = {}
        for span in self.spans:
            if span.start_ms < previous_end:
                raise ValueError("deliberation spans must be ordered and non-overlapping")
            previous_end = span.end_ms
            for arc in span.arcs:
                if arc.arc_id in all_arcs:
                    raise ValueError("lattice arc IDs must be globally unique")
                all_arcs[arc.arc_id] = span.index
        seen_transitions: set[tuple[str, str]] = set()
        for transition in self.transitions:
            key = (transition.left_arc_id, transition.right_arc_id)
            if key in seen_transitions:
                raise ValueError("duplicate transition utility")
            seen_transitions.add(key)
            try:
                left_index = all_arcs[transition.left_arc_id]
                right_index = all_arcs[transition.right_arc_id]
            except KeyError as exc:
                raise ValueError("transition references an unknown arc") from exc
            if right_index != left_index + 1:
                raise ValueError("transition utilities may only connect adjacent spans")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "documentId": self.document_id,
                "sourceAudioSha256": self.source_audio_sha256,
                "spans": [
                    {
                        "spanId": span.span_id,
                        "index": span.index,
                        "startMs": span.start_ms,
                        "endMs": span.end_ms,
                        "retainedArcId": span.retained_arc_id,
                        "arcDigests": [arc.digest for arc in span.arcs],
                    }
                    for span in self.spans
                ],
                "transitionDigests": [row.digest for row in self.transitions],
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
