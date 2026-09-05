"""Joint document-level decoding over per-window Semantic ASR lattices.

The existing long-form deliberation path scores each window independently against frozen
neighbouring text. This module keeps several acoustically admissible local paths per window and
selects one sequence for the complete recording. Overlapping audio is attributed once, overlap
text consistency is a bounded factor, and a complete-document scorer is rank-only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from .audio import require_integer
from .candidate_pool import lenient_surface_key
from .contracts import sha256_json
from .deliberation_evidence import GENERATED_ORIGINS, _is_sha256, _strict_float
from .deliberation_lattice import DocumentContext, LatticeArc, path_digest
from .global_deliberation import (
    DeliberationPolicy,
    GlobalDeliberationDecision,
    PathHypothesis,
    SpanResolution,
    decode_global_lattice,
)
from .global_scorer import GlobalPathScore, GlobalSequenceScorer
from .japanese import join_timed_fragments
from .longform import (
    LongformResult,
    LongformSegment,
    SemanticASRTranscriber,
    join_segment_text,
    sha256_file,
)
from .longform_deliberation import (
    DeliberatedLongformResult,
    DeliberatedLongformSegment,
    SegmentDeliberationTrace,
    SpanProposalProvider,
    _applied_segment,
    _pivot_timeline,
    _posterior,
    _unchanged_segment,
)
from .semantic_deliberation import (
    SemanticDeliberationBuild,
    SemanticDeliberationConfig,
    build_semantic_deliberation_lattice,
    path_is_recombined,
    path_source_candidate_ids,
)

ContextArm = Literal[
    "none",
    "declared-only",
    "left-only",
    "bidirectional-offline",
    "shuffled-context",
]


@dataclass(frozen=True, slots=True)
class FrozenWindowContext:
    """One leakage-auditable context arm for a target first-pass window."""

    target_window_index: int
    arm: ContextArm
    context: DocumentContext
    source_window_indices: tuple[int, ...]
    first_pass_evidence_sha256: str
    shuffle_seed_digest: str | None = None

    def __post_init__(self) -> None:
        require_integer(self.target_window_index, name="target_window_index")
        if self.arm not in {
            "none",
            "declared-only",
            "left-only",
            "bidirectional-offline",
            "shuffled-context",
        }:
            raise ValueError("unknown document context arm")
        if not _is_sha256(self.first_pass_evidence_sha256):
            raise ValueError("first_pass_evidence_sha256 must be a SHA-256 value")
        if self.target_window_index in self.source_window_indices:
            raise ValueError("target window must not be injected into its own context")
        if len(set(self.source_window_indices)) != len(self.source_window_indices):
            raise ValueError("context source window indices must be unique")
        if self.arm in {"none", "declared-only"} and self.source_window_indices:
            raise ValueError(f"{self.arm} cannot contain first-pass source windows")
        if self.arm == "left-only" and any(
            index >= self.target_window_index for index in self.source_window_indices
        ):
            raise ValueError("left-only context cannot contain current or future windows")
        if self.arm == "shuffled-context":
            if self.shuffle_seed_digest is None or not _is_sha256(self.shuffle_seed_digest):
                raise ValueError("shuffled context requires a frozen seed digest")
        elif self.shuffle_seed_digest is not None:
            raise ValueError("shuffle_seed_digest is only valid for shuffled context")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "targetWindowIndex": self.target_window_index,
                "arm": self.arm,
                "contextDigest": self.context.digest,
                "sourceWindowIndices": self.source_window_indices,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "shuffleSeedDigest": self.shuffle_seed_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentBeamConfig:
    """Bounded document search and application policy."""

    enabled: bool = True
    local_paths_per_window: int = 6
    beam_size: int = 96
    global_rescore_paths: int = 32
    overlap_consistency_weight: float = 0.65
    maximum_overlap_similarity_regression: float = 0.35
    minimum_overlap_characters: int = 2
    change_penalty: float = 0.02
    generated_penalty: float = 0.08
    global_context_weight: float = 1.0
    maximum_document_audio_regression: float = 0.10
    maximum_changed_windows: int | None = None
    maximum_generated_windows: int = 2
    minimum_final_margin: float = 0.02
    proposal_context_arm: ContextArm = "bidirectional-offline"
    maximum_left_windows: int = 4
    maximum_right_windows: int = 4
    maximum_context_characters: int = 12_000
    require_sequence_scorer: bool = True
    apply_provisional: bool = False
    fail_closed_to_first_pass: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "local_paths_per_window",
            "beam_size",
            "global_rescore_paths",
            "minimum_overlap_characters",
            "maximum_generated_windows",
            "maximum_left_windows",
            "maximum_right_windows",
            "maximum_context_characters",
        ):
            require_integer(getattr(self, name), name=name)
        if self.local_paths_per_window < 1 or self.beam_size < 1 or self.global_rescore_paths < 1:
            raise ValueError("document beam sizes must be positive")
        if self.minimum_overlap_characters < 1:
            raise ValueError("minimum_overlap_characters must be positive")
        if self.maximum_changed_windows is not None:
            require_integer(self.maximum_changed_windows, name="maximum_changed_windows")
        for name in (
            "overlap_consistency_weight",
            "maximum_overlap_similarity_regression",
            "change_penalty",
            "generated_penalty",
            "global_context_weight",
            "maximum_document_audio_regression",
            "minimum_final_margin",
        ):
            value = _strict_float(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "enabled",
            "require_sequence_scorer",
            "apply_provisional",
            "fail_closed_to_first_pass",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.proposal_context_arm not in {
            "none",
            "declared-only",
            "left-only",
            "bidirectional-offline",
            "shuffled-context",
        }:
            raise ValueError("unknown proposal_context_arm")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class OverlapCompatibility:
    left_window_index: int
    right_window_index: int
    overlap_start_ms: int
    overlap_end_ms: int
    left_text_sha256: str
    right_text_sha256: str
    similarity: float | None
    retained_similarity: float | None
    similarity_delta: float
    compared_characters: int
    compatible: bool

    def __post_init__(self) -> None:
        require_integer(self.left_window_index, name="left_window_index")
        require_integer(self.right_window_index, name="right_window_index")
        require_integer(self.overlap_start_ms, name="overlap_start_ms")
        require_integer(self.overlap_end_ms, name="overlap_end_ms")
        if self.overlap_end_ms <= self.overlap_start_ms:
            raise ValueError("overlap receipt requires a positive time intersection")
        if not _is_sha256(self.left_text_sha256) or not _is_sha256(self.right_text_sha256):
            raise ValueError("overlap text identities must be SHA-256 values")
        for name in ("similarity", "retained_similarity"):
            value = getattr(self, name)
            if value is not None:
                numeric = _strict_float(value, name=name)
                if not 0.0 <= numeric <= 1.0:
                    raise ValueError(f"{name} must be in [0, 1]")
                object.__setattr__(self, name, numeric)
        delta = _strict_float(self.similarity_delta, name="similarity_delta")
        if not -1.0 <= delta <= 1.0:
            raise ValueError("similarity_delta must be in [-1, 1]")
        require_integer(self.compared_characters, name="compared_characters")
        if not isinstance(self.compatible, bool):
            raise TypeError("compatible must be a boolean")
        object.__setattr__(self, "similarity_delta", delta)

    @property
    def duration_ms(self) -> int:
        return self.overlap_end_ms - self.overlap_start_ms

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class WindowPathOption:
    """One acoustically admissible complete path through a window lattice."""

    window_index: int
    start_ms: int
    end_ms: int
    build: SemanticDeliberationBuild = field(repr=False)
    path: PathHypothesis = field(repr=False)
    retained_path_digest: str
    coverage_ms: float
    local_score_delta: float
    audio_regression: float
    changed: bool
    generated: bool
    recombined: bool
    exact_source_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_integer(self.window_index, name="window_index")
        require_integer(self.start_ms, name="start_ms")
        require_integer(self.end_ms, name="end_ms", minimum=1)
        if self.end_ms <= self.start_ms:
            raise ValueError("window option requires a positive time range")
        if not _is_sha256(self.retained_path_digest):
            raise ValueError("retained_path_digest must be a SHA-256 value")
        for name in ("coverage_ms", "local_score_delta", "audio_regression"):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.coverage_ms <= 0.0:
            raise ValueError("coverage_ms must be positive")
        for name in ("changed", "generated", "recombined"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.path.digest == self.retained_path_digest and self.changed:
            raise ValueError("retained path cannot be marked changed")
        if self.path.digest != self.retained_path_digest and not self.changed:
            raise ValueError("alternative path must be marked changed")
        if any(
            arc.source_audio_sha256 is not None
            and arc.source_audio_sha256 != self.build.lattice.source_audio_sha256
            for arc in self.path.arcs
        ):
            raise ValueError("window path contains arcs from a different recording")

    @property
    def text(self) -> str:
        return self.path.text

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "windowIndex": self.window_index,
                "startMs": self.start_ms,
                "endMs": self.end_ms,
                "buildDigest": self.build.digest,
                "pathDigest": self.path.digest,
                "retainedPathDigest": self.retained_path_digest,
                "coverageMs": self.coverage_ms,
                "localScoreDelta": self.local_score_delta,
                "audioRegression": self.audio_regression,
                "changed": self.changed,
                "generated": self.generated,
                "recombined": self.recombined,
                "exactSourceCandidateIds": self.exact_source_candidate_ids,
            }
        )

    def overlap_text(self, start_ms: int, end_ms: int) -> str:
        if end_ms <= start_ms:
            raise ValueError("overlap_text requires a positive interval")
        fragments: list[str] = []
        for span, arc in zip(self.build.lattice.spans, self.path.arcs, strict=True):
            left = max(start_ms, span.start_ms)
            right = min(end_ms, span.end_ms)
            if left >= right or not arc.text:
                continue
            width = span.end_ms - span.start_ms
            start_index = math.floor(len(arc.text) * (left - span.start_ms) / width)
            end_index = math.ceil(len(arc.text) * (right - span.start_ms) / width)
            start_index = min(len(arc.text), max(0, start_index))
            end_index = min(len(arc.text), max(start_index, end_index))
            fragments.append(arc.text[start_index:end_index])
        return "".join(fragments)


@dataclass(frozen=True, slots=True)
class WindowPathSet:
    window_index: int
    segment: LongformSegment = field(repr=False)
    build: SemanticDeliberationBuild = field(repr=False)
    retained: WindowPathOption
    options: tuple[WindowPathOption, ...]
    proposal_context: FrozenWindowContext

    def __post_init__(self) -> None:
        require_integer(self.window_index, name="window_index")
        if self.segment.window.index != self.window_index:
            raise ValueError("window path set does not match the source segment")
        if not self.options:
            raise ValueError("window path set requires options")
        if len({option.digest for option in self.options}) != len(self.options):
            raise ValueError("window path options must be unique")
        if self.retained.digest not in {option.digest for option in self.options}:
            raise ValueError("retained option must be present")
        if any(option.window_index != self.window_index for option in self.options):
            raise ValueError("window path option belongs to another window")
        if self.proposal_context.target_window_index != self.window_index:
            raise ValueError("proposal context belongs to another target window")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "windowIndex": self.window_index,
                "firstPassEvidenceSha256": self.segment.observed.evidence_sha256,
                "buildDigest": self.build.digest,
                "retainedOptionDigest": self.retained.digest,
                "optionDigests": [option.digest for option in self.options],
                "proposalContextDigest": self.proposal_context.digest,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentPathHypothesis:
    options: tuple[WindowPathOption, ...]
    overlap_receipts: tuple[OverlapCompatibility, ...]
    base_score: float
    mean_audio_support: float
    context_score: float = 0.0
    final_score: float = 0.0
    scorer_source: str | None = None
    scorer_profile_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("document path requires at least one window option")
        if tuple(option.window_index for option in self.options) != tuple(range(len(self.options))):
            raise ValueError("document path window indices must be contiguous from zero")
        for name in ("base_score", "mean_audio_support", "context_score", "final_score"):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if not -1.0 <= self.mean_audio_support <= 1.0:
            raise ValueError("mean_audio_support must be in [-1, 1]")
        if not -1.0 <= self.context_score <= 1.0:
            raise ValueError("context_score must be in [-1, 1]")
        if (self.scorer_source is None) != (self.scorer_profile_digest is None):
            raise ValueError("scorer source and profile digest must be supplied together")
        if self.scorer_profile_digest is not None and not _is_sha256(
            self.scorer_profile_digest
        ):
            raise ValueError("scorer_profile_digest must be a SHA-256 value")
        expected_pairs = {
            (left.window_index, right.window_index)
            for left, right in zip(self.options, self.options[1:], strict=False)
            if right.start_ms < left.end_ms
        }
        actual_pairs = {
            (receipt.left_window_index, receipt.right_window_index)
            for receipt in self.overlap_receipts
        }
        if actual_pairs != expected_pairs:
            raise ValueError("document path overlap receipts do not match overlapping windows")

    @property
    def changed_window_count(self) -> int:
        return sum(option.changed for option in self.options)

    @property
    def generated_window_count(self) -> int:
        return sum(option.generated for option in self.options)

    @property
    def text(self) -> str:
        return join_timed_fragments(
            (option.start_ms, option.end_ms, option.text) for option in self.options
        )

    @property
    def digest(self) -> str:
        """Stable path identity independent of later score attachment."""

        return sha256_json(
            {
                "optionDigests": [option.digest for option in self.options],
                "overlapReceiptDigests": [receipt.digest for receipt in self.overlap_receipts],
                "text": self.text,
            }
        )

    @property
    def score_digest(self) -> str:
        return sha256_json(
            {
                "pathDigest": self.digest,
                "baseScore": self.base_score,
                "meanAudioSupport": self.mean_audio_support,
                "contextScore": self.context_score,
                "finalScore": self.final_score,
                "scorerSource": self.scorer_source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )

    def scorer_arc(self, *, source_audio_sha256: str, config_digest: str) -> LatticeArc:
        return LatticeArc(
            arc_id=f"document-path:{self.digest[:20]}",
            span_id=f"document:{source_audio_sha256[:20]}",
            text=self.text,
            origin="first-pass",
            utilities=(),
            observed_eligible=True,
            source_audio_sha256=source_audio_sha256,
            metadata={
                "documentPathDigest": self.digest,
                "windowOptionDigests": tuple(option.digest for option in self.options),
                "overlapReceiptDigests": tuple(
                    receipt.digest for receipt in self.overlap_receipts
                ),
                "configDigest": config_digest,
            },
        )


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
        if self.scorer_profile_digest is not None and not _is_sha256(
            self.scorer_profile_digest
        ):
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


def _bounded(value: str, limit: int, *, suffix: bool) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[-limit:] if suffix else value[:limit]


def _deterministic_shuffle(indices: Sequence[int], seed: str, target: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            indices,
            key=lambda index: sha256_json(
                {"seed": seed, "targetWindowIndex": target, "sourceWindowIndex": index}
            ),
        )
    )


def build_frozen_window_contexts(
    first_pass: LongformResult,
    *,
    arm: ContextArm,
    declared_context: DocumentContext | None = None,
    maximum_left_windows: int = 4,
    maximum_right_windows: int = 4,
    maximum_context_characters: int = 12_000,
    shuffle_seed: str = "semantic-asr-shuffled-context-v1",
) -> tuple[FrozenWindowContext, ...]:
    """Construct control-arm contexts without target-window or corrected-text leakage."""

    first_pass.verify()
    require_integer(maximum_left_windows, name="maximum_left_windows")
    require_integer(maximum_right_windows, name="maximum_right_windows")
    require_integer(maximum_context_characters, name="maximum_context_characters")
    if arm not in {
        "none",
        "declared-only",
        "left-only",
        "bidirectional-offline",
        "shuffled-context",
    }:
        raise ValueError("unknown document context arm")
    declared = declared_context or DocumentContext()
    texts = tuple(segment.observed.text for segment in first_pass.segments)
    seed_digest = sha256_json({"shuffleSeed": shuffle_seed}) if arm == "shuffled-context" else None
    output: list[FrozenWindowContext] = []
    for target in range(len(texts)):
        left_start = max(0, target - maximum_left_windows)
        right_end = min(len(texts), target + 1 + maximum_right_windows)
        left_indices = tuple(range(left_start, target))
        right_indices = tuple(range(target + 1, right_end))
        if arm in {"none", "declared-only"}:
            source_indices: tuple[int, ...] = ()
            left_rows: tuple[str, ...] = ()
            right_rows: tuple[str, ...] = ()
        elif arm == "left-only":
            source_indices = left_indices
            left_rows = tuple(texts[index] for index in source_indices)
            right_rows = ()
        elif arm == "bidirectional-offline":
            source_indices = (*left_indices, *right_indices)
            left_rows = tuple(texts[index] for index in left_indices)
            right_rows = tuple(texts[index] for index in right_indices)
        else:
            candidates = (*left_indices, *right_indices)
            source_indices = _deterministic_shuffle(candidates, shuffle_seed, target)
            midpoint = min(len(left_indices), len(source_indices))
            left_rows = tuple(texts[index] for index in source_indices[:midpoint])
            right_rows = tuple(texts[index] for index in source_indices[midpoint:])

        include_declared = arm != "none"
        left = "\n".join(
            (
                *((declared.left_context,) if include_declared and declared.left_context else ()),
                *left_rows,
            )
        )
        right = "\n".join(
            (
                *right_rows,
                *((declared.right_context,) if include_declared and declared.right_context else ()),
            )
        )
        half = maximum_context_characters // 2
        left = _bounded(left, half, suffix=True)
        right = _bounded(
            right,
            maximum_context_characters - len(left),
            suffix=False,
        )
        context = DocumentContext(
            left_context=left,
            right_context=right,
            topic_summary=declared.topic_summary if include_declared else "",
            entity_ids=declared.entity_ids if include_declared else (),
            metadata={
                **(declared.metadata if include_declared else {}),
                "contextArm": arm,
                "targetWindowIndex": target,
                "sourceWindowIndices": source_indices,
                "firstPassEvidenceSha256": first_pass.evidence_sha256,
                "offline": arm in {"bidirectional-offline", "shuffled-context"},
                "usesFutureFirstPass": any(index > target for index in source_indices),
                "shuffleSeedDigest": seed_digest,
            },
        )
        output.append(
            FrozenWindowContext(
                target_window_index=target,
                arm=arm,
                context=context,
                source_window_indices=source_indices,
                first_pass_evidence_sha256=first_pass.evidence_sha256,
                shuffle_seed_digest=seed_digest,
            )
        )
    return tuple(output)


def _coverage_attribution(segments: Sequence[LongformSegment]) -> tuple[float, ...]:
    boundaries = sorted(
        {value for segment in segments for value in (segment.window.start_ms, segment.window.end_ms)}
    )
    attributed = [0.0] * len(segments)
    for start, end in zip(boundaries, boundaries[1:], strict=False):
        if end <= start:
            continue
        active = [
            index
            for index, segment in enumerate(segments)
            if segment.window.start_ms < end and segment.window.end_ms > start
        ]
        if not active:
            continue
        share = (end - start) / len(active)
        for index in active:
            attributed[index] += share
    if any(value <= 0.0 for value in attributed):
        raise ValueError("every long-form window must receive positive audio coverage")
    return tuple(attributed)


def _option(
    segment: LongformSegment,
    build: SemanticDeliberationBuild,
    path: PathHypothesis,
    retained: PathHypothesis,
    coverage_ms: float,
) -> WindowPathOption:
    exact = path_source_candidate_ids(path.arcs)
    generated = any(arc.origin in GENERATED_ORIGINS for arc in path.arcs)
    return WindowPathOption(
        window_index=segment.window.index,
        start_ms=segment.window.start_ms,
        end_ms=segment.window.end_ms,
        build=build,
        path=path,
        retained_path_digest=retained.digest,
        coverage_ms=coverage_ms,
        local_score_delta=path.base_score - retained.base_score,
        audio_regression=retained.mean_audio_support - path.mean_audio_support,
        changed=path.digest != retained.digest,
        generated=generated,
        recombined=path_is_recombined(path.arcs),
        exact_source_candidate_ids=exact,
    )


def _build_window_set(
    first_pass: LongformResult,
    segment: LongformSegment,
    context: FrozenWindowContext,
    *,
    coverage_ms: float,
    config: DocumentBeamConfig,
    build_config: SemanticDeliberationConfig,
    local_policy: DeliberationPolicy,
    proposal_provider: SpanProposalProvider | None,
    audio_path: str | Path | None,
) -> WindowPathSet:
    build = build_semantic_deliberation_lattice(
        segment.observed.candidates,
        posterior=_posterior(segment),
        pivot_candidate_id=segment.observed.selected_candidate_id,
        document_id=f"{first_pass.source_audio_sha256[:16]}:window:{segment.window.index:04d}",
        source_audio_sha256=first_pass.source_audio_sha256,
        segment_start_ms=segment.window.start_ms,
        segment_end_ms=segment.window.end_ms,
        pivot_timeline=_pivot_timeline(segment),
        config=build_config,
    )
    if proposal_provider is not None:
        proposals = proposal_provider(
            audio_path=audio_path,
            segment_index=segment.window.index,
            segment=segment,
            build=build,
            context=context.context,
            source_audio_sha256=first_pass.source_audio_sha256,
        )
        if proposals:
            build = build_semantic_deliberation_lattice(
                segment.observed.candidates,
                posterior=_posterior(segment),
                pivot_candidate_id=segment.observed.selected_candidate_id,
                document_id=f"{first_pass.source_audio_sha256[:16]}:window:{segment.window.index:04d}",
                source_audio_sha256=first_pass.source_audio_sha256,
                segment_start_ms=segment.window.start_ms,
                segment_end_ms=segment.window.end_ms,
                pivot_timeline=_pivot_timeline(segment),
                proposals=proposals,
                config=build_config,
            )
    local = decode_global_lattice(
        build.lattice,
        policy=local_policy,
        context=DocumentContext(
            metadata={
                "mode": "document-beam-local-acoustic-filter",
                "firstPassEvidenceSha256": first_pass.evidence_sha256,
            }
        ),
        sequence_scorer=None,
    )
    retained_path = local.retained
    paths = list(local.alternatives[: config.local_paths_per_window])
    if retained_path.digest not in {path.digest for path in paths}:
        paths.append(retained_path)
    paths = sorted(
        {path.digest: path for path in paths}.values(),
        key=lambda path: (-path.base_score, path.digest),
    )
    options = tuple(_option(segment, build, path, retained_path, coverage_ms) for path in paths)
    retained_option = next(option for option in options if not option.changed)
    return WindowPathSet(
        window_index=segment.window.index,
        segment=segment,
        build=build,
        retained=retained_option,
        options=options,
        proposal_context=context,
    )


def _normalized_overlap(value: str) -> str:
    return lenient_surface_key(value)


def _overlap_receipt(
    left: WindowPathOption,
    right: WindowPathOption,
    retained_left: WindowPathOption,
    retained_right: WindowPathOption,
    *,
    config: DocumentBeamConfig,
) -> OverlapCompatibility | None:
    start = max(left.start_ms, right.start_ms)
    end = min(left.end_ms, right.end_ms)
    if end <= start:
        return None
    left_text = _normalized_overlap(left.overlap_text(start, end))
    right_text = _normalized_overlap(right.overlap_text(start, end))
    retained_left_text = _normalized_overlap(retained_left.overlap_text(start, end))
    retained_right_text = _normalized_overlap(retained_right.overlap_text(start, end))

    compared = min(len(left_text), len(right_text))
    retained_compared = min(len(retained_left_text), len(retained_right_text))
    similarity = (
        SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
        if compared >= config.minimum_overlap_characters
        else None
    )
    retained_similarity = (
        SequenceMatcher(
            None,
            retained_left_text,
            retained_right_text,
            autojunk=False,
        ).ratio()
        if retained_compared >= config.minimum_overlap_characters
        else None
    )
    delta = (
        0.0 if similarity is None or retained_similarity is None else similarity - retained_similarity
    )
    compatible = not (
        similarity is not None
        and retained_similarity is not None
        and retained_similarity - similarity > config.maximum_overlap_similarity_regression
    )
    return OverlapCompatibility(
        left_window_index=left.window_index,
        right_window_index=right.window_index,
        overlap_start_ms=start,
        overlap_end_ms=end,
        left_text_sha256=sha256_json({"text": left_text}),
        right_text_sha256=sha256_json({"text": right_text}),
        similarity=similarity,
        retained_similarity=retained_similarity,
        similarity_delta=delta,
        compared_characters=compared,
        compatible=compatible,
    )


def _mean_audio_support(options: Sequence[WindowPathOption]) -> float:
    rows = [
        (option.coverage_ms, option.path.mean_audio_support)
        for option in options
        if option.path.mean_audio_support > -1.0
    ]
    if not rows:
        return -1.0
    total = sum(weight for weight, _ in rows)
    return sum(weight * value for weight, value in rows) / total


def _document_hypothesis(state: _BeamState) -> DocumentPathHypothesis:
    return DocumentPathHypothesis(
        options=state.options,
        overlap_receipts=state.overlap_receipts,
        base_score=state.base_score,
        mean_audio_support=_mean_audio_support(state.options),
        final_score=state.base_score,
    )


def _base_document_paths(
    window_sets: Sequence[WindowPathSet],
    *,
    config: DocumentBeamConfig,
) -> tuple[DocumentPathHypothesis, ...]:
    total_coverage = sum(window.retained.coverage_ms for window in window_sets)
    retained = tuple(window.retained for window in window_sets)
    beam = [_BeamState((), (), 0.0, 0, 0)]
    for index, window in enumerate(window_sets):
        expanded: list[_BeamState] = []
        for state in beam:
            for option in window.options:
                changed = state.changed_windows + int(option.changed)
                generated = state.generated_windows + int(option.generated)
                if (
                    config.maximum_changed_windows is not None
                    and changed > config.maximum_changed_windows
                ):
                    continue
                if generated > config.maximum_generated_windows:
                    continue
                receipts = state.overlap_receipts
                overlap_delta = 0.0
                if state.options:
                    receipt = _overlap_receipt(
                        state.options[-1],
                        option,
                        retained[index - 1],
                        retained[index],
                        config=config,
                    )
                    if receipt is not None:
                        if not receipt.compatible:
                            continue
                        receipts = (*receipts, receipt)
                        overlap_delta = (
                            config.overlap_consistency_weight
                            * (receipt.duration_ms / total_coverage)
                            * receipt.similarity_delta
                        )
                score = state.base_score
                score += (option.coverage_ms / total_coverage) * option.local_score_delta
                score += overlap_delta
                if option.changed:
                    score -= config.change_penalty
                if option.generated:
                    score -= config.generated_penalty
                expanded.append(
                    _BeamState(
                        options=(*state.options, option),
                        overlap_receipts=receipts,
                        base_score=score,
                        changed_windows=changed,
                        generated_windows=generated,
                    )
                )
        if not expanded:
            raise ValueError(f"document beam has no compatible path at window {index}")
        expanded.sort(
            key=lambda row: (
                -row.base_score,
                row.changed_windows,
                row.generated_windows,
                tuple(option.digest for option in row.options),
            )
        )
        beam = expanded[: config.beam_size]

    paths = [_document_hypothesis(state) for state in beam]
    retained_path = _retained_hypothesis_with_config(window_sets, config=config)
    if retained_path.digest not in {path.digest for path in paths}:
        paths.append(retained_path)
    paths = sorted(
        {path.digest: path for path in paths}.values(),
        key=lambda path: (-path.base_score, path.changed_window_count, path.digest),
    )
    guarded = [
        path
        for path in paths
        if retained_path.mean_audio_support - path.mean_audio_support
        <= config.maximum_document_audio_regression
    ]
    if retained_path.digest not in {path.digest for path in guarded}:
        guarded.append(retained_path)
    return tuple(
        sorted(
            guarded,
            key=lambda path: (-path.base_score, path.changed_window_count, path.digest),
        )
    )


def _retained_hypothesis_with_config(
    window_sets: Sequence[WindowPathSet],
    *,
    config: DocumentBeamConfig,
) -> DocumentPathHypothesis:
    options = tuple(window.retained for window in window_sets)
    receipts = tuple(
        receipt
        for left, right in zip(window_sets, window_sets[1:], strict=False)
        if (
            receipt := _overlap_receipt(
                left.retained,
                right.retained,
                left.retained,
                right.retained,
                config=config,
            )
        )
        is not None
    )
    return DocumentPathHypothesis(
        options=options,
        overlap_receipts=receipts,
        base_score=0.0,
        mean_audio_support=_mean_audio_support(options),
        final_score=0.0,
    )


def _score_document_paths(
    paths: Sequence[DocumentPathHypothesis],
    retained_digest: str,
    *,
    first_pass: LongformResult,
    config: DocumentBeamConfig,
    sequence_scorer: GlobalSequenceScorer | None,
    declared_context: DocumentContext,
) -> tuple[tuple[DocumentPathHypothesis, ...], str | None, str | None, DocumentContext]:
    candidates = list(paths[: config.global_rescore_paths])
    retained = next(path for path in paths if path.digest == retained_digest)
    if retained.digest not in {path.digest for path in candidates}:
        candidates.append(retained)
    candidates = list({path.digest: path for path in candidates}.values())
    context = DocumentContext(
        left_context=declared_context.left_context,
        right_context=declared_context.right_context,
        topic_summary=declared_context.topic_summary,
        entity_ids=declared_context.entity_ids,
        metadata={
            **declared_context.metadata,
            "mode": "whole-document-bidirectional-offline",
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "windowCount": len(first_pass.segments),
            "candidateDocumentContainsAllWindows": True,
            "streaming": False,
        },
    )
    if sequence_scorer is None:
        return tuple(candidates), None, None, context

    scorer_paths = tuple(
        (
            path.scorer_arc(
                source_audio_sha256=first_pass.source_audio_sha256,
                config_digest=config.digest,
            ),
        )
        for path in candidates
    )
    score_many = getattr(sequence_scorer, "score_many", None)
    rows = (
        tuple(score_many(scorer_paths, context=context))
        if callable(score_many)
        else tuple(sequence_scorer.score(path, context=context) for path in scorer_paths)
    )
    if len(rows) != len(candidates):
        raise ValueError("document scorer returned the wrong number of path scores")
    expected = {
        path_digest(scorer_path): candidate
        for candidate, scorer_path in zip(candidates, scorer_paths, strict=True)
    }
    by_document: dict[str, GlobalPathScore] = {}
    for row in rows:
        candidate = expected.get(row.path_digest)
        if candidate is None:
            raise ValueError("document scorer returned an unknown path digest")
        if row.context_digest != context.digest:
            raise ValueError("document scorer returned a score for different context")
        if candidate.digest in by_document:
            raise ValueError("document scorer returned a duplicate path score")
        by_document[candidate.digest] = row
    if set(by_document) != {path.digest for path in candidates}:
        raise ValueError("document scorer omitted a candidate path")
    sources = {row.source for row in rows}
    profiles = {row.profile_digest for row in rows}
    if len(sources) != 1 or len(profiles) != 1:
        raise ValueError("one document decision cannot mix scorer identities")
    source = next(iter(sources))
    profile = next(iter(profiles))
    rescored = tuple(
        DocumentPathHypothesis(
            options=path.options,
            overlap_receipts=path.overlap_receipts,
            base_score=path.base_score,
            mean_audio_support=path.mean_audio_support,
            context_score=by_document[path.digest].value,
            final_score=path.base_score
            + config.global_context_weight * by_document[path.digest].value,
            scorer_source=source,
            scorer_profile_digest=profile,
        )
        for path in candidates
    )
    return rescored, source, profile, context


def plan_document_deliberation(
    first_pass: LongformResult,
    *,
    config: DocumentBeamConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    sequence_scorer: GlobalSequenceScorer | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
    audio_path: str | Path | None = None,
    shuffle_seed: str = "semantic-asr-shuffled-context-v1",
) -> DocumentDeliberationPlan:
    """Plan one jointly selected sequence of window paths for the complete recording."""

    config = config or DocumentBeamConfig()
    first_pass.verify()
    if audio_path is not None and sha256_file(audio_path) != first_pass.source_audio_sha256:
        raise ValueError("document deliberation audio_path belongs to a different recording")
    if config.require_sequence_scorer and sequence_scorer is None:
        raise ValueError("document deliberation requires an explicit complete-document scorer")
    build_config = build_config or SemanticDeliberationConfig()
    local_policy = local_policy or DeliberationPolicy.conservative_default()
    declared = declared_context or DocumentContext()
    contexts = build_frozen_window_contexts(
        first_pass,
        arm=config.proposal_context_arm,
        declared_context=declared,
        maximum_left_windows=config.maximum_left_windows,
        maximum_right_windows=config.maximum_right_windows,
        maximum_context_characters=config.maximum_context_characters,
        shuffle_seed=shuffle_seed,
    )
    coverages = _coverage_attribution(first_pass.segments)
    window_sets = tuple(
        _build_window_set(
            first_pass,
            segment,
            contexts[index],
            coverage_ms=coverages[index],
            config=config,
            build_config=build_config,
            local_policy=local_policy,
            proposal_provider=proposal_provider,
            audio_path=audio_path,
        )
        for index, segment in enumerate(first_pass.segments)
    )
    base_paths = _base_document_paths(window_sets, config=config)
    retained = _retained_hypothesis_with_config(window_sets, config=config)
    scored, scorer_source, scorer_profile, scorer_context = _score_document_paths(
        base_paths,
        retained.digest,
        first_pass=first_pass,
        config=config,
        sequence_scorer=sequence_scorer,
        declared_context=declared,
    )
    alternatives = tuple(
        sorted(
            scored,
            key=lambda path: (
                -path.final_score,
                path.changed_window_count,
                path.generated_window_count,
                path.digest,
            ),
        )
    )
    selected = alternatives[0]
    retained_scored = next(
        (path for path in alternatives if path.digest == retained.digest),
        retained,
    )
    has_runner_up = len(alternatives) > 1
    margin = selected.final_score - alternatives[1].final_score if has_runner_up else 0.0
    reasons: list[str] = ["document-joint-beam", "unique-audio-coverage-weighting"]
    if any(receipt.similarity is not None for receipt in selected.overlap_receipts):
        reasons.append("overlap-aware-compatibility")
    status: Literal["accepted", "provisional"] = "accepted"
    if has_runner_up and margin < config.minimum_final_margin:
        status = "provisional"
        reasons.append("low-document-margin")
    if selected.generated_window_count:
        status = "provisional"
        reasons.append("selected-generated-window")
    if sequence_scorer is not None:
        reasons.append("whole-document-context-applied")
    else:
        reasons.append("no-document-scorer")
    decision = DocumentDeliberationDecision(
        selected=selected,
        retained=retained_scored,
        alternatives=alternatives,
        status=status,
        margin=margin,
        reasons=tuple(reasons),
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        config_digest=config.digest,
        local_policy_digest=local_policy.digest,
        context_digest=scorer_context.digest,
        scorer_source=scorer_source,
        scorer_profile_digest=scorer_profile,
    )
    return DocumentDeliberationPlan(
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        window_sets=window_sets,
        decision=decision,
        config_digest=config.digest,
        local_policy_digest=local_policy.digest,
        context_plan_digest=sha256_json([context.digest for context in contexts]),
    )


def _resolution_mode(selected: LatticeArc, retained: LatticeArc) -> str:
    if selected.arc_id == retained.arc_id:
        return "retained-first-pass"
    if (
        selected.pronunciation_key is not None
        and selected.pronunciation_key == retained.pronunciation_key
    ):
        return "context-resolved-orthography"
    if selected.origin in GENERATED_ORIGINS:
        return "acoustically-verified-proposal"
    return "acoustic-context-consensus"


def _local_decision(
    window: WindowPathSet,
    option: WindowPathOption,
    document: DocumentDeliberationDecision,
    *,
    policy_digest: str,
) -> GlobalDeliberationDecision:
    resolutions = tuple(
        SpanResolution(
            span_id=span.span_id,
            retained_arc_id=span.retained_arc_id,
            selected_arc_id=arc.arc_id,
            mode=_resolution_mode(arc, span.retained_arc),  # type: ignore[arg-type]
            retained_audio_support=None,
            selected_audio_support=None,
        )
        for span, arc in zip(window.build.lattice.spans, option.path.arcs, strict=True)
    )
    return GlobalDeliberationDecision(
        selected=option.path,
        retained=window.retained.path,
        alternatives=tuple(row.path for row in window.options),
        status=document.status,
        margin=document.margin,
        resolutions=resolutions,
        reasons=(
            "selected-by-document-joint-beam",
            f"document-decision:{document.digest}",
        ),
        lattice_digest=window.build.lattice.digest,
        policy_digest=policy_digest,
        context_digest=document.context_digest,
        scorer_source=document.scorer_source,
        scorer_profile_digest=document.scorer_profile_digest,
    )


def _trace(
    window: WindowPathSet,
    option: WindowPathOption,
    local: GlobalDeliberationDecision,
    document: DocumentDeliberationDecision,
    *,
    applied: bool,
    config_digest: str,
    policy_digest: str,
    reason: str,
) -> SegmentDeliberationTrace:
    changed_spans = tuple(
        resolution.span_id
        for resolution in local.resolutions
        if resolution.selected_arc_id != resolution.retained_arc_id
    )
    return SegmentDeliberationTrace(
        attempted=True,
        applied=applied,
        reason=f"{reason};document={document.digest}",
        first_pass_evidence_sha256=window.segment.observed.evidence_sha256,
        context_digest=document.context_digest,
        config_digest=config_digest,
        policy_digest=policy_digest,
        build_digest=window.build.digest,
        lattice_digest=window.build.lattice.digest,
        decision_digest=local.digest,
        selected_path_digest=option.path.digest,
        retained_path_digest=window.retained.path.digest,
        scorer_source=document.scorer_source,
        scorer_profile_digest=document.scorer_profile_digest,
        decision_status=document.status,
        margin=document.margin,
        selected_text_sha256=sha256_json({"text": option.text}),
        changed_span_ids=changed_spans,
        proposal_digests=window.build.proposal_digests,
        exact_source_candidate_ids=option.exact_source_candidate_ids,
        recombined=option.recombined,
    )


def _result(
    first_pass: LongformResult,
    segments: Sequence[DeliberatedLongformSegment],
    *,
    config_digest: str,
    policy_digest: str,
    decision: DocumentDeliberationDecision,
    plan_digest: str,
) -> DeliberatedLongformResult:
    rows = tuple(segments)
    observed_text = join_segment_text(rows)
    normalized_text = join_segment_text(rows, normalized=True)
    deliberation_evidence_sha256 = sha256_json(
        {
            "traceDigests": [segment.trace.digest for segment in rows],
            "configDigest": config_digest,
            "policyDigest": policy_digest,
        }
    )
    evidence_sha256 = sha256_json(
        {
            "sourceAudioSha256": first_pass.source_audio_sha256,
            "durationMs": first_pass.duration_ms,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "deliberationEvidenceSha256": deliberation_evidence_sha256,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in rows],
            "segmentWindows": [asdict(segment.window) for segment in rows],
            "normalizations": [asdict(segment.normalized) for segment in rows],
        }
    )
    diagnostics = {
        **dict(first_pass.diagnostics),
        "provisionalWindowCount": sum(
            segment.observed.decision != "accepted" for segment in rows
        ),
        "globalDeliberation": {
            "enabled": True,
            "mode": "document-joint-beam-v1",
            "configDigest": config_digest,
            "policyDigest": policy_digest,
            "documentDecisionDigest": decision.digest,
            "documentPlanDigest": plan_digest,
            "documentStatus": decision.status,
            "documentMargin": decision.margin,
            "scorerSource": decision.scorer_source,
            "scorerProfileDigest": decision.scorer_profile_digest,
            "changedWindowCount": sum(segment.changed for segment in rows),
            "candidateDocumentCount": len(decision.alternatives),
            "overlapReceiptDigests": tuple(
                receipt.digest for receipt in decision.selected.overlap_receipts
            ),
            "evidenceSha256": deliberation_evidence_sha256,
        },
    }
    return DeliberatedLongformResult(
        source_name=first_pass.source_name,
        source_audio_sha256=first_pass.source_audio_sha256,
        duration_ms=first_pass.duration_ms,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=rows,
        evidence_sha256=evidence_sha256,
        diagnostics=diagnostics,
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        deliberation_evidence_sha256=deliberation_evidence_sha256,
        config_digest=config_digest,
        policy_digest=policy_digest,
        first_pass=first_pass,
    )


def apply_document_deliberation(
    first_pass: LongformResult,
    *,
    config: DocumentBeamConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    sequence_scorer: GlobalSequenceScorer | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
    audio_path: str | Path | None = None,
    shuffle_seed: str = "semantic-asr-shuffled-context-v1",
) -> LongformResult | DeliberatedLongformResult:
    """Jointly choose and optionally apply one window-path sequence for a recording."""

    config = config or DocumentBeamConfig()
    if not config.enabled:
        return first_pass
    local_policy = local_policy or DeliberationPolicy.conservative_default()
    try:
        plan = plan_document_deliberation(
            first_pass,
            config=config,
            build_config=build_config,
            local_policy=local_policy,
            sequence_scorer=sequence_scorer,
            proposal_provider=proposal_provider,
            declared_context=declared_context,
            audio_path=audio_path,
            shuffle_seed=shuffle_seed,
        )
    except Exception:
        if config.fail_closed_to_first_pass:
            return first_pass
        raise

    document = plan.decision
    apply_changes = document.changed and (
        document.status == "accepted" or config.apply_provisional
    )
    bound_policy_digest = sha256_json(
        {
            "localPolicyDigest": local_policy.digest,
            "documentDecisionDigest": document.digest,
            "application": "document-joint-beam-v1",
        }
    )
    segments: list[DeliberatedLongformSegment] = []
    for window, option in zip(plan.window_sets, document.selected.options, strict=True):
        local = _local_decision(
            window,
            option,
            document,
            policy_digest=bound_policy_digest,
        )
        applied = apply_changes and option.changed
        if applied:
            reason = "applied-document-joint-beam"
        elif option.changed:
            reason = "document-provisional-not-applied"
        else:
            reason = "document-retained-first-pass"
        trace = _trace(
            window,
            option,
            local,
            document,
            applied=applied,
            config_digest=config.digest,
            policy_digest=bound_policy_digest,
            reason=reason,
        )
        segments.append(
            _applied_segment(window.segment, window.build, local, trace)
            if applied
            else _unchanged_segment(window.segment, trace)
        )
    return _result(
        first_pass,
        segments,
        config_digest=config.digest,
        policy_digest=bound_policy_digest,
        decision=document,
        plan_digest=plan.digest,
    )


class DocumentDeliberatingTranscriber:
    """Composition wrapper; the measured first pass remains untouched."""

    def __init__(
        self,
        first_pass: SemanticASRTranscriber,
        *,
        config: DocumentBeamConfig | None = None,
        build_config: SemanticDeliberationConfig | None = None,
        local_policy: DeliberationPolicy | None = None,
        sequence_scorer: GlobalSequenceScorer | None = None,
        proposal_provider: SpanProposalProvider | None = None,
        declared_context: DocumentContext | None = None,
    ) -> None:
        self.first_pass = first_pass
        self.config = config or DocumentBeamConfig()
        self.build_config = build_config or SemanticDeliberationConfig()
        self.local_policy = local_policy or DeliberationPolicy.conservative_default()
        self.sequence_scorer = sequence_scorer
        self.proposal_provider = proposal_provider
        self.declared_context = declared_context or DocumentContext()
        if self.config.require_sequence_scorer and sequence_scorer is None:
            raise ValueError("document-deliberating transcriber requires a scorer")

    def __getattr__(self, name: str):
        return getattr(self.first_pass, name)

    def transcribe(self, audio_path: str | Path, **kwargs):
        first_pass = self.first_pass.transcribe(audio_path, **kwargs)
        return apply_document_deliberation(
            first_pass,
            config=self.config,
            build_config=self.build_config,
            local_policy=self.local_policy,
            sequence_scorer=self.sequence_scorer,
            proposal_provider=self.proposal_provider,
            declared_context=self.declared_context,
            audio_path=audio_path,
        )


def with_document_deliberation(
    transcriber: SemanticASRTranscriber,
    *,
    sequence_scorer: GlobalSequenceScorer,
    config: DocumentBeamConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    proposal_provider: SpanProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
) -> DocumentDeliberatingTranscriber:
    return DocumentDeliberatingTranscriber(
        transcriber,
        config=config,
        build_config=build_config,
        local_policy=local_policy,
        sequence_scorer=sequence_scorer,
        proposal_provider=proposal_provider,
        declared_context=declared_context,
    )
