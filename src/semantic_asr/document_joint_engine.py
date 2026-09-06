"""Canonical joint document-lattice engine for Semantic ASR v0.3 research.

This engine jointly chooses policy-eligible local paths, overlap emissions, and complete-document
language evidence. It never changes the measured v0.2 default path and never treats context as
acoustic proof. All failures can be represented by an auditable first-pass-preserving receipt.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .contracts import CandidateEvidence, NormalizedTranscript, sha256_json
from .deliberation_evidence import GENERATED_ORIGINS, _is_sha256, _strict_float
from .deliberation_lattice import DocumentContext, LatticeArc, path_digest
from .global_deliberation import (
    DeliberationPolicy,
    PathHypothesis,
    decode_global_lattice,
)
from .global_scorer import GlobalPathScore, GlobalSequenceScorer
from .japanese import deterministic_normalize, join_japanese_fragments
from .longform import LongformResult, LongformSegment, SemanticASRTranscriber, Window
from .semantic_deliberation import (
    SemanticDeliberationBuild,
    SemanticDeliberationConfig,
    VerifiedSpanProposal,
    build_semantic_deliberation_lattice,
    path_is_recombined,
    path_source_candidate_ids,
)

OverlapMethod = Literal[
    "first-window",
    "no-window-overlap",
    "exact-suffix-prefix",
    "normalized-suffix-prefix",
    "full-window-duplicate-retained",
    "full-window-duplicate-suppressed",
    "no-safe-match",
    "ambiguous-conflict",
    "unapplied-first-pass",
]
DocumentDecisionStatus = Literal["accepted", "provisional"]


class DocumentProposalProvider(Protocol):
    def __call__(
        self,
        *,
        audio_path: str | Path | None,
        segment_index: int,
        segment: LongformSegment,
        build: SemanticDeliberationBuild,
        context: DocumentContext,
        source_audio_sha256: str,
    ) -> Mapping[str, Sequence[VerifiedSpanProposal]]: ...


@dataclass(frozen=True, slots=True)
class OverlapPolicy:
    minimum_exact_characters: int = 4
    minimum_normalized_characters: int = 6
    maximum_search_characters: int = 120
    ambiguous_similarity_threshold: float = 0.68
    exact_reward_scale: int = 24
    normalized_reward_scale: int = 32
    no_match_utility: float = -0.05
    ambiguous_utility: float = -0.60
    suppress_full_window_duplicate: bool = False
    schema_version: str = "2"

    def __post_init__(self) -> None:
        for name in (
            "minimum_exact_characters",
            "minimum_normalized_characters",
            "maximum_search_characters",
            "exact_reward_scale",
            "normalized_reward_scale",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_exact_characters > self.maximum_search_characters:
            raise ValueError("minimum exact overlap exceeds the search window")
        if self.minimum_normalized_characters > self.maximum_search_characters:
            raise ValueError("minimum normalized overlap exceeds the search window")
        threshold = _strict_float(
            self.ambiguous_similarity_threshold,
            name="ambiguous_similarity_threshold",
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("ambiguous_similarity_threshold must be in [0, 1]")
        for name in ("no_match_utility", "ambiguous_utility"):
            value = _strict_float(getattr(self, name), name=name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "ambiguous_similarity_threshold", threshold)

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentDeliberationConfig:
    local_paths_per_window: int = 6
    document_beam_size: int = 48
    overlap_weight: float = 0.50
    global_document_weight: float = 1.0
    maximum_document_audio_regression: float = 0.10
    maximum_changed_windows: int = 12
    maximum_changed_ratio: float = 0.50
    minimum_document_margin: float = 0.03
    minimum_distinct_surfaces: int = 2
    require_document_scorer: bool = True
    apply_provisional: bool = False
    provisional_on_generated: bool = True
    provisional_on_ambiguous_overlap: bool = True
    fail_closed_to_first_pass: bool = True
    overlap: OverlapPolicy = field(default_factory=OverlapPolicy)
    schema_version: str = "2"

    def __post_init__(self) -> None:
        for name in ("local_paths_per_window", "document_beam_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("maximum_changed_windows",):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.minimum_distinct_surfaces, bool) or self.minimum_distinct_surfaces < 2:
            raise ValueError("minimum_distinct_surfaces must be at least two")
        for name in (
            "overlap_weight",
            "global_document_weight",
            "maximum_document_audio_regression",
            "maximum_changed_ratio",
            "minimum_document_margin",
        ):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.overlap_weight < 0.0 or self.global_document_weight < 0.0:
            raise ValueError("document score weights must be non-negative")
        if self.maximum_document_audio_regression < 0.0:
            raise ValueError("maximum_document_audio_regression must be non-negative")
        if not 0.0 <= self.maximum_changed_ratio <= 1.0:
            raise ValueError("maximum_changed_ratio must be in [0, 1]")
        if self.minimum_document_margin < 0.0:
            raise ValueError("minimum_document_margin must be non-negative")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                **asdict(self),
                "overlapDigest": self.overlap.digest,
                "decoder": "hierarchical-document-beam-v2",
            }
        )


@dataclass(frozen=True, slots=True)
class OverlapReceipt:
    left_window_index: int | None
    right_window_index: int
    overlap_ms: int
    method: OverlapMethod
    right_trim_characters: int
    matched_characters: int
    normalized_matched_characters: int
    similarity: float
    utility: float
    left_text_sha256: str | None
    right_text_sha256: str
    emitted_text_sha256: str
    policy_digest: str

    def __post_init__(self) -> None:
        if self.right_window_index < 0:
            raise ValueError("right_window_index must be non-negative")
        if self.left_window_index is not None and self.left_window_index < 0:
            raise ValueError("left_window_index must be non-negative")
        if self.overlap_ms < 0:
            raise ValueError("overlap_ms must be non-negative")
        for value in (
            self.right_trim_characters,
            self.matched_characters,
            self.normalized_matched_characters,
        ):
            if value < 0:
                raise ValueError("overlap character counts must be non-negative")
        similarity = _strict_float(self.similarity, name="overlap similarity")
        utility = _strict_float(self.utility, name="overlap utility")
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("overlap similarity must be in [0, 1]")
        if not -1.0 <= utility <= 1.0:
            raise ValueError("overlap utility must be in [-1, 1]")
        for digest in (
            self.right_text_sha256,
            self.emitted_text_sha256,
            self.policy_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("overlap receipt contains an invalid digest")
        if self.left_text_sha256 is not None and not _is_sha256(self.left_text_sha256):
            raise ValueError("left_text_sha256 must be a SHA-256 value")
        object.__setattr__(self, "similarity", similarity)
        object.__setattr__(self, "utility", utility)

    @property
    def ambiguous(self) -> bool:
        return self.method == "ambiguous-conflict"

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class WindowPathOption:
    segment_index: int
    window: Window
    build: SemanticDeliberationBuild
    path: PathHypothesis
    retained_path_digest: str
    option_rank: int

    def __post_init__(self) -> None:
        if self.segment_index < 0 or self.option_rank < 0:
            raise ValueError("window option indexes must be non-negative")
        if not _is_sha256(self.retained_path_digest):
            raise ValueError("retained_path_digest must be a SHA-256 value")
        if len(self.path.arcs) != len(self.build.lattice.spans):
            raise ValueError("window path does not cover every local lattice span")
        if any(
            arc.span_id != span.span_id
            for arc, span in zip(self.path.arcs, self.build.lattice.spans, strict=True)
        ):
            raise ValueError("window path arc order does not match the local lattice")

    @property
    def text(self) -> str:
        return self.path.text

    @property
    def changed(self) -> bool:
        return self.path.digest != self.retained_path_digest

    @property
    def generated(self) -> bool:
        return any(arc.origin in GENERATED_ORIGINS for arc in self.path.arcs)

    @property
    def exact_source_candidate_ids(self) -> tuple[str, ...]:
        return path_source_candidate_ids(self.path.arcs)

    @property
    def recombined(self) -> bool:
        return path_is_recombined(self.path.arcs)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "segmentIndex": self.segment_index,
                "window": asdict(self.window),
                "buildDigest": self.build.digest,
                "pathDigest": self.path.digest,
                "retainedPathDigest": self.retained_path_digest,
                "optionRank": self.option_rank,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentPathCandidate:
    options: tuple[WindowPathOption, ...]
    emitted_texts: tuple[str, ...]
    overlap_receipts: tuple[OverlapReceipt, ...]
    local_score: float
    overlap_score: float
    mean_audio_support: float
    global_score: float = 0.0
    final_score: float = 0.0
    scorer_source: str | None = None
    scorer_profile_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("document candidate requires at least one window option")
        if len(self.options) != len(self.emitted_texts):
            raise ValueError("one emitted text is required for every window option")
        if len(self.options) != len(self.overlap_receipts):
            raise ValueError("one overlap receipt is required for every window option")
        if tuple(option.segment_index for option in self.options) != tuple(range(len(self.options))):
            raise ValueError("document candidate window indexes must be contiguous from zero")
        for name in (
            "local_score",
            "overlap_score",
            "mean_audio_support",
            "global_score",
            "final_score",
        ):
            _strict_float(getattr(self, name), name=name)
        if not -1.0 <= self.mean_audio_support <= 1.0:
            raise ValueError("document mean_audio_support must be in [-1, 1]")
        if not -1.0 <= self.global_score <= 1.0:
            raise ValueError("document global_score must be in [-1, 1]")
        if (self.scorer_source is None) != (self.scorer_profile_digest is None):
            raise ValueError("document scorer source and profile must be supplied together")
        if self.scorer_profile_digest is not None and not _is_sha256(
            self.scorer_profile_digest
        ):
            raise ValueError("document scorer profile must be a SHA-256 value")
        for option, emitted, receipt in zip(
            self.options,
            self.emitted_texts,
            self.overlap_receipts,
            strict=True,
        ):
            if option.text[receipt.right_trim_characters :] != emitted:
                raise ValueError("overlap receipt does not reconstruct emitted window text")
            if _text_sha256(emitted) != receipt.emitted_text_sha256:
                raise ValueError("emitted text digest does not match overlap receipt")

    @property
    def text(self) -> str:
        return join_japanese_fragments(self.emitted_texts)

    @property
    def selection_digest(self) -> str:
        """Score-independent structural identity used across pre/post-scoring objects."""

        return sha256_json(
            {
                "optionDigests": [option.digest for option in self.options],
                "emittedTextSha256": [_text_sha256(text) for text in self.emitted_texts],
                "overlapReceiptDigests": [row.digest for row in self.overlap_receipts],
            }
        )

    @property
    def changed_window_indexes(self) -> tuple[int, ...]:
        return tuple(option.segment_index for option in self.options if option.changed)

    @property
    def generated_window_indexes(self) -> tuple[int, ...]:
        return tuple(option.segment_index for option in self.options if option.generated)

    @property
    def ambiguous_overlap_indexes(self) -> tuple[int, ...]:
        return tuple(
            receipt.right_window_index
            for receipt in self.overlap_receipts
            if receipt.ambiguous
        )

    @property
    def path_arcs(self) -> tuple[LatticeArc, ...]:
        return tuple(arc for option in self.options for arc in option.path.arcs)

    @property
    def scoring_arcs(self) -> tuple[LatticeArc, ...]:
        """One evidence-bound arc whose text is exactly the emitted document text."""

        return (
            LatticeArc(
                arc_id=f"document-emission:{self.selection_digest[:24]}",
                span_id="document-emission",
                text=self.text,
                origin="first-pass",
                utilities=(),
                observed_eligible=False,
                source_audio_sha256=self.options[0].build.lattice.source_audio_sha256,
                is_epsilon=not self.text,
                metadata={
                    "selectionDigest": self.selection_digest,
                    "optionDigests": tuple(option.digest for option in self.options),
                    "overlapReceiptDigests": tuple(
                        receipt.digest for receipt in self.overlap_receipts
                    ),
                    "underlyingArcDigests": tuple(
                        arc.digest for option in self.options for arc in option.path.arcs
                    ),
                    "emittedTextSha256": _text_sha256(self.text),
                },
            ),
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "selectionDigest": self.selection_digest,
                "localScore": self.local_score,
                "overlapScore": self.overlap_score,
                "meanAudioSupport": self.mean_audio_support,
                "globalScore": self.global_score,
                "finalScore": self.final_score,
                "scorerSource": self.scorer_source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentDeliberationDecision:
    selected: DocumentPathCandidate
    retained: DocumentPathCandidate
    alternatives: tuple[DocumentPathCandidate, ...]
    status: DocumentDecisionStatus
    applied: bool
    margin: float
    reasons: tuple[str, ...]
    first_pass_evidence_sha256: str
    config_digest: str
    local_policy_digest: str
    context_digest: str
    source_audio_sha256: str

    def __post_init__(self) -> None:
        margin = _strict_float(self.margin, name="document decision margin")
        if margin < 0.0:
            raise ValueError("document decision margin must be non-negative")
        for digest in (
            self.first_pass_evidence_sha256,
            self.config_digest,
            self.local_policy_digest,
            self.context_digest,
            self.source_audio_sha256,
        ):
            if not _is_sha256(digest):
                raise ValueError("document decision contains an invalid digest")
        if self.applied and self.status == "provisional" and (
            "explicitly-applied-provisional" not in self.reasons
        ):
            raise ValueError("applied provisional decision requires an explicit reason")
        object.__setattr__(self, "margin", margin)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "selectedDigest": self.selected.digest,
                "retainedDigest": self.retained.digest,
                "alternativeDigests": [row.digest for row in self.alternatives],
                "status": self.status,
                "applied": self.applied,
                "margin": self.margin,
                "reasons": self.reasons,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "configDigest": self.config_digest,
                "localPolicyDigest": self.local_policy_digest,
                "contextDigest": self.context_digest,
                "sourceAudioSha256": self.source_audio_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class DocumentFailureReceipt:
    error_type: str
    error_message_sha256: str
    first_pass_evidence_sha256: str
    source_audio_sha256: str
    config_digest: str
    local_policy_digest: str
    stage: str = "joint-document-deliberation"

    def __post_init__(self) -> None:
        if not self.error_type or not self.stage:
            raise ValueError("document failure receipt requires error type and stage")
        for digest in (
            self.error_message_sha256,
            self.first_pass_evidence_sha256,
            self.source_audio_sha256,
            self.config_digest,
            self.local_policy_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("document failure receipt contains an invalid digest")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentPassThroughResult:
    source_name: str
    source_audio_sha256: str
    duration_ms: int
    observed_text: str
    normalized_text: str
    segments: tuple[LongformSegment, ...]
    evidence_sha256: str
    diagnostics: dict[str, object]
    first_pass_evidence_sha256: str
    failure: DocumentFailureReceipt
    first_pass: LongformResult = field(repr=False)

    def __post_init__(self) -> None:
        self.verify()
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @classmethod
    def create(
        cls,
        first_pass: LongformResult,
        *,
        error: Exception,
        config: DocumentDeliberationConfig,
        local_policy: DeliberationPolicy,
    ) -> DocumentPassThroughResult:
        failure = DocumentFailureReceipt(
            error_type=type(error).__name__,
            error_message_sha256=_text_sha256(str(error)),
            first_pass_evidence_sha256=first_pass.evidence_sha256,
            source_audio_sha256=first_pass.source_audio_sha256,
            config_digest=config.digest,
            local_policy_digest=local_policy.digest,
        )
        evidence = sha256_json(
            {
                "firstPassEvidenceSha256": first_pass.evidence_sha256,
                "failureDigest": failure.digest,
                "segmentEvidence": [
                    segment.observed.evidence_sha256 for segment in first_pass.segments
                ],
            }
        )
        diagnostics = {
            **dict(first_pass.diagnostics),
            "documentJointDeliberation": {
                "enabled": True,
                "applied": False,
                "failedClosed": True,
                "failureDigest": failure.digest,
                "errorType": failure.error_type,
                "firstPassEvidenceSha256": first_pass.evidence_sha256,
            },
        }
        return cls(
            source_name=first_pass.source_name,
            source_audio_sha256=first_pass.source_audio_sha256,
            duration_ms=first_pass.duration_ms,
            observed_text=first_pass.observed_text,
            normalized_text=first_pass.normalized_text,
            segments=first_pass.segments,
            evidence_sha256=evidence,
            diagnostics=diagnostics,
            first_pass_evidence_sha256=first_pass.evidence_sha256,
            failure=failure,
            first_pass=first_pass,
        )

    def as_dict(self) -> dict[str, object]:
        payload = dict(self.first_pass.as_dict())
        payload.update(
            {
                "evidence_sha256": self.evidence_sha256,
                "first_pass_evidence_sha256": self.first_pass_evidence_sha256,
                "document_failure": asdict(self.failure),
                "document_failure_digest": self.failure.digest,
                "diagnostics": dict(self.diagnostics),
            }
        )
        return payload

    def verify(self) -> None:
        if self.first_pass.evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("pass-through result is linked to different first-pass evidence")
        if self.failure.first_pass_evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("failure receipt is linked to different first-pass evidence")
        if self.observed_text != self.first_pass.observed_text:
            raise ValueError("fail-closed result changed observed text")
        expected = sha256_json(
            {
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "failureDigest": self.failure.digest,
                "segmentEvidence": [
                    segment.observed.evidence_sha256 for segment in self.segments
                ],
            }
        )
        if expected != self.evidence_sha256:
            raise ValueError("pass-through evidence hash mismatch")


@dataclass(frozen=True, slots=True)
class DocumentArcReceipt:
    arc_id: str
    span_id: str
    arc_digest: str
    text: str
    origin: str
    source_candidate_ids: tuple[str, ...]
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.arc_id or not self.span_id or not _is_sha256(self.arc_digest):
            raise ValueError("document arc receipt requires IDs and a SHA-256")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("document arc receipt has an invalid time range")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class DocumentObservedTranscript:
    text: str
    full_window_text: str
    trim_prefix_characters: int
    selected_candidate_id: str
    source_audio_sha256: str
    evidence_sha256: str
    first_pass_evidence_sha256: str
    document_decision_digest: str
    window_option_digest: str
    overlap_receipt: OverlapReceipt
    path_arcs: tuple[DocumentArcReceipt, ...]
    decision: str
    candidates: tuple[CandidateEvidence, ...]
    ranked: tuple[object, ...]
    uncertainty_spans: tuple[dict[str, object], ...] = ()
    selected_posterior: float = 0.0

    def __post_init__(self) -> None:
        if not self.selected_candidate_id or not self.full_window_text or not self.path_arcs:
            raise ValueError("document observed transcript requires a selected full-window path")
        if self.trim_prefix_characters < 0:
            raise ValueError("trim_prefix_characters must be non-negative")
        if self.full_window_text[self.trim_prefix_characters :] != self.text:
            raise ValueError("emitted observed text does not match its overlap trim")
        if not self.candidates or not self.ranked:
            raise ValueError("document observed transcript must retain first-pass evidence")
        if self.decision not in {"accepted", "provisional"}:
            raise ValueError("document observed decision must be accepted or provisional")
        for digest in (
            self.source_audio_sha256,
            self.evidence_sha256,
            self.first_pass_evidence_sha256,
            self.document_decision_digest,
            self.window_option_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("document observed transcript contains an invalid digest")
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        option: WindowPathOption,
        emitted_text: str,
        receipt: OverlapReceipt,
        first_pass_segment: LongformSegment,
        document_decision_digest: str,
        decision: DocumentDecisionStatus,
    ) -> DocumentObservedTranscript:
        selected_candidate_id = (
            f"document-path-{option.path.digest[:14]}-{receipt.digest[:8]}"
        )
        arc_receipts = tuple(
            DocumentArcReceipt(
                arc_id=arc.arc_id,
                span_id=arc.span_id,
                arc_digest=arc.digest,
                text=arc.text,
                origin=arc.origin,
                source_candidate_ids=arc.source_candidate_ids,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
            )
            for span, arc in zip(option.build.lattice.spans, option.path.arcs, strict=True)
        )
        payload = {
            "text": emitted_text,
            "fullWindowText": option.text,
            "trimPrefixCharacters": receipt.right_trim_characters,
            "selectedCandidateId": selected_candidate_id,
            "sourceAudioSha256": option.build.lattice.source_audio_sha256,
            "firstPassEvidenceSha256": first_pass_segment.observed.evidence_sha256,
            "documentDecisionDigest": document_decision_digest,
            "windowOptionDigest": option.digest,
            "overlapReceiptDigest": receipt.digest,
            "pathArcDigests": [row.digest for row in arc_receipts],
            "decision": decision,
        }
        return cls(
            text=emitted_text,
            full_window_text=option.text,
            trim_prefix_characters=receipt.right_trim_characters,
            selected_candidate_id=selected_candidate_id,
            source_audio_sha256=option.build.lattice.source_audio_sha256,
            evidence_sha256=sha256_json(payload),
            first_pass_evidence_sha256=first_pass_segment.observed.evidence_sha256,
            document_decision_digest=document_decision_digest,
            window_option_digest=option.digest,
            overlap_receipt=receipt,
            path_arcs=arc_receipts,
            decision=decision,
            candidates=first_pass_segment.observed.candidates,
            ranked=first_pass_segment.observed.ranked,
            uncertainty_spans=tuple(
                {
                    "spanId": span.span_id,
                    "startMs": span.start_ms,
                    "endMs": span.end_ms,
                    "selectedArcId": arc.arc_id,
                    "retainedArcId": span.retained_arc_id,
                }
                for span, arc in zip(option.build.lattice.spans, option.path.arcs, strict=True)
                if arc.arc_id != span.retained_arc_id
            ),
        )

    def verify(self) -> None:
        if "".join(row.text for row in self.path_arcs) != self.full_window_text:
            raise ValueError("document arc receipts do not reconstruct full-window text")
        payload = {
            "text": self.text,
            "fullWindowText": self.full_window_text,
            "trimPrefixCharacters": self.trim_prefix_characters,
            "selectedCandidateId": self.selected_candidate_id,
            "sourceAudioSha256": self.source_audio_sha256,
            "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
            "documentDecisionDigest": self.document_decision_digest,
            "windowOptionDigest": self.window_option_digest,
            "overlapReceiptDigest": self.overlap_receipt.digest,
            "pathArcDigests": [row.digest for row in self.path_arcs],
            "decision": self.decision,
        }
        if sha256_json(payload) != self.evidence_sha256:
            raise ValueError("document observed evidence hash mismatch")


@dataclass(frozen=True, slots=True)
class DocumentNormalizedTranscript:
    """Normalization link that permits a fully suppressed duplicate emission."""

    text: str
    observed_evidence_sha256: str
    mode: str = "deterministic"
    normalizer_version: str = "document-emission-v1"

    def __post_init__(self) -> None:
        if not _is_sha256(self.observed_evidence_sha256):
            raise ValueError("normalized emission requires an observed evidence SHA-256")
        if not self.mode or not self.normalizer_version:
            raise ValueError("normalized emission requires mode and version")


@dataclass(frozen=True, slots=True)
class DocumentDeliberatedSegment:
    window: Window
    observed: DocumentObservedTranscript
    normalized: NormalizedTranscript | DocumentNormalizedTranscript
    diagnostics: dict[str, object]
    actions: tuple[object, ...] = ()
    cache_hits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.observed.verify()
        if self.normalized.observed_evidence_sha256 != self.observed.evidence_sha256:
            raise ValueError("document segment normalization is linked to different evidence")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def as_dict(self) -> dict[str, object]:
        return {
            "window": asdict(self.window),
            "observed": asdict(self.observed),
            "normalized": asdict(self.normalized),
            "diagnostics": dict(self.diagnostics),
            "actions": [
                asdict(action) if hasattr(action, "__dataclass_fields__") else action
                for action in self.actions
            ],
            "cache_hits": list(self.cache_hits),
        }


@dataclass(frozen=True, slots=True)
class DocumentDeliberatedResult:
    source_name: str
    source_audio_sha256: str
    duration_ms: int
    observed_text: str
    normalized_text: str
    segments: tuple[DocumentDeliberatedSegment, ...]
    evidence_sha256: str
    diagnostics: dict[str, object]
    first_pass_evidence_sha256: str
    document_decision: DocumentDeliberationDecision
    first_pass: LongformResult = field(repr=False)

    def __post_init__(self) -> None:
        self.verify()
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def as_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_audio_sha256": self.source_audio_sha256,
            "duration_ms": self.duration_ms,
            "observed_text": self.observed_text,
            "normalized_text": self.normalized_text,
            "segments": [segment.as_dict() for segment in self.segments],
            "evidence_sha256": self.evidence_sha256,
            "first_pass_evidence_sha256": self.first_pass_evidence_sha256,
            "document_decision": asdict(self.document_decision),
            "document_decision_digest": self.document_decision.digest,
            "diagnostics": dict(self.diagnostics),
        }

    def verify(self) -> None:
        if self.first_pass.evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("document result is linked to different first-pass evidence")
        if self.document_decision.first_pass_evidence_sha256 != self.first_pass_evidence_sha256:
            raise ValueError("document decision is linked to different first-pass evidence")
        if self.document_decision.source_audio_sha256 != self.source_audio_sha256:
            raise ValueError("document decision is linked to different source audio")
        for segment in self.segments:
            segment.observed.verify()
        observed = join_japanese_fragments(segment.observed.text for segment in self.segments)
        normalized = join_japanese_fragments(segment.normalized.text for segment in self.segments)
        if observed != self.observed_text or normalized != self.normalized_text:
            raise ValueError("document text does not match emitted segment sequence")
        expected = sha256_json(
            {
                "sourceAudioSha256": self.source_audio_sha256,
                "durationMs": self.duration_ms,
                "observedText": self.observed_text,
                "normalizedText": self.normalized_text,
                "firstPassEvidenceSha256": self.first_pass_evidence_sha256,
                "documentDecisionDigest": self.document_decision.digest,
                "segmentEvidence": [segment.observed.evidence_sha256 for segment in self.segments],
            }
        )
        if expected != self.evidence_sha256:
            raise ValueError("document deliberation evidence hash mismatch")


@dataclass(frozen=True, slots=True)
class _BeamState:
    options: tuple[WindowPathOption, ...]
    emitted_texts: tuple[str, ...]
    receipts: tuple[OverlapReceipt, ...]
    local_score: float
    overlap_score: float
    weighted_audio_sum: float
    audio_weight: float

    @property
    def ranking_score(self) -> float:
        return self.local_score + self.overlap_score

    @property
    def mean_audio_support(self) -> float:
        return -1.0 if self.audio_weight <= 0.0 else self.weighted_audio_sum / self.audio_weight


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalization_map(text: str) -> tuple[str, tuple[int, ...]]:
    output: list[str] = []
    original_ends: list[int] = []
    for index, character in enumerate(text):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for value in normalized:
            category = unicodedata.category(value)
            if value.isspace() or category.startswith("P") or category.startswith("S"):
                continue
            output.append(value)
            original_ends.append(index + 1)
    return "".join(output), tuple(original_ends)


def _longest_suffix_prefix(left: str, right: str, maximum: int) -> int:
    limit = min(len(left), len(right), maximum)
    for length in range(limit, 0, -1):
        if left[-length:] == right[:length]:
            return length
    return 0


def _sequence_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(left), len(right)))


def _receipt(
    *,
    left_text: str | None,
    right_text: str,
    emitted: str,
    left_window_index: int | None,
    right_window_index: int,
    overlap_ms: int,
    method: OverlapMethod,
    trim: int,
    exact: int,
    normalized: int,
    similarity: float,
    utility: float,
    policy: OverlapPolicy,
) -> OverlapReceipt:
    return OverlapReceipt(
        left_window_index=left_window_index,
        right_window_index=right_window_index,
        overlap_ms=overlap_ms,
        method=method,
        right_trim_characters=trim,
        matched_characters=exact,
        normalized_matched_characters=normalized,
        similarity=similarity,
        utility=utility,
        left_text_sha256=None if left_text is None else _text_sha256(left_text),
        right_text_sha256=_text_sha256(right_text),
        emitted_text_sha256=_text_sha256(emitted),
        policy_digest=policy.digest,
    )


def resolve_window_overlap(
    left_text: str | None,
    right_text: str,
    *,
    left_window_index: int | None,
    right_window_index: int,
    overlap_ms: int,
    policy: OverlapPolicy,
) -> tuple[str, OverlapReceipt]:
    if not right_text:
        raise ValueError("right window text must not be empty")
    if left_text is None:
        return right_text, _receipt(
            left_text=None,
            right_text=right_text,
            emitted=right_text,
            left_window_index=None,
            right_window_index=right_window_index,
            overlap_ms=0,
            method="first-window",
            trim=0,
            exact=0,
            normalized=0,
            similarity=0.0,
            utility=0.0,
            policy=policy,
        )
    if overlap_ms <= 0:
        return right_text, _receipt(
            left_text=left_text,
            right_text=right_text,
            emitted=right_text,
            left_window_index=left_window_index,
            right_window_index=right_window_index,
            overlap_ms=0,
            method="no-window-overlap",
            trim=0,
            exact=0,
            normalized=0,
            similarity=0.0,
            utility=0.0,
            policy=policy,
        )

    exact = _longest_suffix_prefix(left_text, right_text, policy.maximum_search_characters)
    if exact >= policy.minimum_exact_characters:
        if exact == len(right_text) and not policy.suppress_full_window_duplicate:
            return right_text, _receipt(
                left_text=left_text,
                right_text=right_text,
                emitted=right_text,
                left_window_index=left_window_index,
                right_window_index=right_window_index,
                overlap_ms=overlap_ms,
                method="full-window-duplicate-retained",
                trim=0,
                exact=exact,
                normalized=exact,
                similarity=1.0,
                utility=0.0,
                policy=policy,
            )
        emitted = right_text[exact:]
        method: OverlapMethod = (
            "full-window-duplicate-suppressed" if not emitted else "exact-suffix-prefix"
        )
        return emitted, _receipt(
            left_text=left_text,
            right_text=right_text,
            emitted=emitted,
            left_window_index=left_window_index,
            right_window_index=right_window_index,
            overlap_ms=overlap_ms,
            method=method,
            trim=exact,
            exact=exact,
            normalized=exact,
            similarity=1.0,
            utility=min(1.0, exact / policy.exact_reward_scale),
            policy=policy,
        )

    normalized_left, _ = _normalization_map(left_text)
    normalized_right, right_map = _normalization_map(right_text)
    normalized_match = _longest_suffix_prefix(
        normalized_left,
        normalized_right,
        policy.maximum_search_characters,
    )
    if normalized_match >= policy.minimum_normalized_characters and right_map:
        trim = right_map[normalized_match - 1]
        if trim == len(right_text) and not policy.suppress_full_window_duplicate:
            return right_text, _receipt(
                left_text=left_text,
                right_text=right_text,
                emitted=right_text,
                left_window_index=left_window_index,
                right_window_index=right_window_index,
                overlap_ms=overlap_ms,
                method="full-window-duplicate-retained",
                trim=0,
                exact=0,
                normalized=normalized_match,
                similarity=1.0,
                utility=0.0,
                policy=policy,
            )
        emitted = right_text[trim:]
        method = (
            "full-window-duplicate-suppressed" if not emitted else "normalized-suffix-prefix"
        )
        return emitted, _receipt(
            left_text=left_text,
            right_text=right_text,
            emitted=emitted,
            left_window_index=left_window_index,
            right_window_index=right_window_index,
            overlap_ms=overlap_ms,
            method=method,
            trim=trim,
            exact=0,
            normalized=normalized_match,
            similarity=1.0,
            utility=min(1.0, normalized_match / policy.normalized_reward_scale),
            policy=policy,
        )

    left_tail = normalized_left[-policy.maximum_search_characters :]
    right_head = normalized_right[: policy.maximum_search_characters]
    comparison = min(len(left_tail), len(right_head))
    similarity = (
        _sequence_similarity(left_tail[-comparison:], right_head[:comparison])
        if comparison
        else 0.0
    )
    ambiguous = similarity >= policy.ambiguous_similarity_threshold
    return right_text, _receipt(
        left_text=left_text,
        right_text=right_text,
        emitted=right_text,
        left_window_index=left_window_index,
        right_window_index=right_window_index,
        overlap_ms=overlap_ms,
        method="ambiguous-conflict" if ambiguous else "no-safe-match",
        trim=0,
        exact=0,
        normalized=normalized_match,
        similarity=similarity,
        utility=policy.ambiguous_utility if ambiguous else policy.no_match_utility,
        policy=policy,
    )


def _unapplied_receipt(
    previous_text: str | None,
    current_text: str,
    *,
    index: int,
    window: Window,
    previous_window: Window | None,
    policy: OverlapPolicy,
) -> OverlapReceipt:
    overlap_ms = 0 if previous_window is None else max(0, previous_window.end_ms - window.start_ms)
    return _receipt(
        left_text=previous_text,
        right_text=current_text,
        emitted=current_text,
        left_window_index=None if previous_window is None else index - 1,
        right_window_index=index,
        overlap_ms=overlap_ms,
        method="unapplied-first-pass",
        trim=0,
        exact=0,
        normalized=0,
        similarity=0.0,
        utility=0.0,
        policy=policy,
    )


def _posterior(segment: LongformSegment) -> dict[str, float] | None:
    if not segment.observed.ranked:
        return None
    values = dict(segment.observed.ranked[0].gate.posterior)
    expected = {candidate.candidate_id for candidate in segment.observed.candidates}
    return values if set(values) == expected else None


def _pivot_timeline(segment: LongformSegment) -> tuple[tuple[int, int, int, int], ...]:
    candidate = next(
        (
            row
            for row in segment.observed.candidates
            if row.candidate_id == segment.observed.selected_candidate_id
        ),
        None,
    )
    if candidate is None or not candidate.mora_units:
        return ()
    duration = segment.window.end_ms - segment.window.start_ms
    output: list[tuple[int, int, int, int]] = []
    for unit in candidate.mora_units:
        if (
            unit.char_start is None
            or unit.char_end is None
            or unit.start_ms is None
            or unit.end_ms is None
        ):
            return ()
        start = round(float(unit.start_ms))
        end = round(float(unit.end_ms))
        if 0 <= start < end <= duration:
            start += segment.window.start_ms
            end += segment.window.start_ms
        output.append((unit.char_start, unit.char_end, start, end))
    return tuple(output)


def _window_context(
    first_pass: LongformResult,
    segment_index: int,
    declared: DocumentContext,
) -> DocumentContext:
    return DocumentContext(
        left_context="\n".join(
            value
            for value in (
                declared.left_context,
                *(segment.observed.text for segment in first_pass.segments[:segment_index]),
            )
            if value
        ),
        right_context="\n".join(
            value
            for value in (
                *(segment.observed.text for segment in first_pass.segments[segment_index + 1 :]),
                declared.right_context,
            )
            if value
        ),
        topic_summary=declared.topic_summary,
        entity_ids=declared.entity_ids,
        metadata={
            **declared.metadata,
            "targetWindowIndex": segment_index,
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "contextSource": "frozen-full-first-pass-document-v1",
        },
    )


def _document_context(first_pass: LongformResult, declared: DocumentContext) -> DocumentContext:
    return DocumentContext(
        left_context=declared.left_context,
        right_context=declared.right_context,
        topic_summary=declared.topic_summary,
        entity_ids=declared.entity_ids,
        metadata={
            **declared.metadata,
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "sourceAudioSha256": first_pass.source_audio_sha256,
            "windowCount": len(first_pass.segments),
            "contextSource": "declared-external-plus-complete-path-v1",
        },
    )


def _window_options(
    first_pass: LongformResult,
    segment_index: int,
    *,
    config: DocumentDeliberationConfig,
    build_config: SemanticDeliberationConfig,
    local_policy: DeliberationPolicy,
    proposal_provider: DocumentProposalProvider | None,
    declared_context: DocumentContext,
    audio_path: str | Path | None,
) -> tuple[WindowPathOption, ...]:
    segment = first_pass.segments[segment_index]
    document_id = f"{first_pass.source_audio_sha256[:16]}:document-window:{segment_index:04d}"
    kwargs = {
        "candidates": segment.observed.candidates,
        "posterior": _posterior(segment),
        "pivot_candidate_id": segment.observed.selected_candidate_id,
        "document_id": document_id,
        "source_audio_sha256": first_pass.source_audio_sha256,
        "segment_start_ms": segment.window.start_ms,
        "segment_end_ms": segment.window.end_ms,
        "pivot_timeline": _pivot_timeline(segment),
        "config": build_config,
    }
    build = build_semantic_deliberation_lattice(**kwargs)
    if proposal_provider is not None:
        proposals = proposal_provider(
            audio_path=audio_path,
            segment_index=segment_index,
            segment=segment,
            build=build,
            context=_window_context(first_pass, segment_index, declared_context),
            source_audio_sha256=first_pass.source_audio_sha256,
        )
        if proposals:
            build = build_semantic_deliberation_lattice(**kwargs, proposals=proposals)
    local = decode_global_lattice(
        build.lattice,
        policy=local_policy,
        context=DocumentContext(),
        sequence_scorer=None,
    )
    retained_digest = local.retained.digest
    candidates = list(local.alternatives)
    if retained_digest not in {row.digest for row in candidates}:
        candidates.append(local.retained)
    candidates.sort(key=lambda row: (-row.base_score, row.digest))
    distinct = {candidate.text for candidate in segment.observed.candidates}
    if len(distinct) < config.minimum_distinct_surfaces:
        selected = [local.retained]
    else:
        selected = candidates[: config.local_paths_per_window]
        if retained_digest not in {row.digest for row in selected}:
            selected[-1] = local.retained
            selected.sort(key=lambda row: (-row.base_score, row.digest))
    return tuple(
        WindowPathOption(
            segment_index=segment_index,
            window=segment.window,
            build=build,
            path=path,
            retained_path_digest=retained_digest,
            option_rank=rank,
        )
        for rank, path in enumerate(selected)
    )


def _window_audio_weight(option: WindowPathOption) -> float:
    return (
        0.0
        if option.path.mean_audio_support < -0.999999
        else max(1.0, option.window.end_ms - option.window.start_ms)
    )


def _changed_limit(config: DocumentDeliberationConfig, window_count: int) -> int:
    ratio_limit = math.floor(config.maximum_changed_ratio * window_count)
    if config.maximum_changed_ratio > 0.0 and ratio_limit == 0:
        ratio_limit = 1
    return min(config.maximum_changed_windows, ratio_limit)


def _expand_document_beam(
    options_by_window: Sequence[Sequence[WindowPathOption]],
    *,
    config: DocumentDeliberationConfig,
) -> tuple[DocumentPathCandidate, ...]:
    states = (
        _BeamState(
            options=(),
            emitted_texts=(),
            receipts=(),
            local_score=0.0,
            overlap_score=0.0,
            weighted_audio_sum=0.0,
            audio_weight=0.0,
        ),
    )
    changed_limit = _changed_limit(config, len(options_by_window))
    for index, window_options in enumerate(options_by_window):
        expanded: list[_BeamState] = []
        for state in states:
            left_option = state.options[-1] if state.options else None
            for option in window_options:
                changed = sum(row.changed for row in state.options) + int(option.changed)
                if changed > changed_limit:
                    continue
                overlap_ms = (
                    0
                    if left_option is None
                    else max(0, left_option.window.end_ms - option.window.start_ms)
                )
                emitted, receipt = resolve_window_overlap(
                    None if left_option is None else left_option.text,
                    option.text,
                    left_window_index=None if left_option is None else left_option.segment_index,
                    right_window_index=option.segment_index,
                    overlap_ms=overlap_ms,
                    policy=config.overlap,
                )
                weight = _window_audio_weight(option)
                expanded.append(
                    _BeamState(
                        options=state.options + (option,),
                        emitted_texts=state.emitted_texts + (emitted,),
                        receipts=state.receipts + (receipt,),
                        local_score=state.local_score + option.path.base_score,
                        overlap_score=state.overlap_score + config.overlap_weight * receipt.utility,
                        weighted_audio_sum=(
                            state.weighted_audio_sum + weight * option.path.mean_audio_support
                        ),
                        audio_weight=state.audio_weight + weight,
                    )
                )
        if not expanded:
            raise ValueError(f"document beam exhausted at window {index}")
        expanded.sort(
            key=lambda row: (
                -row.ranking_score,
                tuple(option.digest for option in row.options),
            )
        )
        states = tuple(expanded[: config.document_beam_size])
    return tuple(
        DocumentPathCandidate(
            options=state.options,
            emitted_texts=state.emitted_texts,
            overlap_receipts=state.receipts,
            local_score=state.local_score,
            overlap_score=state.overlap_score,
            mean_audio_support=state.mean_audio_support,
            final_score=state.ranking_score,
        )
        for state in states
    )


def _retained_document_candidate(
    options_by_window: Sequence[Sequence[WindowPathOption]],
    *,
    config: DocumentDeliberationConfig,
) -> DocumentPathCandidate:
    rows = []
    for options in options_by_window:
        retained = next(
            option for option in options if option.path.digest == option.retained_path_digest
        )
        rows.append((retained,))
    return _expand_document_beam(rows, config=config)[0]


def _score_document_candidates(
    candidates: Sequence[DocumentPathCandidate],
    scorer: GlobalSequenceScorer | None,
    *,
    context: DocumentContext,
    config: DocumentDeliberationConfig,
) -> tuple[DocumentPathCandidate, ...]:
    if not candidates:
        raise ValueError("document candidates must not be empty")
    if scorer is None:
        return tuple(candidates)
    score_many = getattr(scorer, "score_many", None)
    if callable(score_many):
        rows = tuple(
            score_many(
                tuple(candidate.scoring_arcs for candidate in candidates),
                context=context,
            )
        )
    else:
        rows = tuple(
            scorer.score(candidate.scoring_arcs, context=context) for candidate in candidates
        )
    if len(rows) != len(candidates):
        raise ValueError("document scorer returned the wrong number of scores")
    by_digest: dict[str, GlobalPathScore] = {}
    for row in rows:
        if row.context_digest != context.digest:
            raise ValueError("document score is bound to different context")
        if row.path_digest in by_digest:
            raise ValueError("document scorer returned a duplicate path score")
        by_digest[row.path_digest] = row
    expected = {path_digest(candidate.scoring_arcs) for candidate in candidates}
    if set(by_digest) != expected:
        raise ValueError("document scorer returned unknown or missing path scores")
    if len({row.source for row in rows}) != 1 or len({row.profile_digest for row in rows}) != 1:
        raise ValueError("one document decision cannot mix scorer identities")
    output = []
    for candidate in candidates:
        row = by_digest[path_digest(candidate.scoring_arcs)]
        output.append(
            DocumentPathCandidate(
                options=candidate.options,
                emitted_texts=candidate.emitted_texts,
                overlap_receipts=candidate.overlap_receipts,
                local_score=candidate.local_score,
                overlap_score=candidate.overlap_score,
                mean_audio_support=candidate.mean_audio_support,
                global_score=row.value,
                final_score=(
                    candidate.local_score
                    + candidate.overlap_score
                    + config.global_document_weight * row.value
                ),
                scorer_source=row.source,
                scorer_profile_digest=row.profile_digest,
            )
        )
    output.sort(key=lambda row: (-row.final_score, -row.local_score, row.selection_digest))
    return tuple(output)


def _output_changed(first_pass: LongformResult, candidate: DocumentPathCandidate) -> bool:
    return candidate.text != first_pass.observed_text


def _make_decision(
    first_pass: LongformResult,
    scored: Sequence[DocumentPathCandidate],
    retained_before_scoring: DocumentPathCandidate,
    *,
    config: DocumentDeliberationConfig,
    local_policy: DeliberationPolicy,
    context: DocumentContext,
) -> DocumentDeliberationDecision:
    if not scored:
        raise ValueError("no document path survived policy guards")
    selected = scored[0]
    retained = next(
        (
            row
            for row in scored
            if row.selection_digest == retained_before_scoring.selection_digest
        ),
        retained_before_scoring,
    )
    has_runner_up = len(scored) > 1
    margin = selected.final_score - scored[1].final_score if has_runner_up else 0.0
    reasons: list[str] = []
    status: DocumentDecisionStatus = "accepted"
    if has_runner_up and margin < config.minimum_document_margin:
        status = "provisional"
        reasons.append("low-document-margin")
    if not has_runner_up:
        reasons.append("single-surviving-document-path")
    if selected.generated_window_indexes and config.provisional_on_generated:
        status = "provisional"
        reasons.append("selected-generated-window-path")
    if selected.ambiguous_overlap_indexes and config.provisional_on_ambiguous_overlap:
        status = "provisional"
        reasons.append("ambiguous-window-overlap")
    if selected.changed_window_indexes:
        reasons.append("changed-window-paths")
    same_local_selection = selected.selection_digest == retained.selection_digest
    if same_local_selection:
        reasons.append("retained-first-pass-document")
    if selected.text != first_pass.observed_text and not selected.changed_window_indexes:
        reasons.append("overlap-emission-changed-document")
    evidence_or_output_changed = (
        not same_local_selection or _output_changed(first_pass, selected)
    )
    applied = evidence_or_output_changed and (
        status == "accepted" or config.apply_provisional
    )
    if status == "provisional" and applied:
        reasons.append("explicitly-applied-provisional")
    elif status == "provisional" and evidence_or_output_changed:
        reasons.append("provisional-not-applied")
    return DocumentDeliberationDecision(
        selected=selected,
        retained=retained,
        alternatives=tuple(scored),
        status=status,
        applied=applied,
        margin=margin,
        reasons=tuple(dict.fromkeys(reasons)),
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        config_digest=config.digest,
        local_policy_digest=local_policy.digest,
        context_digest=context.digest,
        source_audio_sha256=first_pass.source_audio_sha256,
    )


def _normalization(observed: DocumentObservedTranscript):
    normalized = deterministic_normalize(observed.text)
    if normalized:
        return NormalizedTranscript.attach(
            observed,
            text=normalized,
            mode="deterministic",
        )
    return DocumentNormalizedTranscript(
        text="",
        observed_evidence_sha256=observed.evidence_sha256,
    )


def _build_result(
    first_pass: LongformResult,
    options_by_window: Sequence[Sequence[WindowPathOption]],
    decision: DocumentDeliberationDecision,
    *,
    config: DocumentDeliberationConfig,
) -> DocumentDeliberatedResult:
    chosen = decision.selected if decision.applied else decision.retained
    if decision.applied:
        emitted_texts = chosen.emitted_texts
        receipts = chosen.overlap_receipts
    else:
        emitted_texts = tuple(segment.observed.text for segment in first_pass.segments)
        receipts = tuple(
            _unapplied_receipt(
                None if index == 0 else first_pass.segments[index - 1].observed.text,
                segment.observed.text,
                index=index,
                window=segment.window,
                previous_window=None if index == 0 else first_pass.segments[index - 1].window,
                policy=config.overlap,
            )
            for index, segment in enumerate(first_pass.segments)
        )
    segments = []
    for index, (option, emitted, receipt) in enumerate(
        zip(chosen.options, emitted_texts, receipts, strict=True)
    ):
        first_segment = first_pass.segments[index]
        observed = DocumentObservedTranscript.create(
            option=option,
            emitted_text=emitted,
            receipt=receipt,
            first_pass_segment=first_segment,
            document_decision_digest=decision.digest,
            decision=decision.status,
        )
        segments.append(
            DocumentDeliberatedSegment(
                window=first_segment.window,
                observed=observed,
                normalized=_normalization(observed),
                diagnostics={
                    **dict(first_segment.diagnostics),
                    "firstPassTopPosterior": dict(first_segment.diagnostics).get(
                        "topPosterior"
                    ),
                    "topPosterior": None,
                    "confidenceInvalidatedByDocumentDeliberation": True,
                    "documentDecisionDigest": decision.digest,
                    "windowOptionDigest": option.digest,
                    "overlapReceipt": asdict(receipt),
                    "overlapReceiptDigest": receipt.digest,
                    "fullWindowText": option.text,
                    "emittedText": emitted,
                    "changedWindowChoice": option.changed,
                    "recombinedWindowPath": option.recombined,
                    "exactSourceCandidateIds": option.exact_source_candidate_ids,
                    "documentDecisionApplied": decision.applied,
                },
                actions=first_segment.actions,
                cache_hits=first_segment.cache_hits,
            )
        )
    observed_text = join_japanese_fragments(row.observed.text for row in segments)
    normalized_text = join_japanese_fragments(row.normalized.text for row in segments)
    evidence = sha256_json(
        {
            "sourceAudioSha256": first_pass.source_audio_sha256,
            "durationMs": first_pass.duration_ms,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
            "documentDecisionDigest": decision.digest,
            "segmentEvidence": [row.observed.evidence_sha256 for row in segments],
        }
    )
    diagnostics = {
        **dict(first_pass.diagnostics),
        "provisionalWindowCount": len(segments) if decision.status == "provisional" else 0,
        "documentJointDeliberation": {
            "enabled": True,
            "decisionDigest": decision.digest,
            "status": decision.status,
            "applied": decision.applied,
            "margin": decision.margin,
            "reasons": decision.reasons,
            "selectedDocumentPathDigest": decision.selected.digest,
            "selectedDocumentSelectionDigest": decision.selected.selection_digest,
            "retainedDocumentPathDigest": decision.retained.digest,
            "retainedDocumentSelectionDigest": decision.retained.selection_digest,
            "changedWindowIndexes": decision.selected.changed_window_indexes,
            "generatedWindowIndexes": decision.selected.generated_window_indexes,
            "ambiguousOverlapIndexes": decision.selected.ambiguous_overlap_indexes,
            "firstPassEvidenceSha256": first_pass.evidence_sha256,
        },
    }
    return DocumentDeliberatedResult(
        source_name=first_pass.source_name,
        source_audio_sha256=first_pass.source_audio_sha256,
        duration_ms=first_pass.duration_ms,
        observed_text=observed_text,
        normalized_text=normalized_text,
        segments=tuple(segments),
        evidence_sha256=evidence,
        diagnostics=diagnostics,
        first_pass_evidence_sha256=first_pass.evidence_sha256,
        document_decision=decision,
        first_pass=first_pass,
    )


def apply_joint_document_deliberation(
    first_pass: LongformResult,
    *,
    config: DocumentDeliberationConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    document_scorer: GlobalSequenceScorer | None = None,
    proposal_provider: DocumentProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
    audio_path: str | Path | None = None,
) -> LongformResult | DocumentDeliberatedResult | DocumentPassThroughResult:
    config = config or DocumentDeliberationConfig()
    build_config = build_config or SemanticDeliberationConfig()
    local_policy = local_policy or DeliberationPolicy.conservative_default()
    declared_context = declared_context or DocumentContext()
    if config.require_document_scorer and document_scorer is None:
        raise ValueError("joint document deliberation requires an explicit document scorer")
    if not first_pass.segments:
        return first_pass
    try:
        options_by_window = tuple(
            _window_options(
                first_pass,
                index,
                config=config,
                build_config=build_config,
                local_policy=local_policy,
                proposal_provider=proposal_provider,
                declared_context=declared_context,
                audio_path=audio_path,
            )
            for index in range(len(first_pass.segments))
        )
        candidates = _expand_document_beam(options_by_window, config=config)
        retained = _retained_document_candidate(options_by_window, config=config)
        guarded = tuple(
            candidate
            for candidate in candidates
            if retained.mean_audio_support - candidate.mean_audio_support
            <= config.maximum_document_audio_regression
        )
        if retained.selection_digest not in {row.selection_digest for row in guarded}:
            guarded = (*guarded, retained)
        context = _document_context(first_pass, declared_context)
        scored = _score_document_candidates(
            guarded,
            document_scorer,
            context=context,
            config=config,
        )
        decision = _make_decision(
            first_pass,
            scored,
            retained,
            config=config,
            local_policy=local_policy,
            context=context,
        )
        return _build_result(
            first_pass,
            options_by_window,
            decision,
            config=config,
        )
    except Exception as exc:
        if not config.fail_closed_to_first_pass:
            raise
        return DocumentPassThroughResult.create(
            first_pass,
            error=exc,
            config=config,
            local_policy=local_policy,
        )


class JointDocumentSemanticASRTranscriber:
    def __init__(
        self,
        first_pass: SemanticASRTranscriber,
        *,
        config: DocumentDeliberationConfig | None = None,
        build_config: SemanticDeliberationConfig | None = None,
        local_policy: DeliberationPolicy | None = None,
        document_scorer: GlobalSequenceScorer | None = None,
        proposal_provider: DocumentProposalProvider | None = None,
        declared_context: DocumentContext | None = None,
    ) -> None:
        self.first_pass = first_pass
        self.document_config = config or DocumentDeliberationConfig()
        self.document_build_config = build_config or SemanticDeliberationConfig()
        self.document_local_policy = local_policy or DeliberationPolicy.conservative_default()
        self.document_scorer = document_scorer
        self.document_proposal_provider = proposal_provider
        self.document_context = declared_context or DocumentContext()
        if self.document_config.require_document_scorer and document_scorer is None:
            raise ValueError("joint document transcriber requires an explicit document scorer")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.first_pass, name)

    def transcribe(
        self,
        audio_path: str | Path,
        **kwargs: Any,
    ) -> LongformResult | DocumentDeliberatedResult | DocumentPassThroughResult:
        first_pass = self.first_pass.transcribe(audio_path, **kwargs)
        return apply_joint_document_deliberation(
            first_pass,
            config=self.document_config,
            build_config=self.document_build_config,
            local_policy=self.document_local_policy,
            document_scorer=self.document_scorer,
            proposal_provider=self.document_proposal_provider,
            declared_context=self.document_context,
            audio_path=audio_path,
        )


def with_joint_document_deliberation(
    transcriber: SemanticASRTranscriber,
    *,
    document_scorer: GlobalSequenceScorer,
    config: DocumentDeliberationConfig | None = None,
    build_config: SemanticDeliberationConfig | None = None,
    local_policy: DeliberationPolicy | None = None,
    proposal_provider: DocumentProposalProvider | None = None,
    declared_context: DocumentContext | None = None,
) -> JointDocumentSemanticASRTranscriber:
    return JointDocumentSemanticASRTranscriber(
        transcriber,
        config=config,
        build_config=build_config,
        local_policy=local_policy,
        document_scorer=document_scorer,
        proposal_provider=proposal_provider,
        declared_context=declared_context,
    )
