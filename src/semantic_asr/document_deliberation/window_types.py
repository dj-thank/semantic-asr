"""Per-window path types for document deliberation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..audio import require_integer
from ..contracts import sha256_json
from ..deliberation_evidence import _is_sha256, _strict_float
from ..global_deliberation import PathHypothesis
from ..longform import LongformSegment
from ..semantic_deliberation import SemanticDeliberationBuild
from .context_types import FrozenWindowContext


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
