"""Ambiguity-only phone/mora proposal provider for semantic deliberation spans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..deliberation_evidence import UtilityCalibrationProfile, _is_sha256
from ..deliberation_lattice import DeliberationSpan, DocumentContext
from ..longform import LongformSegment
from ..phonetic_bridge import (
    FrozenPronunciationLexicon,
    PhoneticBridgeConfig,
    propose_text_from_pronunciation,
)
from ..phonetic_evidence import PosteriorSequence
from ..semantic_deliberation import SemanticDeliberationBuild, VerifiedSpanProposal


class PhoneMoraPosteriorRuntime(Protocol):
    @property
    def profile_digest(self) -> str: ...

    @property
    def source(self) -> str: ...

    def infer(
        self,
        audio_path: str | Path,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        expected_source_audio_sha256: str | None = None,
    ) -> tuple[PosteriorSequence, PosteriorSequence]: ...


SpanLexiconProvider = Callable[
    [DeliberationSpan, DocumentContext],
    FrozenPronunciationLexicon | None,
]


@dataclass(frozen=True, slots=True)
class PhoneticProposalProviderConfig:
    maximum_spans: int = 8
    maximum_proposals_per_span: int = 6
    padding_ms: int = 120
    minimum_factor_weight: float = 0.0
    require_both_heads: bool = False
    include_existing_surfaces: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in ("maximum_spans", "maximum_proposals_per_span"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.padding_ms, bool) or not isinstance(self.padding_ms, int):
            raise TypeError("padding_ms must be an integer")
        if self.padding_ms < 0:
            raise ValueError("padding_ms must be non-negative")
        if isinstance(self.minimum_factor_weight, bool):
            raise TypeError("minimum_factor_weight must be a real number")
        if not 0.0 <= float(self.minimum_factor_weight) <= 1.0:
            raise ValueError("minimum_factor_weight must be in [0, 1]")
        if not isinstance(self.require_both_heads, bool):
            raise TypeError("require_both_heads must be a boolean")
        if not isinstance(self.include_existing_surfaces, bool):
            raise TypeError("include_existing_surfaces must be a boolean")
        object.__setattr__(self, "minimum_factor_weight", float(self.minimum_factor_weight))


@dataclass(slots=True)
class SourceAudioPhoneticProposalProvider:
    runtime: PhoneMoraPosteriorRuntime
    lexicon_provider: SpanLexiconProvider
    phone_calibration: UtilityCalibrationProfile
    mora_calibration: UtilityCalibrationProfile
    config: PhoneticProposalProviderConfig = PhoneticProposalProviderConfig()

    def __post_init__(self) -> None:
        if not _is_sha256(self.runtime.profile_digest):
            raise ValueError("phonetic runtime profile_digest must be SHA-256")
        if self.phone_calibration.channel != "phone":
            raise ValueError("phone calibration must emit the phone utility channel")
        if self.mora_calibration.channel != "mora":
            raise ValueError("mora calibration must emit the mora utility channel")

    def _selected_spans(self, build: SemanticDeliberationBuild) -> tuple[DeliberationSpan, ...]:
        rows = [
            span
            for span in build.lattice.spans
            if bool(span.metadata.get("isContradiction"))
            and float(span.metadata.get("factorWeight", 0.0))
            >= self.config.minimum_factor_weight
        ]
        rows.sort(
            key=lambda span: (
                -float(span.metadata.get("semanticCriticality", 0.0)),
                -float(span.metadata.get("posteriorAmbiguity", 0.0)),
                -float(span.metadata.get("factorWeight", 0.0)),
                span.index,
            )
        )
        return tuple(rows[: self.config.maximum_spans])

    def __call__(
        self,
        *,
        audio_path: str | Path | None,
        segment_index: int,
        segment: LongformSegment,
        build: SemanticDeliberationBuild,
        context: DocumentContext,
        source_audio_sha256: str,
    ) -> Mapping[str, Sequence[VerifiedSpanProposal]]:
        if audio_path is None:
            raise ValueError("phonetic proposal provider requires the source audio path")
        if not _is_sha256(source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        if build.lattice.source_audio_sha256 != source_audio_sha256:
            raise ValueError("deliberation lattice belongs to different source audio")
        output: dict[str, tuple[VerifiedSpanProposal, ...]] = {}
        for span in self._selected_spans(build):
            lexicon = self.lexicon_provider(span, context)
            if lexicon is None:
                continue
            start_ms = max(segment.window.start_ms, span.start_ms - self.config.padding_ms)
            end_ms = min(segment.window.end_ms, span.end_ms + self.config.padding_ms)
            if end_ms <= start_ms:
                raise ValueError("phonetic proposal crop has a non-positive duration")
            phone_posterior, mora_posterior = self.runtime.infer(
                audio_path,
                start_ms=start_ms,
                end_ms=end_ms,
                expected_source_audio_sha256=source_audio_sha256,
            )
            if phone_posterior.source_audio_sha256 != source_audio_sha256:
                raise ValueError("phone posterior is bound to different source audio")
            if mora_posterior.source_audio_sha256 != source_audio_sha256:
                raise ValueError("mora posterior is bound to different source audio")
            if self.config.require_both_heads and (
                not phone_posterior.frames or not mora_posterior.frames
            ):
                raise ValueError("phonetic provider requires both non-empty posterior heads")
            proposals = propose_text_from_pronunciation(
                lexicon,
                phone_posterior=phone_posterior,
                mora_posterior=mora_posterior,
                phone_calibration=self.phone_calibration,
                mora_calibration=self.mora_calibration,
                config=PhoneticBridgeConfig(
                    top_k=self.config.maximum_proposals_per_span,
                ),
            )
            existing = {arc.text for arc in span.arcs}
            verified: list[VerifiedSpanProposal] = []
            for proposal in proposals:
                if not self.config.include_existing_surfaces and proposal.text in existing:
                    continue
                verified.append(
                    VerifiedSpanProposal(
                        proposal_id=(
                            f"dual-ctc-{segment_index:04d}-{span.index:04d}-"
                            f"{proposal.candidate_id}"
                        ),
                        text=proposal.text,
                        utilities=proposal.utilities,
                        source_audio_sha256=source_audio_sha256,
                        origin="phonetic-proposal",
                        pronunciation_key=proposal.pronunciation_key,
                        source_candidate_ids=(),
                        observed_eligible=True,
                        metadata={
                            "runtimeProfileDigest": self.runtime.profile_digest,
                            "runtimeSource": self.runtime.source,
                            "phonePosteriorDigest": phone_posterior.digest,
                            "moraPosteriorDigest": mora_posterior.digest,
                            "lexiconDigest": lexicon.digest,
                            "cropStartMs": start_ms,
                            "cropEndMs": end_ms,
                            "combinedPhoneticUtility": proposal.combined_utility,
                        },
                    )
                )
            if verified:
                output[span.span_id] = tuple(verified)
        return output
