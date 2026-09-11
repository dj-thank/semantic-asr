"""Selective phone/mora proposal acquisition for document-lattice contradiction spans.

The provider loads the original mono recording once, extracts only policy-selected ambiguous spans,
runs frozen candidate-independent phone/mora backends, and converts their calibrated CTC evidence
into source-audio-bound ``VerifiedSpanProposal`` objects. Every proposal records the exact sample
slice and posterior artifacts that produced it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .audio_posterior_adapters import (
    DualPosteriorExtractor,
    PosteriorBundle,
    canonical_audio_sha256,
)
from .contracts import sha256_json
from .deliberation_evidence import UtilityCalibrationProfile, _is_sha256, _strict_float
from .deliberation_lattice import DocumentContext, DeliberationSpan
from .longform import LongformSegment
from .phonetic_bridge import (
    FrozenPronunciationLexicon,
    PhoneticBridgeConfig,
    propose_text_from_pronunciation,
)
from .semantic_deliberation import SemanticDeliberationBuild, VerifiedSpanProposal


@dataclass(frozen=True, slots=True)
class LoadedMonoAudio:
    samples: tuple[float, ...]
    sample_rate: int
    source_audio_sha256: str
    source_name: str = "audio"

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("loaded mono audio must not be empty")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if not _is_sha256(self.source_audio_sha256):
            raise ValueError("source_audio_sha256 must be a SHA-256 value")
        normalized = []
        for sample in self.samples:
            value = _strict_float(sample, name="audio sample")
            normalized.append(value)
        object.__setattr__(self, "samples", tuple(normalized))

    @property
    def duration_ms(self) -> int:
        return round(len(self.samples) * 1000 / self.sample_rate)


class MonoAudioLoader(Protocol):
    def load(self, path: str | Path) -> LoadedMonoAudio: ...


class SoundFileMonoAudioLoader:
    """Strict file loader: mono only, no implicit resampling or loudness transformation."""

    def load(self, path: str | Path) -> LoadedMonoAudio:
        try:
            import soundfile
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SoundFileMonoAudioLoader requires soundfile") from exc
        audio_path = Path(path)
        data, sample_rate = soundfile.read(audio_path, dtype="float32", always_2d=True)
        if data.shape[1] != 1:
            raise ValueError("phonetic span extraction requires explicitly mono audio")
        return LoadedMonoAudio(
            samples=tuple(float(value) for value in data[:, 0]),
            sample_rate=int(sample_rate),
            source_audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            source_name=audio_path.name,
        )


class SpanLexiconProvider(Protocol):
    """Return a lexicon frozen independently of evaluation references and target audio labels."""

    def __call__(
        self,
        *,
        span: DeliberationSpan,
        context: DocumentContext,
        build: SemanticDeliberationBuild,
    ) -> FrozenPronunciationLexicon: ...


@dataclass(frozen=True, slots=True)
class StaticSpanLexiconProvider:
    lexicon: FrozenPronunciationLexicon

    def __call__(
        self,
        *,
        span: DeliberationSpan,
        context: DocumentContext,
        build: SemanticDeliberationBuild,
    ) -> FrozenPronunciationLexicon:
        del span, context, build
        return self.lexicon


@dataclass(frozen=True, slots=True)
class PhoneticSpanProviderConfig:
    maximum_spans: int = 8
    minimum_factor_weight: float = 0.0
    minimum_semantic_criticality: float = 0.0
    padding_ms: int = 80
    maximum_span_duration_ms: int = 4_000
    proposals_per_span: int = 4
    minimum_combined_utility: float = -0.20
    skip_existing_surface_without_new_channel: bool = True
    fail_closed_per_span: bool = True
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "maximum_spans",
            "padding_ms",
            "maximum_span_duration_ms",
            "proposals_per_span",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.maximum_spans < 1 or self.proposals_per_span < 1:
            raise ValueError("maximum_spans and proposals_per_span must be positive")
        if self.maximum_span_duration_ms < 1:
            raise ValueError("maximum_span_duration_ms must be positive")
        for name in (
            "minimum_factor_weight",
            "minimum_semantic_criticality",
            "minimum_combined_utility",
        ):
            value = _strict_float(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if not 0.0 <= self.minimum_factor_weight <= 1.0:
            raise ValueError("minimum_factor_weight must be in [0, 1]")
        if not 0.0 <= self.minimum_semantic_criticality <= 1.0:
            raise ValueError("minimum_semantic_criticality must be in [0, 1]")
        if not -1.0 <= self.minimum_combined_utility <= 1.0:
            raise ValueError("minimum_combined_utility must be in [-1, 1]")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class SpanAudioReceipt:
    span_id: str
    source_audio_sha256: str
    source_name: str
    requested_start_ms: int
    requested_end_ms: int
    extracted_start_ms: int
    extracted_end_ms: int
    sample_start: int
    sample_end: int
    sample_rate: int
    canonical_clip_sha256: str
    posterior_bundle_digest: str
    extractor_digest: str
    provider_config_digest: str

    def __post_init__(self) -> None:
        if not self.span_id or not self.source_name:
            raise ValueError("span audio receipt requires span ID and source name")
        if not (
            0 <= self.requested_start_ms < self.requested_end_ms
            and 0 <= self.extracted_start_ms < self.extracted_end_ms
        ):
            raise ValueError("span audio receipt has an invalid time range")
        if not 0 <= self.sample_start < self.sample_end:
            raise ValueError("span audio receipt has an invalid sample range")
        if isinstance(self.sample_rate, bool) or self.sample_rate < 1:
            raise ValueError("span audio receipt sample rate must be positive")
        for digest in (
            self.source_audio_sha256,
            self.canonical_clip_sha256,
            self.posterior_bundle_digest,
            self.extractor_digest,
            self.provider_config_digest,
        ):
            if not _is_sha256(digest):
                raise ValueError("span audio receipt contains an invalid digest")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class SpanProposalFailure:
    span_id: str
    error_type: str
    error_message_sha256: str

    def __post_init__(self) -> None:
        if not self.span_id or not self.error_type:
            raise ValueError("span proposal failure requires span ID and error type")
        if not _is_sha256(self.error_message_sha256):
            raise ValueError("error_message_sha256 must be a SHA-256 value")

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(slots=True)
class SelectivePhoneticSpanProposalProvider:
    extractor: DualPosteriorExtractor
    lexicon_provider: SpanLexiconProvider
    phone_calibration: UtilityCalibrationProfile | None = None
    mora_calibration: UtilityCalibrationProfile | None = None
    audio_loader: MonoAudioLoader = field(default_factory=SoundFileMonoAudioLoader)
    config: PhoneticSpanProviderConfig = field(default_factory=PhoneticSpanProviderConfig)
    _audio_cache: dict[str, LoadedMonoAudio] = field(default_factory=dict, init=False, repr=False)
    receipts: dict[str, SpanAudioReceipt] = field(default_factory=dict, init=False)
    failures: dict[str, SpanProposalFailure] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.extractor.phone is not None and self.phone_calibration is None:
            raise ValueError("phone extractor requires a phone utility calibration profile")
        if self.extractor.mora is not None and self.mora_calibration is None:
            raise ValueError("mora extractor requires a mora utility calibration profile")
        if self.phone_calibration is not None and self.phone_calibration.channel != "phone":
            raise ValueError("phone calibration must produce the phone channel")
        if self.mora_calibration is not None and self.mora_calibration.channel != "mora":
            raise ValueError("mora calibration must produce the mora channel")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "phoneExtractorDigest": (
                    None if self.extractor.phone is None else self.extractor.phone.digest
                ),
                "moraExtractorDigest": (
                    None if self.extractor.mora is None else self.extractor.mora.digest
                ),
                "phoneCalibrationDigest": (
                    None if self.phone_calibration is None else self.phone_calibration.digest
                ),
                "moraCalibrationDigest": (
                    None if self.mora_calibration is None else self.mora_calibration.digest
                ),
                "providerConfigDigest": self.config.digest,
                "implementation": "selective-phonetic-span-provider-v1",
            }
        )

    def _load(self, path: str | Path) -> LoadedMonoAudio:
        key = str(Path(path).resolve())
        if key not in self._audio_cache:
            self._audio_cache[key] = self.audio_loader.load(path)
        return self._audio_cache[key]

    def _eligible_spans(self, build: SemanticDeliberationBuild) -> tuple[DeliberationSpan, ...]:
        rows = []
        for span in build.lattice.spans:
            factor = float(span.metadata.get("factorWeight", 0.0))
            criticality = float(span.metadata.get("semanticCriticality", 0.0))
            contradiction = bool(span.metadata.get("isContradiction", False))
            if not contradiction:
                continue
            if factor < self.config.minimum_factor_weight:
                continue
            if criticality < self.config.minimum_semantic_criticality:
                continue
            rows.append((criticality, factor, span.start_ms, span))
        rows.sort(key=lambda row: (-row[0], -row[1], row[2], row[3].span_id))
        return tuple(row[3] for row in rows[: self.config.maximum_spans])

    def _extract_span(
        self,
        audio: LoadedMonoAudio,
        span: DeliberationSpan,
        *,
        source_audio_sha256: str,
    ) -> tuple[PosteriorBundle, SpanAudioReceipt]:
        start_ms = max(0, span.start_ms - self.config.padding_ms)
        end_ms = min(audio.duration_ms, span.end_ms + self.config.padding_ms)
        if end_ms <= start_ms:
            raise ValueError("span is outside the loaded audio duration")
        if end_ms - start_ms > self.config.maximum_span_duration_ms:
            center = (span.start_ms + span.end_ms) // 2
            half = self.config.maximum_span_duration_ms // 2
            start_ms = max(0, center - half)
            end_ms = min(audio.duration_ms, start_ms + self.config.maximum_span_duration_ms)
            start_ms = max(0, end_ms - self.config.maximum_span_duration_ms)
        sample_start = max(0, math.floor(start_ms * audio.sample_rate / 1000))
        sample_end = min(
            len(audio.samples),
            math.ceil(end_ms * audio.sample_rate / 1000),
        )
        if sample_end <= sample_start:
            raise ValueError("span maps to an empty sample range")
        clip = audio.samples[sample_start:sample_end]
        bundle = self.extractor.extract(
            clip,
            sample_rate=audio.sample_rate,
            source_audio_sha256=source_audio_sha256,
        )
        extractor_digest = sha256_json(
            {
                "phone": None if self.extractor.phone is None else self.extractor.phone.digest,
                "mora": None if self.extractor.mora is None else self.extractor.mora.digest,
            }
        )
        receipt = SpanAudioReceipt(
            span_id=span.span_id,
            source_audio_sha256=source_audio_sha256,
            source_name=audio.source_name,
            requested_start_ms=span.start_ms,
            requested_end_ms=span.end_ms,
            extracted_start_ms=start_ms,
            extracted_end_ms=end_ms,
            sample_start=sample_start,
            sample_end=sample_end,
            sample_rate=audio.sample_rate,
            canonical_clip_sha256=canonical_audio_sha256(clip, audio.sample_rate),
            posterior_bundle_digest=bundle.digest,
            extractor_digest=extractor_digest,
            provider_config_digest=self.config.digest,
        )
        return bundle, receipt

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
        del segment_index, segment
        if audio_path is None:
            raise ValueError("phonetic span proposal provider requires the original audio path")
        if build.lattice.source_audio_sha256 != source_audio_sha256:
            raise ValueError("deliberation build is bound to different source audio")
        audio = self._load(audio_path)
        if audio.source_audio_sha256 != source_audio_sha256:
            raise ValueError("loaded audio SHA-256 does not match the first-pass recording")
        output: dict[str, tuple[VerifiedSpanProposal, ...]] = {}
        for span in self._eligible_spans(build):
            try:
                bundle, receipt = self._extract_span(
                    audio,
                    span,
                    source_audio_sha256=source_audio_sha256,
                )
                lexicon = self.lexicon_provider(span=span, context=context, build=build)
                proposals = propose_text_from_pronunciation(
                    lexicon,
                    phone_posterior=bundle.phone,
                    mora_posterior=bundle.mora,
                    phone_calibration=self.phone_calibration,
                    mora_calibration=self.mora_calibration,
                    config=PhoneticBridgeConfig(
                        top_k=self.config.proposals_per_span,
                        minimum_combined_utility=self.config.minimum_combined_utility,
                    ),
                )
                existing = {
                    arc.text: set(arc.independent_audio_channels) for arc in span.arcs
                }
                verified = []
                for proposal in proposals:
                    proposed_channels = {utility.channel for utility in proposal.utilities}
                    if (
                        self.config.skip_existing_surface_without_new_channel
                        and proposal.text in existing
                        and proposed_channels.issubset(existing[proposal.text])
                    ):
                        continue
                    verified.append(
                        VerifiedSpanProposal(
                            proposal_id=(
                                f"phonetic:{span.span_id}:{proposal.candidate_id}:"
                                f"{receipt.digest[:12]}"
                            ),
                            text=proposal.text,
                            utilities=proposal.utilities,
                            source_audio_sha256=source_audio_sha256,
                            origin="phonetic-proposal",
                            pronunciation_key=proposal.pronunciation_key,
                            source_candidate_ids=(),
                            observed_eligible=True,
                            metadata={
                                "providerDigest": self.digest,
                                "spanAudioReceiptDigest": receipt.digest,
                                "spanAudioReceipt": asdict(receipt),
                                "posteriorBundleDigest": bundle.digest,
                                "lexiconDigest": lexicon.digest,
                                "phoneticProposalCandidateId": proposal.candidate_id,
                                "phoneScoreDigest": (
                                    None
                                    if proposal.phone_score is None
                                    else proposal.phone_score.evidence.metadata.get(
                                        "pronunciationDigest"
                                    )
                                ),
                                "moraScoreDigest": (
                                    None
                                    if proposal.mora_score is None
                                    else proposal.mora_score.evidence.metadata.get(
                                        "pronunciationDigest"
                                    )
                                ),
                            },
                        )
                    )
                if verified:
                    output[span.span_id] = tuple(verified)
                self.receipts[span.span_id] = receipt
            except Exception as exc:
                failure = SpanProposalFailure(
                    span_id=span.span_id,
                    error_type=type(exc).__name__,
                    error_message_sha256=hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                )
                self.failures[span.span_id] = failure
                if not self.config.fail_closed_per_span:
                    raise
        return output
