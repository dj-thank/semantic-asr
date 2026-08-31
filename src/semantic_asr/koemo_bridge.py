from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ChannelKind = Literal["microphone", "system", "imported", "mixed"]
TranscriptAuthority = Literal["live-provisional", "final-observed", "normalized-derivative"]


@dataclass(frozen=True, slots=True)
class KoemoChannelSpan:
    span_id: str
    channel: ChannelKind
    start_ms: int
    end_ms: int
    audio_sha256: str
    source_recording_sha256: str
    aec_applied: bool = False
    aec_configuration_digest: str | None = None
    speaker_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("span_id is required")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("invalid Koemo span")
        for name, value in (
            ("audio_sha256", self.audio_sha256),
            ("source_recording_sha256", self.source_recording_sha256),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in value
            ):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if self.aec_applied and not self.aec_configuration_digest:
            raise ValueError("AEC provenance is required when AEC was applied")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class KoemoTranscriptEvent:
    event_id: str
    span_id: str
    text: str
    authority: TranscriptAuthority
    evidence_digest: str | None = None
    observed_evidence_digest: str | None = None
    backend: str | None = None
    backend_revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.span_id or not self.text:
            raise ValueError("event ID, span ID and text are required")
        if self.authority == "live-provisional" and self.observed_evidence_digest is not None:
            raise ValueError("live preview cannot claim observed evidence")
        if self.authority == "final-observed" and not self.evidence_digest:
            raise ValueError("final observed events require evidence_digest")
        if self.authority == "normalized-derivative" and not self.observed_evidence_digest:
            raise ValueError("normalized derivatives must reference observed evidence")


@dataclass(frozen=True, slots=True)
class KoemoMeetingEvidence:
    meeting_id: str
    spans: tuple[KoemoChannelSpan, ...]
    events: tuple[KoemoTranscriptEvent, ...]
    semantic_asr_revision: str
    configuration_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.meeting_id or not self.semantic_asr_revision or not self.configuration_digest:
            raise ValueError("meeting ID, Semantic ASR revision and configuration digest are required")
        span_ids = {span.span_id for span in self.spans}
        if len(span_ids) != len(self.spans):
            raise ValueError("Koemo span IDs must be unique")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ValueError("Koemo event IDs must be unique")
        if any(event.span_id not in span_ids for event in self.events):
            raise ValueError("transcript event references unknown span")
        observed = {
            event.evidence_digest
            for event in self.events
            if event.authority == "final-observed" and event.evidence_digest
        }
        if any(
            event.observed_evidence_digest not in observed
            for event in self.events
            if event.authority == "normalized-derivative"
        ):
            raise ValueError("normalized event references unknown observed evidence")

    @property
    def digest(self) -> str:
        payload = {
            "meetingId": self.meeting_id,
            "spans": [asdict(span) for span in self.spans],
            "events": [asdict(event) for event in self.events],
            "semanticAsrRevision": self.semantic_asr_revision,
            "configurationDigest": self.configuration_digest,
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def authoritative_events(self) -> tuple[KoemoTranscriptEvent, ...]:
        return tuple(
            event for event in self.events if event.authority == "final-observed"
        )

    def normalized_events(self) -> tuple[KoemoTranscriptEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.authority == "normalized-derivative"
        )


@dataclass(frozen=True, slots=True)
class KoemoIntegrationPolicy:
    live_text_is_authoritative: bool = False
    allow_regex_on_observed: bool = False
    preserve_channel_identity: bool = True
    require_aec_provenance: bool = True
    require_audio_digest: bool = True

    def __post_init__(self) -> None:
        if self.live_text_is_authoritative:
            raise ValueError("Koemo live text must remain provisional")
        if self.allow_regex_on_observed:
            raise ValueError("regex correction cannot mutate observed evidence")


def live_event(
    *,
    span_id: str,
    text: str,
    backend: str,
    backend_revision: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> KoemoTranscriptEvent:
    payload = {
        "spanId": span_id,
        "text": text,
        "backend": backend,
        "revision": backend_revision,
        "authority": "live-provisional",
    }
    return KoemoTranscriptEvent(
        event_id="live-" + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        span_id=span_id,
        text=text,
        authority="live-provisional",
        backend=backend,
        backend_revision=backend_revision,
        metadata=dict(metadata or {}),
    )


def observed_event(
    *,
    span_id: str,
    text: str,
    evidence_digest: str,
    backend: str,
    backend_revision: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> KoemoTranscriptEvent:
    if not evidence_digest:
        raise ValueError("evidence_digest is required")
    payload = {
        "spanId": span_id,
        "text": text,
        "evidenceDigest": evidence_digest,
        "backend": backend,
        "revision": backend_revision,
        "authority": "final-observed",
    }
    return KoemoTranscriptEvent(
        event_id="observed-" + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        span_id=span_id,
        text=text,
        authority="final-observed",
        evidence_digest=evidence_digest,
        backend=backend,
        backend_revision=backend_revision,
        metadata=dict(metadata or {}),
    )


def normalized_event(
    *,
    span_id: str,
    text: str,
    observed_evidence_digest: str,
    normalizer: str,
    normalizer_revision: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> KoemoTranscriptEvent:
    if not observed_evidence_digest:
        raise ValueError("observed_evidence_digest is required")
    payload = {
        "spanId": span_id,
        "text": text,
        "observedEvidenceDigest": observed_evidence_digest,
        "normalizer": normalizer,
        "revision": normalizer_revision,
        "authority": "normalized-derivative",
    }
    return KoemoTranscriptEvent(
        event_id="normalized-" + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        span_id=span_id,
        text=text,
        authority="normalized-derivative",
        observed_evidence_digest=observed_evidence_digest,
        backend=normalizer,
        backend_revision=normalizer_revision,
        metadata=dict(metadata or {}),
    )


def channel_span_from_samples(
    *,
    span_id: str,
    channel: ChannelKind,
    start_ms: int,
    sample_rate: int,
    sample_count: int,
    audio_sha256: str,
    source_recording_sha256: str,
    aec_applied: bool = False,
    aec_configuration_digest: str | None = None,
    speaker_label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> KoemoChannelSpan:
    if sample_rate < 1 or sample_count < 1:
        raise ValueError("sample_rate and sample_count must be positive")
    duration_ms = round(sample_count / sample_rate * 1000)
    if duration_ms < 1 or not math.isfinite(float(duration_ms)):
        raise ValueError("audio span duration is invalid")
    return KoemoChannelSpan(
        span_id=span_id,
        channel=channel,
        start_ms=start_ms,
        end_ms=start_ms + duration_ms,
        audio_sha256=audio_sha256,
        source_recording_sha256=source_recording_sha256,
        aec_applied=aec_applied,
        aec_configuration_digest=aec_configuration_digest,
        speaker_label=speaker_label,
        metadata=dict(metadata or {}),
    )
