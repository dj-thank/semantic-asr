"""One-call transcription facade with immutable runtime profiles.

``docs/ARCHITECTURE_ROADMAP_2026-09-02.md`` selects a default-caller facade backed by the
existing long-form orchestrator. A caller that only wants a transcript should not have to
learn windowing, path pooling, score domains, the loop guard, or evidence hashing::

    from semantic_asr.api import transcribe

    result = transcribe("meeting.wav", profile="cpu-ja-v1")
    print(result.observed_text)
    for segment in result.segments:
        print(segment.start_ms, segment.end_ms, segment.normalized)

Every profile is a frozen, named, digestible configuration. Backend knobs never leak into the
call signature; a new combination is a new profile. ``transcribe_segments`` mirrors the
Koemo ``Transcriber.transcribe_segments`` contract so the meeting recorder can swap its direct
faster-whisper call for the evidence-preserving core without changing its callers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .adapters import ASRAdapter
from .candidate_pool import lenient_surface_key
from .longform import LongformResult, SemanticASRTranscriber
from .outputs import write_outputs
from .pipeline import EffortName, effort_profile

ProgressCallback = Callable[[str], None]
DeviceName = Literal["cpu", "cuda", "auto"]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Immutable named runtime configuration.

    Values here are the measured defaults from ``docs/RESEARCH_2026-09-02.md``: the padded
    CTranslate2 path with the loop guard enabled and no learned reranker, second ear, or
    teacher, because none of those improved the locked test split.
    """

    name: str
    description: str
    model: str = "large-v3-turbo"
    device: DeviceName = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    hypotheses: int = 5
    patience: float = 1.0
    effort: EffortName = "ultra-light"
    window_ms: int = 28_000
    overlap_ms: int = 1_200
    loop_guard: bool = True
    language: str = "ja"
    model_revision: str | None = None
    # Platt mapping (a, b) from logit(top posterior) to P(lenient CER == 0) fitted on the
    # ReazonSpeech calibration split (n=119, 1-30 s clips, lenient surface pooling, default
    # fusion). Test split: ECE 0.180 -> 0.029, AUROC 0.80. None disables confidence.
    confidence_calibration: tuple[float, float] | None = (0.1211, -0.6687)
    confidence_note: str = (
        "calibrated on 1-30 s clips of ReazonSpeech (2026-09-03); long-form windows are "
        "28 s spans, so treat the value as a relative ranking signal, not a guarantee"
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name is required")
        if self.beam_size < 1 or self.hypotheses < 1:
            raise ValueError("beam_size and hypotheses must be positive")
        if self.hypotheses > self.beam_size:
            raise ValueError("hypotheses cannot exceed beam_size")
        if self.window_ms <= 0 or self.overlap_ms < 0 or self.overlap_ms >= self.window_ms:
            raise ValueError("window_ms must be positive and larger than overlap_ms")
        if self.window_ms > 30_000:
            raise ValueError("Whisper windows cannot exceed 30 s")
        effort_profile(self.effort)

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


PROFILES: dict[str, RuntimeProfile] = {
    "cpu-ja-v1": RuntimeProfile(
        name="cpu-ja-v1",
        description=(
            "Japanese, CPU int8 large-v3-turbo, beam 5, padded window, loop guard; the "
            "measured default for machines without a supported GPU."
        ),
    ),
    "cpu-ja-quality-v1": RuntimeProfile(
        name="cpu-ja-quality-v1",
        description=(
            "Japanese, CPU int8 large-v3-turbo, beam 12 with patience 1.4 and up to 12 "
            "candidates for offline batch jobs where wall-clock is not critical."
        ),
        beam_size=12,
        hypotheses=12,
        patience=1.4,
        effort="cpu-quality",
    ),
    "gpu-ja-v1": RuntimeProfile(
        name="gpu-ja-v1",
        description=(
            "Japanese, CUDA float16 large-v3-turbo, beam 12; requires a CUDA runtime that "
            "CTranslate2 can load. Not measured on this repository's CPU host."
        ),
        device="cuda",
        compute_type="float16",
        beam_size=12,
        hypotheses=12,
        patience=1.4,
        effort="cpu-quality",
    ),
}


def runtime_profile(name: str | RuntimeProfile) -> RuntimeProfile:
    if isinstance(name, RuntimeProfile):
        return name
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown runtime profile: {name!r}; known: {sorted(PROFILES)}") from exc


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    index: int
    start_ms: int
    end_ms: int
    observed: str
    normalized: str
    status: str
    confidence: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0


@dataclass(frozen=True, slots=True)
class Utterance:
    """Timestamp-token utterance inside a window, in absolute recording time."""

    index: int
    segment_index: int
    start_ms: int
    end_ms: int
    text: str
    status: str
    confidence: float | None = None

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0


def _selected_candidate(observed: Any) -> Any:
    candidates = getattr(observed, "candidates", None) or ()
    wanted = getattr(observed, "selected_candidate_id", None)
    for candidate in candidates:
        identifier = getattr(candidate, "candidate_id", None)
        if identifier is None and isinstance(candidate, Mapping):
            identifier = candidate.get("candidate_id")
        if identifier == wanted:
            return candidate
    return None


def _candidate_spans(candidate: Any) -> list[Mapping[str, Any]]:
    metadata = getattr(candidate, "metadata", None)
    if metadata is None and isinstance(candidate, Mapping):
        metadata = candidate.get("metadata")
    spans = (metadata or {}).get("utteranceSpans") if isinstance(metadata, Mapping) else None
    return [span for span in (spans or []) if isinstance(span, Mapping) and span.get("text")]


def utterances_from_segments(
    longform: LongformResult,
    segments: Sequence[TranscriptSegment],
    *,
    overlap_tolerance_ms: int = 250,
) -> tuple[Utterance, ...]:
    """Flatten window-level utterance spans into absolute-time utterances.

    Windows overlap by ``overlap_ms``; an utterance that starts before the previous kept
    utterance ended (minus a small tolerance) is a duplicate of overlap audio and is dropped.
    A window whose selected candidate carries no spans contributes itself as one utterance.
    """

    output: list[Utterance] = []
    last_end = -1
    for segment, raw in zip(segments, longform.segments, strict=True):
        spans = _candidate_spans(_selected_candidate(raw.observed))
        window_start = int(raw.window.start_ms)
        window_end = int(raw.window.end_ms)
        rows: list[tuple[int, int, str]] = []
        for span in spans:
            start = window_start + int(span.get("startMs") or 0)
            end_value = span.get("endMs")
            end = window_end if end_value is None else window_start + int(end_value)
            rows.append((start, min(max(end, start), window_end), str(span["text"]).strip()))
        if not rows:
            text = segment.observed.strip()
            if text:
                rows.append((window_start, window_end, text))
        for start, end, text in rows:
            if not text:
                continue
            if start < last_end - overlap_tolerance_ms and output:
                previous = output[-1]
                previous_key = lenient_surface_key(previous.text)
                current_key = lenient_surface_key(text)
                if current_key and current_key in previous_key:
                    continue  # duplicate of overlap audio already emitted
                if previous_key and current_key.startswith(previous_key):
                    # The previous window was cut mid-utterance; this row completes it.
                    output.pop()
                    start = min(start, previous.start_ms)
                else:
                    start = max(start, last_end)
            output.append(
                Utterance(
                    index=len(output) + 1,
                    segment_index=segment.index,
                    start_ms=start,
                    end_ms=end,
                    text=text,
                    status=segment.status,
                    confidence=segment.confidence,
                )
            )
            last_end = max(last_end, end)
    return tuple(output)


def render_utterance_srt(utterances: Sequence[Utterance]) -> str:
    def timecode(milliseconds: int) -> str:
        milliseconds = max(0, int(milliseconds))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    lines: list[str] = []
    for utterance in utterances:
        lines.extend(
            [
                str(utterance.index),
                f"{timecode(utterance.start_ms)} --> {timecode(utterance.end_ms)}",
                utterance.text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    profile: RuntimeProfile
    source_name: str
    source_audio_sha256: str
    duration_ms: int
    observed_text: str
    normalized_text: str
    segments: tuple[TranscriptSegment, ...]
    evidence_sha256: str
    provenance: dict[str, Any]
    diagnostics: dict[str, Any]
    longform: LongformResult
    utterances: tuple[Utterance, ...] = ()

    @property
    def provisional_segment_count(self) -> int:
        return sum(1 for segment in self.segments if segment.status != "accepted")

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": asdict(self.profile),
            "profileDigest": self.profile.digest,
            "sourceName": self.source_name,
            "sourceAudioSha256": self.source_audio_sha256,
            "durationMs": self.duration_ms,
            "observedText": self.observed_text,
            "normalizedText": self.normalized_text,
            "segments": [asdict(segment) for segment in self.segments],
            "utterances": [asdict(utterance) for utterance in self.utterances],
            "evidenceSha256": self.evidence_sha256,
            "provenance": self.provenance,
            "diagnostics": self.diagnostics,
        }

    def write(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
        formats: set[str] | None = None,
    ) -> dict[str, str]:
        """Write the standard artifact set plus an utterance-level SRT and facade JSON."""

        outputs = write_outputs(self.longform, output_dir, overwrite=overwrite, formats=formats)
        root = Path(output_dir)
        stem = Path(self.source_name).stem
        if self.utterances and (formats is None or "srt" in formats):
            target = root / f"{stem}.utterances.srt"
            if overwrite or not target.exists():
                target.write_text(render_utterance_srt(self.utterances), encoding="utf-8")
                outputs["utterances_srt"] = str(target)
        if formats is None or "json" in formats:
            target = root / f"{stem}.transcript.json"
            if overwrite or not target.exists():
                target.write_text(
                    json.dumps(self.as_dict(), ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
                outputs["transcript_json"] = str(target)
        return outputs


def build_adapter(profile: RuntimeProfile) -> ASRAdapter:
    """Construct the measured primary decoder for a profile (requires the ``asr`` extra)."""

    from .advanced_adapters import LoopGuardConfig, PathPreservingFasterWhisperAdapter

    return PathPreservingFasterWhisperAdapter(
        model=profile.model,
        device=profile.device,
        compute_type=profile.compute_type,
        patience=profile.patience,
        loop_guard=LoopGuardConfig(enabled=profile.loop_guard),
    )


def calibrated_confidence(profile: RuntimeProfile, top_posterior: Any) -> float | None:
    """Map the fusion top posterior to a calibrated correctness probability."""

    if profile.confidence_calibration is None:
        return None
    try:
        posterior = float(top_posterior)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(posterior):
        return None
    posterior = min(max(posterior, 1e-6), 1.0 - 1e-6)
    slope, intercept = profile.confidence_calibration
    logit = math.log(posterior / (1.0 - posterior))
    return 1.0 / (1.0 + math.exp(-(slope * logit + intercept)))


def _segment_status(segment: Any) -> str:
    observed = getattr(segment, "observed", None)
    for attribute in ("status", "decision", "acceptance"):
        value = getattr(observed, attribute, None)
        if isinstance(value, str) and value:
            return value
    diagnostics = getattr(segment, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        for key in ("status", "decision", "observedStatus"):
            value = diagnostics.get(key)
            if isinstance(value, str) and value:
                return value
        if diagnostics.get("provisional") is True:
            return "provisional"
    return "accepted"


def _materialise_audio(audio: Any, *, sample_rate: int = 16_000) -> tuple[Path, Path | None]:
    """Return a readable audio path; arrays are written to a temporary 16 kHz WAV."""

    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, None
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is part of the asr extra
        raise TypeError("array audio input requires numpy; pass a file path instead") from exc
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if array.size == 0:
        raise ValueError("audio array is empty")
    clipped = np.clip(array, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with tempfile.NamedTemporaryFile(prefix="semantic-asr-", suffix=".wav", delete=False) as handle:
        name = handle.name
    with wave.open(name, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.tobytes())
    path = Path(name)
    return path, path


def transcribe(
    audio: Any,
    *,
    profile: str | RuntimeProfile = "cpu-ja-v1",
    language: str | None = None,
    hotwords: Iterable[str] = (),
    initial_prompt: str | None = None,
    on_progress: ProgressCallback | None = None,
    adapter: ASRAdapter | None = None,
    transcriber: SemanticASRTranscriber | None = None,
    duration_ms: int | None = None,
) -> TranscriptResult:
    """Transcribe one recording with a named runtime profile.

    ``audio`` is a file path or a 16 kHz mono float32 array. ``adapter`` and ``transcriber``
    exist for tests and for callers that keep a warm model between calls; production callers
    should use :func:`load_transcriber` once and pass it here.
    """

    resolved = runtime_profile(profile)
    source, temporary = _materialise_audio(audio)
    try:
        if transcriber is None:
            if on_progress:
                on_progress(f"loading {resolved.model} ({resolved.device}/{resolved.compute_type})")
            transcriber = load_transcriber(resolved, adapter=adapter)
        if on_progress:
            on_progress("transcribing")
        longform = transcriber.transcribe(
            source,
            duration_ms=duration_ms,
            language=language or resolved.language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
    segments = tuple(
        TranscriptSegment(
            index=index,
            start_ms=int(segment.window.start_ms),
            end_ms=int(segment.window.end_ms),
            observed=segment.observed.text,
            normalized=segment.normalized.text,
            status=_segment_status(segment),
            confidence=calibrated_confidence(
                resolved, dict(segment.diagnostics).get("topPosterior")
            ),
            diagnostics=dict(segment.diagnostics),
        )
        for index, segment in enumerate(longform.segments, 1)
    )
    base = transcriber.base_adapter
    provenance = {
        "profile": resolved.name,
        "profileDigest": resolved.digest,
        "adapter": getattr(base, "name", type(base).__name__),
        "model": getattr(base, "model_name", resolved.model),
        "modelRevision": resolved.model_revision,
        "device": getattr(base, "device", resolved.device),
        "computeType": getattr(base, "compute_type", resolved.compute_type),
        "ompNumThreads": os.environ.get("OMP_NUM_THREADS"),
        "windowMs": transcriber.window_ms,
        "overlapMs": transcriber.overlap_ms,
    }
    if on_progress:
        on_progress("done")
    utterances = utterances_from_segments(longform, segments)
    return TranscriptResult(
        profile=resolved,
        source_name=longform.source_name,
        source_audio_sha256=longform.source_audio_sha256,
        duration_ms=longform.duration_ms,
        observed_text=longform.observed_text,
        normalized_text=longform.normalized_text,
        segments=segments,
        evidence_sha256=longform.evidence_sha256,
        provenance=provenance,
        diagnostics=dict(longform.diagnostics),
        longform=longform,
        utterances=utterances,
    )


def load_transcriber(
    profile: str | RuntimeProfile = "cpu-ja-v1",
    *,
    adapter: ASRAdapter | None = None,
) -> SemanticASRTranscriber:
    """Load the model once so repeated :func:`transcribe` calls stay warm."""

    resolved = runtime_profile(profile)
    return SemanticASRTranscriber(
        adapter or build_adapter(resolved),
        window_ms=resolved.window_ms,
        overlap_ms=resolved.overlap_ms,
    )


def transcribe_segments(
    audio: Any,
    *,
    profile: str | RuntimeProfile = "cpu-ja-v1",
    language: str | None = None,
    normalized: bool = False,
    on_progress: ProgressCallback | None = None,
    transcriber: SemanticASRTranscriber | None = None,
) -> list[tuple[float, float, str]]:
    """Koemo-compatible ``[(start_seconds, end_seconds, text), ...]``.

    Rows are timestamp-token utterances when the decoder produced them, otherwise one row
    per window. The observed transcript is returned by default because it is the auditable
    channel; ``normalized=True`` returns the readable derivative at window granularity.
    """

    result = transcribe(
        audio,
        profile=profile,
        language=language,
        on_progress=on_progress,
        transcriber=transcriber,
    )
    if not normalized and result.utterances:
        return [
            (utterance.start_seconds, utterance.end_seconds, utterance.text)
            for utterance in result.utterances
        ]
    return [
        (
            segment.start_seconds,
            segment.end_seconds,
            (segment.normalized if normalized else segment.observed).strip(),
        )
        for segment in result.segments
        if (segment.normalized if normalized else segment.observed).strip()
    ]


__all__ = [
    "PROFILES",
    "Utterance",
    "utterances_from_segments",
    "RuntimeProfile",
    "TranscriptResult",
    "TranscriptSegment",
    "build_adapter",
    "load_transcriber",
    "runtime_profile",
    "transcribe",
    "transcribe_segments",
]
