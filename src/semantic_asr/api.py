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
from .context_catalog import ContextCatalog, ContextSelection, load_context_catalog
from .longform import LongformResult, SemanticASRTranscriber
from .outputs import write_outputs
from .pipeline import EffortName, effort_profile
from .planner import EvidenceBudget
from .revisions import FASTER_WHISPER_MODEL_REVISIONS

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
        if not math.isfinite(float(self.patience)) or self.patience <= 0:
            raise ValueError("patience must be finite and positive")
        effort = effort_profile(self.effort)
        if self.hypotheses > effort.maximum_candidates:
            raise ValueError(
                f"{self.effort!r} allows at most {effort.maximum_candidates} candidates"
            )

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
        model_revision=FASTER_WHISPER_MODEL_REVISIONS["large-v3-turbo"],
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
        model_revision=FASTER_WHISPER_MODEL_REVISIONS["large-v3-turbo"],
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
        model_revision=FASTER_WHISPER_MODEL_REVISIONS["large-v3-turbo"],
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
    """Flatten decoder spans into monotonic absolute-time utterances.

    Malformed or non-finite timestamp rows are ignored. Relative timestamps are
    clamped to their source window, duplicate overlap rows are removed only when
    their normalized text is an exact prefix match, and a later row may replace a
    truncated prefix from the previous window.
    """

    if overlap_tolerance_ms < 0:
        raise ValueError("overlap_tolerance_ms must be non-negative")

    def relative_ms(value: Any, *, default: int) -> int | None:
        if value is None:
            return default
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return round(number)

    output: list[Utterance] = []
    for segment, raw in zip(segments, longform.segments, strict=True):
        spans = _candidate_spans(_selected_candidate(raw.observed))
        window_start = int(raw.window.start_ms)
        window_end = max(window_start, int(raw.window.end_ms))
        window_duration = window_end - window_start
        rows: list[tuple[int, int, str]] = []
        for span in spans:
            text = str(span.get("text") or "").strip()
            start_relative = relative_ms(span.get("startMs"), default=0)
            end_relative = relative_ms(span.get("endMs"), default=window_duration)
            if not text or start_relative is None or end_relative is None:
                continue
            start_relative = min(max(start_relative, 0), window_duration)
            end_relative = min(max(end_relative, 0), window_duration)
            if end_relative <= start_relative:
                continue
            rows.append(
                (
                    window_start + start_relative,
                    window_start + end_relative,
                    text,
                )
            )
        if not rows:
            text = segment.observed.strip()
            if text and window_end > window_start:
                rows.append((window_start, window_end, text))
        rows.sort(key=lambda row: (row[0], row[1]))

        for start_ms, end_ms, text in rows:
            if output and start_ms < output[-1].end_ms - overlap_tolerance_ms:
                previous = output[-1]
                previous_key = lenient_surface_key(previous.text)
                current_key = lenient_surface_key(text)
                if current_key and previous_key.startswith(current_key):
                    continue
                if (
                    previous_key
                    and current_key.startswith(previous_key)
                    and len(current_key) > len(previous_key)
                ):
                    output.pop()
                    start_ms = min(start_ms, previous.start_ms)
                else:
                    start_ms = max(start_ms, previous.end_ms)
            if end_ms <= start_ms:
                continue
            output.append(
                Utterance(
                    index=len(output) + 1,
                    segment_index=segment.index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    status=segment.status,
                    confidence=segment.confidence,
                )
            )
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
        model_revision=profile.model_revision,
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


def _mono_float32(audio: Any, np: Any) -> Any:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        first, second = (int(value) for value in array.shape)
        if first == 1:
            channel_axis = 0
        elif second == 1:
            channel_axis = 1
        else:
            candidates = [
                axis
                for axis, size in enumerate((first, second))
                if size <= 8 and size < (second, first)[axis]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "2-D audio must have one identifiable channel axis with at most 8 channels"
                )
            channel_axis = candidates[0]
        array = array.mean(axis=channel_axis)
    elif array.ndim != 1:
        raise ValueError("audio array must be one-dimensional or a 2-D samples/channels array")
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError("audio array is empty")
    if not bool(np.isfinite(array).all()):
        raise ValueError("audio array contains NaN or infinity")
    return array


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
    array = _mono_float32(audio, np)
    clipped = np.clip(array, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with tempfile.NamedTemporaryFile(prefix="semantic-asr-", suffix=".wav", delete=False) as handle:
        name = handle.name
    path = Path(name)
    try:
        with wave.open(name, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(pcm.tobytes())
    except Exception:
        with contextlib.suppress(OSError):
            path.unlink()
        raise
    return path, path


def _hotword_digest(values: Sequence[str]) -> str | None:
    if not values:
        return None
    payload = json.dumps(
        list(values),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _context_diagnostics(selection: ContextSelection | None) -> dict[str, Any]:
    if selection is None:
        return {
            "enabled": False,
            "abstained": True,
            "reason": "disabled",
            "selected": [],
        }
    return {"enabled": True, **selection.receipt()}


def _validate_warm_transcriber(
    profile: RuntimeProfile,
    transcriber: SemanticASRTranscriber,
) -> None:
    bound_digest = getattr(transcriber, "runtime_profile_digest", None)
    bound_name = getattr(transcriber, "runtime_profile_name", None)
    if bound_digest is None:
        raise ValueError(
            "warm transcriber is not bound to a runtime profile; construct it with "
            "load_transcriber()"
        )
    if bound_digest != profile.digest:
        raise ValueError(
            "warm transcriber does not match the requested runtime profile: "
            f"bound to {bound_name or 'unknown'} ({bound_digest}), requested "
            f"{profile.name} ({profile.digest})"
        )

    runtime_effort = effort_profile(profile.effort)
    expected = {
        "window_ms": profile.window_ms,
        "overlap_ms": profile.overlap_ms,
        "beam_size": profile.beam_size,
        "hypotheses": profile.hypotheses,
        "evidence_budget_ms": runtime_effort.evidence_budget_ms,
        "maximum_evidence_actions": runtime_effort.maximum_evidence_actions,
    }
    actual = {
        "window_ms": getattr(transcriber, "window_ms", None),
        "overlap_ms": getattr(transcriber, "overlap_ms", None),
        "beam_size": getattr(transcriber, "beam_size", None),
        "hypotheses": getattr(transcriber, "hypotheses", None),
        "evidence_budget_ms": getattr(
            getattr(transcriber, "evidence_budget", None), "total_cost_ms", None
        ),
        "maximum_evidence_actions": getattr(
            getattr(transcriber, "evidence_budget", None), "max_actions", None
        ),
    }
    mismatches = {
        name: (actual[name], wanted) for name, wanted in expected.items() if actual[name] != wanted
    }
    if mismatches:
        details = ", ".join(
            f"{name}={current!r} (profile requires {wanted!r})"
            for name, (current, wanted) in mismatches.items()
        )
        raise ValueError(
            "warm transcriber does not match the requested runtime profile: " + details
        )


def transcribe(
    audio: Any,
    *,
    profile: str | RuntimeProfile = "cpu-ja-v1",
    language: str | None = None,
    hotwords: Iterable[str] = (),
    initial_prompt: str | None = None,
    catalog: ContextCatalog | str | os.PathLike[str] | None = None,
    context_query: str | None = None,
    context_limit: int = 8,
    context_min_score: float = 0.55,
    context_tags: Iterable[str] = (),
    on_progress: ProgressCallback | None = None,
    adapter: ASRAdapter | None = None,
    transcriber: SemanticASRTranscriber | None = None,
    duration_ms: int | None = None,
) -> TranscriptResult:
    """Transcribe one recording with a named runtime profile.

    ``catalog`` is optional caller-frozen context.  It is activated only by an explicit
    ``context_query`` match; no match is a recorded abstention and injects no catalog term.
    ``audio`` arrays must already be 16 kHz and may be mono, samples-first stereo, or
    channels-first stereo.
    """

    if adapter is not None and transcriber is not None:
        raise ValueError("pass adapter or transcriber, not both")
    resolved = runtime_profile(profile)
    manual_hotwords = tuple(
        dict.fromkeys(str(value).strip() for value in hotwords if str(value).strip())
    )
    loaded_catalog = load_context_catalog(catalog)
    selection = (
        None
        if loaded_catalog is None
        else loaded_catalog.select(
            context_query or "",
            limit=context_limit,
            minimum_score=context_min_score,
            required_tags=context_tags,
        )
    )
    catalog_hotwords = () if selection is None else selection.hotwords
    effective_hotwords = tuple(dict.fromkeys((*manual_hotwords, *catalog_hotwords)))
    context_binding = "" if selection is None else selection.cache_context
    context_info = _context_diagnostics(selection)

    if on_progress and selection is not None:
        on_progress(
            f"context catalog: {len(selection.matches)} selected"
            if selection.matches
            else f"context catalog abstained ({selection.reason})"
        )

    source, temporary = _materialise_audio(audio)
    try:
        if transcriber is None:
            if on_progress:
                on_progress(f"loading {resolved.model} ({resolved.device}/{resolved.compute_type})")
            transcriber = load_transcriber(resolved, adapter=adapter)
        else:
            _validate_warm_transcriber(resolved, transcriber)
        if on_progress:
            on_progress("transcribing")
        longform = transcriber.transcribe(
            source,
            duration_ms=duration_ms,
            language=language or resolved.language,
            initial_prompt=initial_prompt,
            hotwords=effective_hotwords,
            context=context_binding,
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
        "effort": resolved.effort,
        "evidenceBudgetMs": transcriber.evidence_budget.total_cost_ms,
        "maximumEvidenceActions": transcriber.evidence_budget.max_actions,
        "adapter": getattr(base, "name", type(base).__name__),
        "model": getattr(base, "model_name", resolved.model),
        "modelRevision": getattr(base, "model_revision", None) or resolved.model_revision,
        "modelArtifactSha256": getattr(base, "model_artifact_sha256", None),
        "runtimeRevision": getattr(base, "runtime_revision", None),
        "device": getattr(base, "device", resolved.device),
        "computeType": getattr(base, "compute_type", resolved.compute_type),
        "ompNumThreads": os.environ.get("OMP_NUM_THREADS"),
        "windowMs": transcriber.window_ms,
        "overlapMs": transcriber.overlap_ms,
        "beamSize": transcriber.beam_size,
        "hypotheses": transcriber.hypotheses,
        "relistenBeamSize": transcriber.relisten_beam_size,
        "relistenHypotheses": transcriber.relisten_hypotheses,
        "manualHotwordCount": len(manual_hotwords),
        "catalogHotwordCount": len(catalog_hotwords),
        "effectiveHotwordCount": len(effective_hotwords),
        "effectiveHotwordsSha256": _hotword_digest(effective_hotwords),
        "contextCatalog": context_info,
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
        diagnostics={**dict(longform.diagnostics), "contextCatalog": context_info},
        longform=longform,
        utterances=utterances,
    )


def load_transcriber(
    profile: str | RuntimeProfile = "cpu-ja-v1",
    *,
    adapter: ASRAdapter | None = None,
) -> SemanticASRTranscriber:
    """Load and bind a warm model to one immutable runtime profile."""

    resolved = runtime_profile(profile)
    effort = effort_profile(resolved.effort)
    relisten_beam_size = max(12, resolved.beam_size)
    relisten_hypotheses = max(8, resolved.hypotheses)
    transcriber = SemanticASRTranscriber(
        adapter or build_adapter(resolved),
        window_ms=resolved.window_ms,
        overlap_ms=resolved.overlap_ms,
        beam_size=resolved.beam_size,
        hypotheses=resolved.hypotheses,
        relisten_beam_size=relisten_beam_size,
        relisten_hypotheses=relisten_hypotheses,
        evidence_budget=EvidenceBudget(
            total_cost_ms=effort.evidence_budget_ms,
            max_actions=effort.maximum_evidence_actions,
        ),
    )
    transcriber.runtime_profile_name = resolved.name
    transcriber.runtime_profile_digest = resolved.digest
    return transcriber


def transcribe_segments(
    audio: Any,
    *,
    profile: str | RuntimeProfile = "cpu-ja-v1",
    language: str | None = None,
    normalized: bool = False,
    hotwords: Iterable[str] = (),
    initial_prompt: str | None = None,
    catalog: ContextCatalog | str | os.PathLike[str] | None = None,
    context_query: str | None = None,
    context_limit: int = 8,
    context_min_score: float = 0.55,
    context_tags: Iterable[str] = (),
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
        hotwords=hotwords,
        initial_prompt=initial_prompt,
        catalog=catalog,
        context_query=context_query,
        context_limit=context_limit,
        context_min_score=context_min_score,
        context_tags=context_tags,
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
