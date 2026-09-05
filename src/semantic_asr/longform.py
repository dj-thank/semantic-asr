from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .adapters import (
    ASRAdapter,
    DecodeRequest,
    _package_version,
    decode_request_identity,
    score_domain_digest,
)
from .audio import require_integer
from .cache import CacheKey, EvidenceCache, TeacherCacheEntry
from .candidate_pool import (
    SurfacePolicy,
    aggregate_surface_candidates,
    lenient_surface_key,
    merge_candidate_pools,
)
from .contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript, sha256_json
from .evidence_router import (
    QuantileBalancedRouterConfig,
    RouterState,
    route_evidence_actions,
)
from .fusion import FusionConfig, evidence_summary, fuse_candidates
from .japanese import deterministic_normalize, join_timed_fragments
from .planner import EvidenceAction, EvidenceBudget, plan_evidence
from .semantic_lattice import (
    SemanticLattice,
    build_semantic_lattice,
    semantic_change_warnings,
)
from .teachers import DelayedTeacherPolicy, TeacherResult


class TeacherClient(Protocol):
    model: str

    def probabilities(
        self,
        candidates: list[CandidateEvidence],
        *,
        context: str = "",
        locked_consensus: str = "",
        contradiction: str = "",
    ) -> TeacherResult: ...


class ForcedAligner(Protocol):
    name: str
    model_name: str

    def align(self, request: DecodeRequest, *, text: str) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class Window:
    index: int
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        require_integer(self.index, name="window index")
        require_integer(self.start_ms, name="window start_ms")
        require_integer(self.end_ms, name="window end_ms", minimum=1)
        if self.end_ms <= self.start_ms:
            raise ValueError("window end_ms must be greater than start_ms")


@dataclass(frozen=True, slots=True)
class LongformSegment:
    window: Window
    observed: ObservedTranscript
    normalized: NormalizedTranscript
    diagnostics: dict[str, Any]
    actions: tuple[EvidenceAction, ...] = ()
    cache_hits: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LongformResult:
    source_name: str
    source_audio_sha256: str
    duration_ms: int
    observed_text: str
    normalized_text: str
    segments: tuple[LongformSegment, ...]
    evidence_sha256: str
    diagnostics: dict[str, Any]

    evidence_schema: str = "semantic-asr-longform-evidence-v2"

    @classmethod
    def create(
        cls,
        *,
        source_name: str,
        source_audio_sha256: str,
        duration_ms: int,
        segments: Sequence[LongformSegment],
        diagnostics: dict[str, Any],
    ) -> LongformResult:
        rows = tuple(segments)
        result = cls(
            source_name=source_name,
            source_audio_sha256=source_audio_sha256,
            duration_ms=duration_ms,
            observed_text=join_segment_text(rows),
            normalized_text=join_segment_text(rows, normalized=True),
            segments=rows,
            evidence_sha256="",
            diagnostics=dict(diagnostics),
        )
        result = replace(result, evidence_sha256=sha256_json(result.evidence_payload()))
        result.verify()
        return result

    def evidence_payload(self) -> dict[str, Any]:
        return {
            "schema": self.evidence_schema,
            "sourceAudioSha256": self.source_audio_sha256,
            "durationMs": self.duration_ms,
            "observedText": self.observed_text,
            "normalizedText": self.normalized_text,
            "segments": [
                {
                    "window": asdict(segment.window),
                    "observedEvidenceSha256": segment.observed.evidence_sha256,
                    "normalization": asdict(segment.normalized),
                }
                for segment in self.segments
            ],
        }

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
            raise ValueError("first-pass long-form evidence hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        self.verify()
        return asdict(self)


def join_segment_text(segments: Iterable[Any], *, normalized: bool = False) -> str:
    return join_timed_fragments(
        (
            segment.window.start_ms,
            segment.window.end_ms,
            segment.normalized.text if normalized else segment.observed.text,
        )
        for segment in segments
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration_ms(path: str | Path) -> int:
    source = Path(path)
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        duration = float(completed.stdout.strip())
        if math.isfinite(duration) and duration > 0:
            return max(1, round(duration * 1000))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as stream:
            if stream.getnframes() == 0:
                raise ValueError("audio recording is empty")
            return max(1, round(stream.getnframes() / stream.getframerate() * 1000))
    raise RuntimeError("ffprobe is required to determine non-WAV duration")


def plan_windows(
    duration_ms: int,
    *,
    window_ms: int = 28_000,
    overlap_ms: int = 1_200,
) -> list[Window]:
    require_integer(duration_ms, name="duration_ms", minimum=1)
    require_integer(window_ms, name="window_ms", minimum=1)
    require_integer(overlap_ms, name="overlap_ms")
    if window_ms <= 0 or overlap_ms < 0 or overlap_ms >= window_ms:
        raise ValueError("invalid window/overlap configuration")
    output: list[Window] = []
    start = 0
    index = 0
    while start < duration_ms:
        end = min(duration_ms, start + window_ms)
        output.append(Window(index=index, start_ms=start, end_ms=end))
        if end >= duration_ms:
            break
        start = end - overlap_ms
        index += 1
    return output


def _adapter_model(adapter: ASRAdapter) -> str:
    return str(
        getattr(
            adapter,
            "model_name",
            getattr(adapter, "model", getattr(adapter, "name", type(adapter).__name__)),
        )
    )


_PROVENANCE_FIELDS = (
    "device",
    "compute_type",
    "cpu_threads",
    "length_penalty",
    "patience",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "without_timestamps",
    "loop_guard",
    "dtype",
    "device_map",
    "max_inference_batch_size",
    "max_new_tokens",
    "return_timestamps",
    "forced_aligner_revision",
    "forced_aligner_artifact_sha256",
    "maximum_hypotheses",
    "acoustic_temperature",
    "adaptive_config",
    "calibration_profile",
    "lexical_blend",
    "endpoint",
    "protocol",
    "timeout_seconds",
)


def _jsonable_provenance(value: Any, *, depth: int = 0) -> Any:
    """Convert adapter configuration to a stable, non-secret cache-key value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if depth >= 3:
        return f"{type(value).__module__}.{type(value).__qualname__}"
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable_provenance(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_provenance(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, set):
        values = [_jsonable_provenance(item, depth=depth + 1) for item in value]
        return sorted(values, key=lambda item: repr(item))
    if isinstance(value, (list, tuple)):
        return [_jsonable_provenance(item, depth=depth + 1) for item in value]
    digest = getattr(value, "digest", None)
    if isinstance(digest, str):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "digest": digest,
        }
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _nested_adapter(adapter: object, name: str) -> object | None:
    value = getattr(adapter, name, None)
    return value if value is not None and value is not adapter else None


def _legacy_cache_identity_allowed(adapter: object, *, depth: int = 0) -> bool:
    if getattr(adapter, "allow_legacy_cache_identity", False) is True:
        return True
    if depth >= 2:
        return False
    nested = _nested_adapter(adapter, "base")
    return nested is not None and _legacy_cache_identity_allowed(nested, depth=depth + 1)


def _explicit_service_identity(adapter: object) -> bool:
    """Recognize loopback teacher clients whose endpoint/model are the identity."""

    return all(getattr(adapter, name, None) for name in ("model", "endpoint", "protocol"))


def _first_provenance_value(adapter: object, name: str, *, depth: int = 0) -> Any:
    value = getattr(adapter, name, None)
    if value is not None:
        return value
    if depth >= 2:
        return None
    for nested_name in ("base",):
        nested = _nested_adapter(adapter, nested_name)
        if nested is not None:
            inherited = _first_provenance_value(nested, name, depth=depth + 1)
            if inherited is not None:
                return inherited
    return None


def _adapter_cache_provenance(adapter: object, request: DecodeRequest) -> dict[str, Any]:
    """Return model/runtime/config identity shared by all long-form cache paths.

    Production adapters must expose either an immutable model revision or a verified
    local artifact digest.  The only legacy escape hatch is an explicit
    ``allow_legacy_cache_identity`` marker on an in-memory test adapter; this avoids
    silently reusing a cache for an unbound model while keeping small fixtures useful.
    """

    settings: dict[str, Any] = {
        "schema": "semantic-asr-cache-provenance-v1",
        "adapterType": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "adapter": str(getattr(adapter, "name", type(adapter).__name__)),
        "model": _adapter_model(adapter),
        "request": decode_request_identity(request),
    }
    for name in _PROVENANCE_FIELDS:
        value = getattr(adapter, name, None)
        if value is not None:
            settings[name] = _jsonable_provenance(value)
    nested = _nested_adapter(adapter, "base")
    if nested is not None:
        settings["base"] = _adapter_cache_provenance(nested, request)["decode_settings"]
    ranker = _nested_adapter(adapter, "ranker")
    if ranker is not None:
        ranker_revision = getattr(ranker, "model_revision", None)
        ranker_artifact = getattr(ranker, "model_artifact_sha256", None)
        ranker_config = getattr(
            ranker,
            "config_digest",
            getattr(ranker, "configuration_digest", None),
        )
        if (
            ranker_revision is None
            and ranker_artifact is None
            and ranker_config is None
            and not _legacy_cache_identity_allowed(ranker)
            and not _explicit_service_identity(ranker)
        ):
            raise ValueError(
                "ranker identity is required for cache use; provide an exact Hub revision, "
                "verified local artifact SHA-256, or immutable configuration digest"
            )
        settings["ranker"] = _jsonable_provenance(
            {
                "type": f"{type(ranker).__module__}.{type(ranker).__qualname__}",
                "model": getattr(ranker, "model_name", getattr(ranker, "model_id", None)),
                "name": getattr(ranker, "name", None),
                "revision": ranker_revision,
                "artifactSha256": ranker_artifact,
                "runtime": getattr(ranker, "runtime_revision", None),
                "configDigest": ranker_config,
            }
        )
    model_revision = _first_provenance_value(adapter, "model_revision")
    runtime_revision = _first_provenance_value(adapter, "runtime_revision")
    artifact_sha256 = _first_provenance_value(adapter, "model_artifact_sha256")
    if (
        model_revision is None
        and artifact_sha256 is None
        and not _legacy_cache_identity_allowed(adapter)
        and not _explicit_service_identity(adapter)
    ):
        raise ValueError(
            "model identity is required for cache use; provide an exact Hub revision "
            "or verified local artifact SHA-256"
        )
    settings["modelRevision"] = model_revision
    settings["runtimeRevision"] = runtime_revision
    settings["modelArtifactSha256"] = artifact_sha256
    settings["ompNumThreads"] = os.environ.get("OMP_NUM_THREADS")
    adapter_name = str(getattr(adapter, "name", "")).lower()
    model_name = _adapter_model(adapter).lower()
    if "faster-whisper" in adapter_name or "faster-whisper" in model_name:
        settings["runtimePackages"] = {
            "fasterWhisper": _package_version("faster-whisper"),
            "ctranslate2": _package_version("ctranslate2"),
        }
    if "qwen" in adapter_name or "qwen" in model_name:
        settings.setdefault("runtimePackages", {})["qwenAsr"] = _package_version("qwen-asr")
    return {
        "model_revision": model_revision,
        "runtime_revision": runtime_revision,
        "model_artifact_sha256": artifact_sha256,
        "decode_settings": settings,
        "score_domain": score_domain_digest(settings),
    }


def _scoped_candidates(
    candidates: Iterable[CandidateEvidence],
    *,
    namespace: str,
    start_ms: int,
    end_ms: int,
) -> list[CandidateEvidence]:
    output: list[CandidateEvidence] = []
    for index, candidate in enumerate(candidates, 1):
        metadata = dict(candidate.metadata)
        support = set(candidate.source_support)
        metadata.update(
            {
                "sourceSupport": sorted(support),
                "decodeNamespace": namespace,
                "decodeStartMs": start_ms,
                "decodeEndMs": end_ms,
            }
        )
        output.append(
            replace(
                candidate,
                candidate_id=f"{namespace}:{start_ms}-{end_ms}:{index:04d}",
                metadata=metadata,
            )
        )
    return output


def merge_candidates(
    primary: Iterable[CandidateEvidence],
    additional: Iterable[CandidateEvidence],
) -> list[CandidateEvidence]:
    return merge_candidate_pools(primary, additional, id_prefix="merged")


def _longest_common_substring(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for character in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, 1):
            if character == other:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def span_agreement(candidate_text: str, span_text: str) -> float:
    """How much of a sub-span decode is contained in a full-window candidate, in [0, 1]."""

    candidate = lenient_surface_key(candidate_text)
    span = lenient_surface_key(span_text)
    if len(span) < 2 or not candidate:
        return 0.0
    if span in candidate:
        return 1.0
    return _longest_common_substring(candidate, span) / len(span)


def apply_span_evidence(
    candidates: Sequence[CandidateEvidence],
    span_rows: Sequence[CandidateEvidence],
) -> list[CandidateEvidence]:
    """Turn sub-span re-listening decodes into evidence instead of competing transcripts.

    A re-listen of a 300 ms contradiction island returns a few characters. Merging such rows
    into the window pool let them win the whole window on average log-probability (measured
    2026-09-03: a 28 s window collapsed to 「お届けする」). Sub-span rows therefore never enter
    the observed-eligible pool; they raise ``cross_model`` for window candidates that contain
    their text and are recorded in metadata for audit.
    """

    if not span_rows:
        return list(candidates)
    span_texts = [row.text for row in span_rows if row.text.strip()]
    output: list[CandidateEvidence] = []
    for candidate in candidates:
        agreement = max((span_agreement(candidate.text, text) for text in span_texts), default=0.0)
        metadata = dict(candidate.metadata)
        metadata["spanEvidence"] = {
            "rows": len(span_rows),
            "agreement": agreement,
            "namespaces": sorted({str(row.metadata.get("decodeNamespace")) for row in span_rows}),
        }
        cross_model = candidate.cross_model
        if agreement > 0.0:
            cross_model = agreement if cross_model is None else max(float(cross_model), agreement)
        output.append(replace(candidate, cross_model=cross_model, metadata=metadata))
    return output


def _lattice_context(lattice: SemanticLattice) -> tuple[str, str]:
    locked = " / ".join("".join(span.units) for span in lattice.locked_consensus)
    contradiction = " | ".join(
        "/".join(dict.fromkeys("".join(alternative.units) for alternative in island.alternatives))
        for island in lattice.contradiction_islands
    )
    return locked, contradiction


class SemanticASRTranscriber:
    """Long-form, cache-aware, selective Japanese ASR orchestrator."""

    def __init__(
        self,
        base_adapter: ASRAdapter,
        *,
        second_ear: ASRAdapter | None = None,
        forced_aligner: ForcedAligner | None = None,
        teacher: TeacherClient | None = None,
        cache: EvidenceCache | None = None,
        fusion_config: FusionConfig | None = None,
        teacher_policy: DelayedTeacherPolicy | None = None,
        evidence_budget: EvidenceBudget | None = None,
        balanced_router: bool = False,
        router_state: RouterState | None = None,
        router_config: QuantileBalancedRouterConfig | None = None,
        evidence_enricher: Callable[[CandidateEvidence], CandidateEvidence] | None = None,
        window_ms: int = 28_000,
        overlap_ms: int = 1_200,
        beam_size: int = 5,
        hypotheses: int = 5,
        relisten_beam_size: int = 12,
        relisten_hypotheses: int = 8,
        surface_policy: SurfacePolicy = "lenient",
    ) -> None:
        self.base_adapter = base_adapter
        self.runtime_profile_name: str | None = None
        self.runtime_profile_digest: str | None = None
        self.second_ear = second_ear
        self.forced_aligner = forced_aligner
        self.teacher = teacher
        self.cache = cache
        self.fusion_config = fusion_config or FusionConfig()
        self.teacher_policy = teacher_policy or DelayedTeacherPolicy()
        self.evidence_budget = evidence_budget or EvidenceBudget()
        self.balanced_router = bool(balanced_router)
        self.router_state = router_state or RouterState()
        self.router_config = router_config or QuantileBalancedRouterConfig()
        self.evidence_enricher = evidence_enricher
        plan_windows(1, window_ms=window_ms, overlap_ms=overlap_ms)
        if window_ms > 30_000:
            raise ValueError("Whisper windows cannot exceed 30 s")
        for name, value in (
            ("beam_size", beam_size),
            ("hypotheses", hypotheses),
            ("relisten_beam_size", relisten_beam_size),
            ("relisten_hypotheses", relisten_hypotheses),
        ):
            require_integer(value, name=name, minimum=1)
        if hypotheses > beam_size:
            raise ValueError("hypotheses cannot exceed beam_size")
        if relisten_beam_size < 1 or relisten_hypotheses < 1:
            raise ValueError("re-listen beam size and hypotheses must be positive")
        if relisten_hypotheses > relisten_beam_size:
            raise ValueError("re-listen hypotheses cannot exceed re-listen beam size")
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms
        self.beam_size = int(beam_size)
        self.hypotheses = int(hypotheses)
        self.relisten_beam_size = int(relisten_beam_size)
        self.relisten_hypotheses = int(relisten_hypotheses)
        if surface_policy not in {"exact", "lenient"}:
            raise ValueError("surface_policy must be exact or lenient")
        self.surface_policy: SurfacePolicy = surface_policy

    def _cache_key(
        self,
        *,
        namespace: str,
        adapter: ASRAdapter,
        request: DecodeRequest,
        audio_sha256: str,
        context: str,
        calibration_digest: str | None = None,
    ) -> CacheKey:
        provenance = _adapter_cache_provenance(adapter, request)
        return CacheKey.create(
            namespace=namespace,
            audio_sha256=audio_sha256,
            start_ms=request.start_ms or 0,
            end_ms=request.end_ms or 1,
            adapter=adapter.name,
            model=_adapter_model(adapter),
            language=request.language,
            beam_size=request.beam_size,
            hypotheses=request.hypotheses,
            prompt=request.initial_prompt,
            hotwords=request.hotwords,
            context=context,
            calibration_digest=calibration_digest,
            model_revision=provenance["model_revision"],
            runtime_revision=provenance["runtime_revision"],
            model_artifact_sha256=provenance["model_artifact_sha256"],
            decode_settings=provenance["decode_settings"],
            score_domain=provenance["score_domain"],
        )

    def _decode(
        self,
        adapter: ASRAdapter,
        request: DecodeRequest,
        *,
        namespace: str,
        audio_sha256: str,
        context: str,
        calibration_digest: str | None = None,
    ) -> tuple[list[CandidateEvidence], bool]:
        key = self._cache_key(
            namespace=namespace,
            adapter=adapter,
            request=request,
            audio_sha256=audio_sha256,
            context=context,
            calibration_digest=calibration_digest,
        )
        if self.cache is not None:
            cached = self.cache.get_candidates(key)
            if cached is not None:
                return cached, True
        candidates = _scoped_candidates(
            adapter.decode(request),
            namespace=namespace,
            start_ms=request.start_ms or 0,
            end_ms=request.end_ms or 1,
        )
        if self.evidence_enricher is not None:
            candidates = [self.evidence_enricher(candidate) for candidate in candidates]
        if not candidates:
            raise RuntimeError(f"{adapter.name} returned no candidates")
        if self.cache is not None:
            self.cache.put_candidates(key, candidates)
        return candidates, False

    def _teacher_result(
        self,
        candidates: list[CandidateEvidence],
        request: DecodeRequest,
        *,
        audio_sha256: str,
        context: str,
        lattice: SemanticLattice,
        calibration_digest: str | None,
    ) -> tuple[TeacherResult | None, bool]:
        if self.teacher is None:
            return None, False
        locked, contradiction = _lattice_context(lattice)
        candidate_context = json.dumps(
            [{"id": row.candidate_id, "text": row.text} for row in candidates],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        teacher_provenance = _adapter_cache_provenance(self.teacher, request)
        key = CacheKey.create(
            namespace="local-teacher",
            audio_sha256=audio_sha256,
            start_ms=request.start_ms or 0,
            end_ms=request.end_ms or 1,
            adapter="local-teacher",
            model=self.teacher.model,
            language=request.language,
            beam_size=request.beam_size,
            hypotheses=request.hypotheses,
            prompt=request.initial_prompt,
            hotwords=request.hotwords,
            context=context + "\n" + candidate_context,
            calibration_digest=calibration_digest,
            model_revision=teacher_provenance["model_revision"],
            runtime_revision=teacher_provenance["runtime_revision"],
            model_artifact_sha256=teacher_provenance["model_artifact_sha256"],
            decode_settings=teacher_provenance["decode_settings"],
            score_domain=teacher_provenance["score_domain"],
        )
        if self.cache is not None:
            cached = self.cache.get_teacher(key)
            if cached is not None:
                return (
                    TeacherResult(
                        probabilities=cached.probabilities,
                        model=cached.model,
                        endpoint_origin="cache",
                        protocol=cached.protocol,
                        entropy=cached.entropy,
                        abstained=cached.abstained,
                    ),
                    True,
                )
        result = self.teacher.probabilities(
            candidates,
            context=context,
            locked_consensus=locked,
            contradiction=contradiction,
        )
        if self.cache is not None:
            self.cache.put_teacher(
                key,
                TeacherCacheEntry(
                    probabilities=result.probabilities,
                    abstained=result.abstained,
                    entropy=result.entropy,
                    model=result.model,
                    protocol=result.protocol,
                ),
            )
        return result, False

    def _transcribe_window(
        self,
        source: Path,
        window: Window,
        *,
        audio_sha256: str,
        language: str | None,
        initial_prompt: str | None,
        hotwords: tuple[str, ...],
        context: str,
    ) -> LongformSegment:
        cache_hits: list[str] = []
        base_request = DecodeRequest(
            audio_path=str(source),
            language=language,
            beam_size=self.beam_size,
            hypotheses=self.hypotheses,
            start_ms=window.start_ms,
            end_ms=window.end_ms,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
        candidates, hit = self._decode(
            self.base_adapter,
            base_request,
            namespace="base-window",
            audio_sha256=audio_sha256,
            context=context,
        )
        if hit:
            cache_hits.append("base-window")
        if self.surface_policy != "exact":
            candidates = aggregate_surface_candidates(
                candidates, id_prefix="window", policy=self.surface_policy
            )
        ranked = fuse_candidates(candidates, self.fusion_config)
        lattice = build_semantic_lattice(
            candidates,
            posterior=ranked[0].gate.posterior,
            pivot_candidate_id=ranked[0].candidate.candidate_id,
            segment_start_ms=window.start_ms,
            segment_end_ms=window.end_ms,
        )
        plan = plan_evidence(
            ranked,
            lattice,
            budget=self.evidence_budget,
            enabled=tuple(
                kind
                for kind, enabled in (
                    ("whisper-relisten", True),
                    ("qwen-second-ear", self.second_ear is not None),
                    ("forced-align", self.forced_aligner is not None),
                    ("local-teacher", self.teacher is not None),
                    ("lexicon-lookup", self.evidence_enricher is not None),
                )
                if enabled
            ),
        )
        routing_diagnostics: dict[str, Any] = {"enabled": False}
        if self.balanced_router and (plan.selected or plan.rejected):
            routed = route_evidence_actions(
                (*plan.selected, *plan.rejected),
                budget=self.evidence_budget,
                state=self.router_state,
                config=self.router_config,
            )
            plan = routed.plan
            routing_diagnostics = {
                "enabled": True,
                "stateDigest": routed.state_digest,
                "selected": [
                    {
                        "actionId": row.action.action_id,
                        "kind": row.action.kind,
                        "routingScore": row.routing_score,
                        "loadBalanceBonus": row.load_balance_bonus,
                        "empiricalRewardBonus": row.empirical_reward_bonus,
                        "semanticBonus": row.semantic_bonus,
                        "redundancyPenalty": row.redundancy_penalty,
                    }
                    for row in routed.selected
                ],
                "rejectedCount": len(routed.rejected),
            }

        additional: list[CandidateEvidence] = []
        span_rows: list[CandidateEvidence] = []
        alignment_rows: list[dict[str, Any]] = []
        for action_index, action in enumerate(plan.selected):
            if action.start_ms is None or action.end_ms is None:
                continue
            covers_window = action.start_ms <= window.start_ms and action.end_ms >= window.end_ms
            sink = additional if covers_window else span_rows
            if action.kind == "whisper-relisten":
                request = DecodeRequest(
                    audio_path=str(source),
                    language=language,
                    beam_size=self.relisten_beam_size,
                    hypotheses=self.relisten_hypotheses,
                    start_ms=action.start_ms,
                    end_ms=action.end_ms,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                )
                rows, action_hit = self._decode(
                    self.base_adapter,
                    request,
                    namespace=f"whisper-relisten-{action_index:02d}",
                    audio_sha256=audio_sha256,
                    context=context,
                    calibration_digest=ranked[0].gate.calibration_digest,
                )
                sink.extend(rows)
                if action_hit:
                    cache_hits.append("whisper-relisten")
            elif action.kind == "qwen-second-ear" and self.second_ear is not None:
                request = DecodeRequest(
                    audio_path=str(source),
                    language=language,
                    beam_size=1,
                    hypotheses=1,
                    start_ms=action.start_ms,
                    end_ms=action.end_ms,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                    return_timestamps=True,
                )
                rows, action_hit = self._decode(
                    self.second_ear,
                    request,
                    namespace=f"qwen-second-ear-{action_index:02d}",
                    audio_sha256=audio_sha256,
                    context=context,
                    calibration_digest=ranked[0].gate.calibration_digest,
                )
                sink.extend(rows)
                if action_hit:
                    cache_hits.append("qwen-second-ear")
            elif action.kind == "forced-align" and self.forced_aligner is not None:
                request = DecodeRequest(
                    audio_path=str(source),
                    language=language,
                    beam_size=1,
                    hypotheses=1,
                    start_ms=action.start_ms,
                    end_ms=action.end_ms,
                )
                aligned = self.forced_aligner.align(request, text=ranked[0].candidate.text)
                alignment_rows.extend(
                    asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row)
                    for row in aligned
                )

        if span_rows:
            candidates = apply_span_evidence(candidates, span_rows)
        if additional:
            candidates = merge_candidates(candidates, additional)
        if additional or span_rows:
            ranked = fuse_candidates(candidates, self.fusion_config)
            lattice = build_semantic_lattice(
                candidates,
                posterior=ranked[0].gate.posterior,
                pivot_candidate_id=ranked[0].candidate.candidate_id,
                segment_start_ms=window.start_ms,
                segment_end_ms=window.end_ms,
            )

        teacher_result: TeacherResult | None = None
        teacher_cache_hit = False
        teacher_planned = any(action.kind == "local-teacher" for action in plan.selected)
        if self.teacher is not None and (
            teacher_planned or self.teacher_policy.should_query(ranked)
        ):
            teacher_result, teacher_cache_hit = self._teacher_result(
                candidates,
                base_request,
                audio_sha256=audio_sha256,
                context=context,
                lattice=lattice,
                calibration_digest=ranked[0].gate.calibration_digest,
            )
            if teacher_cache_hit:
                cache_hits.append("local-teacher")

        uncertainty_spans = [
            {
                "startMs": action.start_ms,
                "endMs": action.end_ms,
                "reasons": list(action.reasons),
                "priority": action.expected_information_gain,
                "semanticCriticality": action.semantic_criticality,
                "source": action.kind,
                "actionId": action.action_id,
            }
            for action in plan.selected
            if action.start_ms is not None and action.end_ms is not None
        ]
        observed = ObservedTranscript.create(
            selected=ranked[0],
            ranked=ranked,
            uncertainty_spans=uncertainty_spans,
            source_audio_sha256=audio_sha256,
        )
        observed.verify()

        if teacher_result is not None and not teacher_result.abstained:
            selected_id = max(
                teacher_result.probabilities,
                key=lambda candidate_id: (
                    teacher_result.probabilities[candidate_id],
                    candidate_id,
                ),
            )
            selected = next(
                candidate
                for candidate in observed.candidates
                if candidate.candidate_id == selected_id
            )
            normalized_text = selected.text
            normalized = NormalizedTranscript.attach(
                observed,
                text=normalized_text,
                mode="rank-only",
                selected_candidate_id=selected_id,
                semantic_change_warnings=semantic_change_warnings(observed.text, normalized_text),
            )
        else:
            normalized_text = deterministic_normalize(observed.text)
            normalized = NormalizedTranscript.attach(
                observed,
                text=normalized_text,
                mode="deterministic",
                semantic_change_warnings=semantic_change_warnings(observed.text, normalized_text),
            )

        diagnostics = {
            **evidence_summary(ranked),
            "candidateCount": len(candidates),
            "latticeAlignmentLevel": lattice.alignment_level,
            "consensusSpanCount": len(lattice.locked_consensus),
            "contradictionIslandCount": len(lattice.contradiction_islands),
            "criticalIslandKinds": sorted(
                {kind for island in lattice.contradiction_islands for kind in island.kinds}
            ),
            "evidenceBudgetMs": plan.budget_ms,
            "evidenceBudgetUsedMs": plan.used_ms,
            "plannedInformationGain": plan.expected_information_gain,
            "evidenceStoppingReason": plan.stopping_reason,
            "evidenceRouting": routing_diagnostics,
            "teacherUsed": teacher_result is not None,
            "teacherAffectsObserved": False,
            "teacherAbstained": (teacher_result.abstained if teacher_result is not None else None),
            "teacherCacheHit": teacher_cache_hit,
            "forcedAlignment": alignment_rows,
            "observationDecision": observed.decision,
        }
        return LongformSegment(
            window=window,
            observed=observed,
            normalized=normalized,
            diagnostics=diagnostics,
            actions=plan.selected,
            cache_hits=tuple(sorted(set(cache_hits))),
        )

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        duration_ms: int | None = None,
        language: str | None = "ja",
        initial_prompt: str | None = None,
        hotwords: Iterable[str] = (),
        context: str = "",
    ) -> LongformResult:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        audio_sha256 = sha256_file(source)
        duration = probe_duration_ms(source) if duration_ms is None else duration_ms
        windows = plan_windows(duration, window_ms=self.window_ms, overlap_ms=self.overlap_ms)
        normalized_language = None if language in {None, "", "auto"} else language
        hotword_tuple = tuple(dict.fromkeys(str(value) for value in hotwords if str(value)))
        segments = tuple(
            self._transcribe_window(
                source,
                window,
                audio_sha256=audio_sha256,
                language=normalized_language,
                initial_prompt=initial_prompt,
                hotwords=hotword_tuple,
                context=context,
            )
            for window in windows
        )
        return LongformResult.create(
            source_name=source.name,
            source_audio_sha256=audio_sha256,
            duration_ms=duration,
            segments=segments,
            diagnostics={
                "windowCount": len(windows),
                "evidenceBudgetMs": self.evidence_budget.total_cost_ms,
                "maximumEvidenceActions": self.evidence_budget.max_actions,
                "beamSize": self.beam_size,
                "hypotheses": self.hypotheses,
                "relistenBeamSize": self.relisten_beam_size,
                "relistenHypotheses": self.relisten_hypotheses,
                "cacheHitCount": sum(len(segment.cache_hits) for segment in segments),
                "provisionalWindowCount": sum(
                    segment.observed.decision == "provisional" for segment in segments
                ),
                "secondEarActionCount": sum(
                    action.kind == "qwen-second-ear"
                    for segment in segments
                    for action in segment.actions
                ),
                "teacherAbstentionCount": sum(
                    segment.diagnostics["teacherAbstained"] is True for segment in segments
                ),
            },
        )
