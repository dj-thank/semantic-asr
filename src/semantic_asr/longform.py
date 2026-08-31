from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .adapters import ASRAdapter, DecodeRequest
from .cache import CacheKey, EvidenceCache, TeacherCacheEntry
from .candidate_pool import merge_candidate_pools
from .contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript, sha256_json
from .evidence_router import (
    QuantileBalancedRouterConfig,
    RouterState,
    route_evidence_actions,
)
from .fusion import FusionConfig, evidence_summary, fuse_candidates
from .japanese import deterministic_normalize, join_japanese_fragments
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        if duration > 0:
            return max(1, round(duration * 1000))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as stream:
            return max(1, round(stream.getnframes() / stream.getframerate() * 1000))
    raise RuntimeError("ffprobe is required to determine non-WAV duration")


def plan_windows(
    duration_ms: int,
    *,
    window_ms: int = 28_000,
    overlap_ms: int = 1_200,
) -> list[Window]:
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
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
    return str(getattr(adapter, "model_name", getattr(adapter, "name", type(adapter).__name__)))


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
    ) -> None:
        self.base_adapter = base_adapter
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
        self.window_ms = window_ms
        self.overlap_ms = overlap_ms

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
            beam_size=5,
            hypotheses=5,
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
        alignment_rows: list[dict[str, Any]] = []
        for action_index, action in enumerate(plan.selected):
            if action.start_ms is None or action.end_ms is None:
                continue
            if action.kind == "whisper-relisten":
                request = DecodeRequest(
                    audio_path=str(source),
                    language=language,
                    beam_size=12,
                    hypotheses=8,
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
                additional.extend(rows)
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
                additional.extend(rows)
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

        if additional:
            candidates = merge_candidates(candidates, additional)
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
        duration = int(duration_ms or probe_duration_ms(source))
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
        observed_text = join_japanese_fragments(segment.observed.text for segment in segments)
        normalized_text = join_japanese_fragments(segment.normalized.text for segment in segments)
        evidence_payload = {
            "audioSha256": audio_sha256,
            "durationMs": duration,
            "observedText": observed_text,
            "normalizedText": normalized_text,
            "segmentEvidence": [segment.observed.evidence_sha256 for segment in segments],
        }
        return LongformResult(
            source_name=source.name,
            source_audio_sha256=audio_sha256,
            duration_ms=duration,
            observed_text=observed_text,
            normalized_text=normalized_text,
            segments=segments,
            evidence_sha256=sha256_json(evidence_payload),
            diagnostics={
                "windowCount": len(windows),
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
