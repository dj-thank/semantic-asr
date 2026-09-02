from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .adapters import (
    ASRAdapter,
    DecodeRequest,
    FasterWhisperAdapter,
    _digest_text,
    _package_version,
    pad_features_to_window,
    window_frames,
)
from .adaptive import AdaptiveKConfig, select_adaptive_k
from .calibration import CalibrationProfile, calibrate_values
from .candidate_pool import aggregate_surface_candidates
from .contracts import CandidateEvidence
from .mbr import critical_units
from .rerankers import CandidateRanker


def compression_ratio(text: str) -> float:
    """Whisper-style zlib compression ratio; repetition loops compress far above 2.4."""

    data = text.encode("utf-8")
    if not data:
        return 1.0
    return len(data) / len(zlib.compress(data))


def repeated_ngram_fraction(token_ids: Sequence[int], *, order: int = 4) -> float:
    """Fraction of token n-grams that repeat an earlier n-gram in the same sequence."""

    if order < 1:
        raise ValueError("order must be positive")
    if len(token_ids) < order * 2:
        return 0.0
    grams = [tuple(token_ids[index : index + order]) for index in range(len(token_ids) - order + 1)]
    return 1.0 - len(set(grams)) / len(grams)


@dataclass(frozen=True, slots=True)
class LoopGuardConfig:
    """Degenerate-decode guard for direct CTranslate2 N-best generation.

    ``faster_whisper.transcribe`` protects single-best decoding with a duration-independent
    token budget, a compression-ratio check, and a temperature fallback. Calling the raw
    ``generate`` path bypasses all three, so every beam can inherit one repetition loop and the
    whole N-best list becomes useless. This guard restores those protections as explicit,
    auditable evidence: a duration-aware token budget, per-candidate degeneracy features, and a
    sampled fallback whose scores stay in their own score domain.
    """

    enabled: bool = True
    max_tokens_per_second: float = 14.0
    max_tokens_floor: int = 32
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    repeated_ngram_threshold: float = 0.35
    max_characters_per_second: float = 12.0
    character_floor: int = 8
    fallback_temperatures: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    fallback_samples: int = 5
    drop_degenerate: bool = True
    extra_samples: int = 0
    extra_sample_temperature: float = 1.0
    extra_sample_topk: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens_per_second <= 0 or self.max_tokens_floor < 1:
            raise ValueError("token budget must be positive")
        if self.compression_ratio_threshold <= 1.0:
            raise ValueError("compression_ratio_threshold must exceed 1.0")
        if not 0.0 <= self.repeated_ngram_threshold <= 1.0:
            raise ValueError("repeated_ngram_threshold must lie in [0, 1]")
        if any(value <= 0 for value in self.fallback_temperatures):
            raise ValueError("fallback temperatures must be positive")
        if self.fallback_samples < 1:
            raise ValueError("fallback_samples must be positive")
        if self.max_characters_per_second <= 0 or self.character_floor < 1:
            raise ValueError("character budget must be positive")
        if self.extra_samples < 0 or self.extra_sample_topk < 0:
            raise ValueError("extra sampling settings must be non-negative")
        if self.extra_sample_temperature <= 0:
            raise ValueError("extra_sample_temperature must be positive")

    def max_new_tokens(self, duration_seconds: float) -> int | None:
        if not self.enabled:
            return None
        budget = self.max_tokens_floor + duration_seconds * self.max_tokens_per_second
        return int(max(self.max_tokens_floor, min(440, budget)))

    def character_budget(self, duration_seconds: float) -> int:
        return int(self.character_floor + duration_seconds * self.max_characters_per_second)

    def degeneracy(
        self,
        text: str,
        token_ids: Sequence[int],
        avg_logprob: float,
        *,
        duration_seconds: float | None = None,
    ) -> dict:
        ratio = compression_ratio(text)
        repeated = repeated_ngram_fraction(token_ids)
        characters = sum(1 for character in text if not character.isspace())
        budget = None if duration_seconds is None else self.character_budget(duration_seconds)
        reasons: list[str] = []
        if self.enabled:
            if ratio > self.compression_ratio_threshold:
                reasons.append("compression-ratio")
            if repeated > self.repeated_ngram_threshold:
                reasons.append("repeated-ngram")
            if avg_logprob < self.log_prob_threshold:
                reasons.append("low-logprob")
            if budget is not None and characters > budget:
                reasons.append("character-budget")
        return {
            "compressionRatio": ratio,
            "repeatedNgramFraction": repeated,
            "characterCount": characters,
            "characterBudget": budget,
            "degenerate": bool(reasons),
            "degenerateReasons": reasons,
        }

    @property
    def stages(self) -> tuple[tuple[str, float], ...]:
        if not self.enabled:
            return (("beam", 0.0),)
        return (
            ("beam", 0.0),
            *(
                (f"sample-t{temperature:g}", temperature)
                for temperature in self.fallback_temperatures
            ),
        )

    @property
    def enrichment_stage(self) -> tuple[str, float] | None:
        """Always-on sampled candidates for sample-based MBR.

        Re-evaluating MBR for ASR (TMLR 2026) reports that MBR over 4-32 sampled
        hypotheses beats beam search for Whisper-family models, including Japanese
        test sets. Beam N-best lists collapse to a handful of surfaces after path
        aggregation, so this stage adds independent samples in their own score domain.
        """

        if self.extra_samples < 1:
            return None
        return (f"mbr-sample-t{self.extra_sample_temperature:g}", self.extra_sample_temperature)


def apply_loop_guard(
    candidates: Sequence[CandidateEvidence],
    *,
    config: LoopGuardConfig,
) -> list[CandidateEvidence]:
    """Demote or drop degenerate candidates while never returning an empty list."""

    healthy = [row for row in candidates if not row.metadata.get("degenerate")]
    degenerate = [row for row in candidates if row.metadata.get("degenerate")]
    if not healthy:
        return list(candidates)
    kept = healthy if config.drop_degenerate else [*healthy, *degenerate]
    return [
        replace(
            row,
            metadata={
                **row.metadata,
                "rejectedDegeneratePaths": len(degenerate),
            },
        )
        for row in kept
    ]


class PathPreservingFasterWhisperAdapter(FasterWhisperAdapter):
    """CTranslate2 Whisper N-best with decoder-path probability aggregation."""

    name = "faster-whisper-ctranslate2-path-pool"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        *,
        length_penalty: float = 1.0,
        patience: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        loop_guard: LoopGuardConfig | None = None,
        without_timestamps: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            compute_type=compute_type,
            length_penalty=length_penalty,
        )
        if patience <= 0:
            raise ValueError("patience must be positive")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        self.patience = float(patience)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.loop_guard = loop_guard or LoopGuardConfig()
        # Whisper decodes a short clip padded to 30 s far more stably when timestamp tokens
        # are allowed: the decoder can close the segment instead of looping to max_length.
        self.without_timestamps = bool(without_timestamps)

    def _rows_from_result(
        self,
        result: Any,
        *,
        tokenizer: Any,
        stage_id: str,
        temperature: float,
        score_domain: str,
        common_metadata: dict[str, Any],
        duration_seconds: float,
    ) -> list[CandidateEvidence]:
        guard = self.loop_guard
        sequences = list(result.sequences_ids)
        scores = list(result.scores)
        if len(sequences) != len(scores):
            raise RuntimeError("CTranslate2 returned mismatched hypotheses and scores")
        stage_rows: list[CandidateEvidence] = []
        for raw_rank, (tokens, score) in enumerate(zip(sequences, scores, strict=True), 1):
            token_ids = tuple(int(token) for token in tokens)
            text_tokens = [token for token in token_ids if token < tokenizer.timestamp_begin]
            text = tokenizer.decode(text_tokens).strip()
            if not text:
                continue
            sequence_score = float(score)
            token_count = max(1, len(token_ids))
            cumulative_logprob = sequence_score * (token_count**self.length_penalty)
            avg_logprob = cumulative_logprob / (token_count + 1)
            degeneracy = guard.degeneracy(
                text, token_ids, avg_logprob, duration_seconds=duration_seconds
            )
            stage_rows.append(
                CandidateEvidence(
                    candidate_id=f"fw-{stage_id}-{raw_rank:04d}",
                    text=text,
                    token_ids=token_ids,
                    acoustic=avg_logprob,
                    rank=raw_rank,
                    hypothesis_count=len(sequences),
                    sequence_score=sequence_score,
                    avg_logprob=avg_logprob,
                    source=self.name,
                    metadata={
                        **common_metadata,
                        "scoreDomain": score_domain,
                        "decodeStage": stage_id,
                        "samplingTemperature": temperature,
                        "noSpeechProbability": getattr(result, "no_speech_prob", None),
                        "cumulativeLogprob": cumulative_logprob,
                        **degeneracy,
                    },
                )
            )
        return stage_rows

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        try:
            import numpy as np
            from faster_whisper.audio import decode_audio
            from faster_whisper.tokenizer import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("faster-whisper runtime dependencies are unavailable") from exc

        waveform = decode_audio(request.audio_path, sampling_rate=16_000)
        if request.start_ms is not None or request.end_ms is not None:
            start_sample = max(0, int((request.start_ms or 0) * 16))
            end_ms = request.end_ms if request.end_ms is not None else len(waveform) / 16
            end_sample = min(len(waveform), int(end_ms * 16))
            waveform = waveform[start_sample:end_sample]
        duration_seconds = len(waveform) / 16_000
        if duration_seconds <= 0:
            raise ValueError("decode request contains no audio")
        if duration_seconds > 30.0:
            raise ValueError("decode request exceeds one Whisper window")

        language, language_probability, language_policy = self._language(waveform, request.language)
        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        features = self.model.feature_extractor(waveform)
        if isinstance(features, tuple):
            features = features[0]
        features = pad_features_to_window(np.asarray(features), window_frames(self.model))
        encoded = self.model.encode(features)

        initial_tokens = tokenizer.encode(request.initial_prompt) if request.initial_prompt else []
        hotwords = "、".join(request.hotwords) or None
        try:
            prompt = self.model.get_prompt(
                tokenizer,
                previous_tokens=initial_tokens,
                without_timestamps=self.without_timestamps,
                hotwords=hotwords,
            )
        except TypeError:  # pragma: no cover - older faster-whisper
            try:
                prompt = self.model.get_prompt(
                    tokenizer, initial_tokens, self.without_timestamps, None, hotwords
                )
            except TypeError:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, self.without_timestamps)

        guard = self.loop_guard
        max_new_tokens = guard.max_new_tokens(duration_seconds)
        model_max_length = int(getattr(self.model, "max_length", 448))
        max_length = (
            model_max_length
            if max_new_tokens is None
            else min(model_max_length, len(prompt) + max_new_tokens)
        )
        prompt_digest = _digest_text(request.initial_prompt)
        hotwords_digest = _digest_text(hotwords)
        common_metadata = {
            "adapter": self.name,
            "model": self.model_name,
            "scoreKind": "length-normalized-sequence-log-likelihood",
            "durationSeconds": duration_seconds,
            "language": language,
            "languageProbability": language_probability,
            "languagePolicy": language_policy,
            "initialPromptDigest": prompt_digest,
            "hotwordsDigest": hotwords_digest,
            "fasterWhisperVersion": _package_version("faster-whisper"),
            "ctranslate2Version": _package_version("ctranslate2"),
            "lengthPenalty": self.length_penalty,
            "patience": self.patience,
            "repetitionPenalty": self.repetition_penalty,
            "noRepeatNgramSize": self.no_repeat_ngram_size,
            "maxNewTokens": max_new_tokens,
            "withoutTimestamps": self.without_timestamps,
            "loopGuard": asdict(guard),
        }

        rows: list[CandidateEvidence] = []
        stage_log: list[dict[str, object]] = []
        for stage_id, temperature in guard.stages:
            if temperature > 0:
                generate_kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": guard.fallback_samples,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                generate_kwargs = {
                    "beam_size": max(request.beam_size, request.hypotheses),
                    "num_hypotheses": request.hypotheses,
                    "patience": self.patience,
                    "sampling_temperature": 0.0,
                }
            generated = self.model.model.generate(
                encoded,
                [prompt],
                return_scores=True,
                return_no_speech_prob=True,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                max_length=max_length,
                **generate_kwargs,
            )
            if len(generated) != 1:
                raise RuntimeError("expected exactly one generated utterance")
            result = generated[0]
            score_domain = (
                f"{self.name}|{self.model_name}|{request.start_ms}:{request.end_ms}|"
                f"{prompt_digest}|{hotwords_digest}|beam={request.beam_size}|"
                f"patience={self.patience}|lp={self.length_penalty}|stage={stage_id}"
            )
            stage_rows = self._rows_from_result(
                result,
                tokenizer=tokenizer,
                stage_id=stage_id,
                temperature=temperature,
                score_domain=score_domain,
                common_metadata=common_metadata,
                duration_seconds=duration_seconds,
            )
            best = max(stage_rows, key=lambda row: row.sequence_score or -math.inf, default=None)
            stage_log.append(
                {
                    "stage": stage_id,
                    "temperature": temperature,
                    "paths": len(stage_rows),
                    "degeneratePaths": sum(1 for row in stage_rows if row.metadata["degenerate"]),
                    "bestDegenerate": bool(best is not None and best.metadata["degenerate"]),
                }
            )
            rows.extend(stage_rows)
            if best is not None and not best.metadata["degenerate"]:
                break

        enrichment = guard.enrichment_stage
        if enrichment is not None:
            stage_id, temperature = enrichment
            generated = self.model.model.generate(
                encoded,
                [prompt],
                beam_size=1,
                num_hypotheses=guard.extra_samples,
                sampling_topk=guard.extra_sample_topk,
                sampling_temperature=temperature,
                return_scores=True,
                return_no_speech_prob=True,
                length_penalty=self.length_penalty,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                max_length=max_length,
            )
            result = generated[0]
            stage_rows = self._rows_from_result(
                result,
                tokenizer=tokenizer,
                stage_id=stage_id,
                temperature=temperature,
                score_domain=(
                    f"{self.name}|{self.model_name}|{request.start_ms}:{request.end_ms}|"
                    f"{prompt_digest}|{hotwords_digest}|sampling={temperature:g}|"
                    f"topk={guard.extra_sample_topk}|lp={self.length_penalty}|stage={stage_id}"
                ),
                common_metadata=common_metadata,
                duration_seconds=duration_seconds,
            )
            stage_log.append(
                {
                    "stage": stage_id,
                    "temperature": temperature,
                    "paths": len(stage_rows),
                    "degeneratePaths": sum(1 for row in stage_rows if row.metadata["degenerate"]),
                    "bestDegenerate": False,
                    "enrichment": True,
                }
            )
            rows.extend(stage_rows)

        rows = apply_loop_guard(rows, config=guard)
        output = aggregate_surface_candidates(rows, id_prefix="fw")
        output = [
            replace(
                candidate,
                rank=index,
                hypothesis_count=len(output),
                metadata={**candidate.metadata, "decodeStages": stage_log},
            )
            for index, candidate in enumerate(output, 1)
        ]
        if not output:
            raise RuntimeError("faster-whisper returned no non-empty hypothesis")
        return output


def _softmax_scores(
    candidates: list[CandidateEvidence],
    *,
    temperature: float,
) -> dict[str, float]:
    """Normalize comparable acoustic scores inside one candidate set.

    The result is candidate-distribution mass, not a calibrated probability
    that a transcript is correct.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    raw = [
        (
            float(candidate.avg_logprob)
            if candidate.avg_logprob is not None
            else float(candidate.acoustic)
            if candidate.acoustic is not None
            else -20.0
        )
        for candidate in candidates
    ]
    maximum = max(raw)
    exponentials = [
        math.exp(max(-80.0, min(80.0, (value - maximum) / temperature))) for value in raw
    ]
    total = sum(exponentials) or 1.0
    return {
        candidate.candidate_id: value / total
        for candidate, value in zip(candidates, exponentials, strict=True)
    }


def _normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    raw = -sum(value * math.log(value + 1e-12) for value in probabilities)
    return min(1.0, max(0.0, raw / math.log(len(probabilities))))


def _semantic_criticality(candidates: list[CandidateEvidence]) -> float:
    signatures = {tuple(critical_units(candidate.text)) for candidate in candidates}
    if len(signatures) <= 1:
        return 0.0
    maximum = max((len(value) for value in signatures), default=0)
    return min(1.0, 0.45 + 0.12 * maximum)


class AdaptiveRerankingAdapter:
    """Wrap an ASR adapter with adaptive candidate selection and raw-logit reranking.

    An uncalibrated ranker may reorder and prune candidates, but it cannot add a
    probability-like score to the fusion evidence streams. Only an explicitly
    supplied held-out ``CalibrationProfile`` may convert ranker output into the
    lexical stream.
    """

    name = "adaptive-reranking-adapter"

    def __init__(
        self,
        base: ASRAdapter,
        ranker: CandidateRanker,
        *,
        maximum_hypotheses: int = 12,
        acoustic_temperature: float = 0.20,
        adaptive_config: AdaptiveKConfig | None = None,
        calibration_profile: CalibrationProfile | None = None,
        lexical_blend: float = 0.65,
    ) -> None:
        if maximum_hypotheses < 2:
            raise ValueError("maximum_hypotheses must be at least two")
        if not 0 <= lexical_blend <= 1:
            raise ValueError("lexical_blend must be in [0, 1]")
        self.base = base
        self.ranker = ranker
        self.model_name = f"{getattr(base, 'model_name', base.name)}+{ranker.name}"
        self.maximum_hypotheses = int(maximum_hypotheses)
        self.acoustic_temperature = float(acoustic_temperature)
        self.adaptive_config = adaptive_config or AdaptiveKConfig(maximum_k=maximum_hypotheses)
        self.calibration_profile = calibration_profile
        self.lexical_blend = float(lexical_blend)

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        expanded = replace(
            request,
            beam_size=max(request.beam_size, self.maximum_hypotheses),
            hypotheses=max(request.hypotheses, self.maximum_hypotheses),
        )
        candidates = aggregate_surface_candidates(
            self.base.decode(expanded), id_prefix="adaptive-pool"
        )
        acoustic_mass = _softmax_scores(candidates, temperature=self.acoustic_temperature)
        ordered_mass = sorted(acoustic_mass.values(), reverse=True)
        entropy = _normalized_entropy(ordered_mass)
        top = ordered_mass[0] if ordered_mass else 0.0
        second = ordered_mass[1] if len(ordered_mass) > 1 else 0.0
        selective_risk = min(
            1.0,
            0.52 * (1.0 - top) + 0.30 * entropy + 0.18 * (1.0 - (top - second)),
        )
        criticality = _semantic_criticality(candidates)
        decision = select_adaptive_k(
            candidates,
            acoustic_mass,
            selective_risk=selective_risk,
            semantic_criticality=criticality,
            config=self.adaptive_config,
        )
        selected_ids = set(decision.selected_candidate_ids)
        selected = [candidate for candidate in candidates if candidate.candidate_id in selected_ids]
        scores = dict(
            self.ranker.score(
                selected,
                context=request.initial_prompt or "",
                consensus="",
                contradiction="",
            )
        )
        if set(scores) != selected_ids:
            raise ValueError("ranker must return exactly one score for each selected candidate")

        calibrated: list[float | None]
        if self.calibration_profile is None:
            calibrated = [None] * len(selected)
        else:
            calibrated = calibrate_values(
                [scores[candidate.candidate_id] for candidate in selected],
                profile=self.calibration_profile,
                stream_name=f"reranker:{self.ranker.name}",
            )

        output: list[CandidateEvidence] = []
        for candidate, raw_score, calibrated_score in zip(
            selected,
            [scores[candidate.candidate_id] for candidate in selected],
            calibrated,
            strict=True,
        ):
            metadata = dict(candidate.metadata)
            evidence_scores = list(metadata.get("evidenceScores", []))
            evidence_scores.append(
                {
                    "source": self.ranker.name,
                    "kind": "logit",
                    "value": float(raw_score),
                    "calibrated": False,
                }
            )
            if calibrated_score is not None:
                evidence_scores.append(
                    {
                        "source": self.ranker.name,
                        "kind": "probability",
                        "value": float(calibrated_score),
                        "calibrated": True,
                        "calibrationDigest": self.calibration_profile.digest,
                    }
                )
            metadata.update(
                {
                    "evidenceScores": evidence_scores,
                    "rerankerRawLogit": float(raw_score),
                    "rerankerCalibratedProbability": calibrated_score,
                    "rerankerEvidenceInjected": calibrated_score is not None,
                    "rerankerSource": self.ranker.name,
                    "adaptiveK": asdict(decision),
                    "preRerankAcousticMass": acoustic_mass[candidate.candidate_id],
                    "adaptiveSelectiveRisk": selective_risk,
                    "adaptiveSemanticCriticality": criticality,
                }
            )
            lexical = candidate.lexical
            if calibrated_score is not None:
                lexical = (
                    float(calibrated_score)
                    if lexical is None
                    else (1.0 - self.lexical_blend) * float(lexical)
                    + self.lexical_blend * float(calibrated_score)
                )
            output.append(
                replace(
                    candidate,
                    lexical=lexical,
                    metadata=metadata,
                )
            )
        output.sort(
            key=lambda candidate: (
                -float(candidate.metadata.get("rerankerRawLogit", -1e30)),
                -acoustic_mass[candidate.candidate_id],
                candidate.candidate_id,
            )
        )
        return [
            replace(
                candidate,
                rank=index,
                hypothesis_count=len(output),
            )
            for index, candidate in enumerate(output, 1)
        ]
