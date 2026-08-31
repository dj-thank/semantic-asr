from __future__ import annotations

import math
from dataclasses import asdict, replace

from .adapters import (
    ASRAdapter,
    DecodeRequest,
    FasterWhisperAdapter,
    _digest_text,
    _package_version,
)
from .adaptive import AdaptiveKConfig, select_adaptive_k
from .calibration import CalibrationProfile, calibrate_values
from .candidate_pool import aggregate_surface_candidates
from .contracts import CandidateEvidence
from .mbr import critical_units
from .rerankers import CandidateRanker


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
        features = np.asarray(features)
        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        encoded = self.model.encode(features)

        initial_tokens = tokenizer.encode(request.initial_prompt) if request.initial_prompt else []
        hotwords = "、".join(request.hotwords) or None
        try:
            prompt = self.model.get_prompt(
                tokenizer,
                previous_tokens=initial_tokens,
                without_timestamps=True,
                hotwords=hotwords,
            )
        except TypeError:  # pragma: no cover - older faster-whisper
            try:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True, None, hotwords)
            except TypeError:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True)

        generated = self.model.model.generate(
            encoded,
            [prompt],
            beam_size=max(request.beam_size, request.hypotheses),
            patience=self.patience,
            num_hypotheses=request.hypotheses,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=0.0,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        if len(generated) != 1:
            raise RuntimeError("expected exactly one generated utterance")
        result = generated[0]
        sequences = list(result.sequences_ids)
        scores = list(result.scores)
        if len(sequences) != len(scores):
            raise RuntimeError("CTranslate2 returned mismatched hypotheses and scores")

        prompt_digest = _digest_text(request.initial_prompt)
        hotwords_digest = _digest_text(hotwords)
        score_domain = (
            f"{self.name}|{self.model_name}|{request.start_ms}:{request.end_ms}|"
            f"{prompt_digest}|{hotwords_digest}|beam={request.beam_size}|"
            f"patience={self.patience}|lp={self.length_penalty}"
        )
        rows: list[CandidateEvidence] = []
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
            rows.append(
                CandidateEvidence(
                    candidate_id=f"fw-path-{raw_rank:04d}",
                    text=text,
                    token_ids=token_ids,
                    acoustic=avg_logprob,
                    rank=raw_rank,
                    hypothesis_count=len(sequences),
                    sequence_score=sequence_score,
                    avg_logprob=avg_logprob,
                    source=self.name,
                    metadata={
                        "adapter": self.name,
                        "model": self.model_name,
                        "scoreDomain": score_domain,
                        "durationSeconds": duration_seconds,
                        "language": language,
                        "languageProbability": language_probability,
                        "languagePolicy": language_policy,
                        "noSpeechProbability": getattr(result, "no_speech_prob", None),
                        "initialPromptDigest": prompt_digest,
                        "hotwordsDigest": hotwords_digest,
                        "fasterWhisperVersion": _package_version("faster-whisper"),
                        "ctranslate2Version": _package_version("ctranslate2"),
                        "lengthPenalty": self.length_penalty,
                        "patience": self.patience,
                        "repetitionPenalty": self.repetition_penalty,
                        "noRepeatNgramSize": self.no_repeat_ngram_size,
                        "cumulativeLogprob": cumulative_logprob,
                    },
                )
            )
        output = aggregate_surface_candidates(rows, id_prefix="fw")
        output = [
            replace(
                candidate,
                rank=index,
                hypothesis_count=len(output),
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
    """Wrap an ASR adapter with adaptive candidate selection and raw-logit reranking."""

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
        probabilities = _softmax_scores(candidates, temperature=self.acoustic_temperature)
        ordered_mass = sorted(probabilities.values(), reverse=True)
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
            probabilities,
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
                    "calibrated": self.calibration_profile is not None,
                    "calibrationDigest": (
                        self.calibration_profile.digest
                        if self.calibration_profile is not None
                        else None
                    ),
                }
            )
            metadata.update(
                {
                    "evidenceScores": evidence_scores,
                    "rerankerRawLogit": float(raw_score),
                    "rerankerCalibratedScore": calibrated_score,
                    "rerankerSource": self.ranker.name,
                    "adaptiveK": asdict(decision),
                    "preRerankPosterior": probabilities[candidate.candidate_id],
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
                -probabilities[candidate.candidate_id],
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
