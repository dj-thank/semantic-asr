"""Public ASR adapter surface with a hardened standalone decode contract.

The historical implementations live in ``semantic_asr._adapters_legacy``. They remain re-exported
from this stable module; only the request contract and the legacy faster-whisper window reader are
specialized here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

from . import _adapters_legacy as _legacy
from .audio import decode_audio_window, require_integer, validate_audio_span
from .contracts import CandidateEvidence

# Preserve the historical helper surface, including private helpers intentionally used by
# advanced_adapters and longform. Public overrides are defined below.
for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)


@dataclass(frozen=True, slots=True)
class DecodeRequest:
    audio_path: str
    language: str | None = "ja"
    beam_size: int = 5
    hypotheses: int = 5
    start_ms: int | None = None
    end_ms: int | None = None
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    return_timestamps: bool = False

    def __post_init__(self) -> None:
        require_integer(self.beam_size, name="beam_size", minimum=1)
        require_integer(self.hypotheses, name="hypotheses", minimum=1)
        if self.hypotheses > self.beam_size:
            raise ValueError("hypotheses cannot exceed beam_size")
        validate_audio_span(self.start_ms, self.end_ms)
        if not isinstance(self.return_timestamps, bool):
            raise TypeError("return_timestamps must be a boolean")


class FasterWhisperAdapter(_legacy.FasterWhisperAdapter):
    """CTranslate2 N-best adapter using the shared bounded native-WAV reader."""

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        try:
            import numpy as np
            from faster_whisper.audio import decode_audio
            from faster_whisper.tokenizer import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("faster-whisper runtime dependencies are unavailable") from exc

        waveform = decode_audio_window(
            request.audio_path,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            decoder=decode_audio,
        )
        duration_seconds = len(waveform) / 16_000
        if duration_seconds <= 0:
            raise ValueError("decode request contains no audio")
        if duration_seconds > 30.0:
            raise ValueError("decode request exceeds one Whisper window")

        language, language_probability, language_policy = self._language(waveform, request.language)
        score_domain = self._score_domain(request, language=language)
        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        features = self.model.feature_extractor(waveform)
        if isinstance(features, tuple):
            features = features[0]
        features = _legacy.pad_features_to_window(
            np.asarray(features),
            _legacy.window_frames(self.model),
        )
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
        except TypeError:  # compatibility with older faster-whisper signatures
            try:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True, None, hotwords)
            except TypeError:
                prompt = self.model.get_prompt(tokenizer, initial_tokens, True)

        generated = self.model.model.generate(
            encoded,
            [prompt],
            beam_size=request.beam_size,
            num_hypotheses=request.hypotheses,
            return_scores=True,
            return_no_speech_prob=True,
            sampling_temperature=0.0,
            length_penalty=self.length_penalty,
        )
        if len(generated) != 1:
            raise RuntimeError("expected exactly one generated utterance")
        result = generated[0]
        sequences = list(result.sequences_ids)
        scores = list(result.scores)
        if len(sequences) != len(scores):
            raise RuntimeError("CTranslate2 returned mismatched hypotheses and scores")

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
                    candidate_id=f"fw-{raw_rank:04d}",
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
                        "modelRevision": self.model_revision,
                        "modelArtifactSha256": self.model_artifact_sha256,
                        "runtimeRevision": self.runtime_revision,
                        "device": self.device,
                        "computeType": self.compute_type,
                        "lengthPenalty": self.length_penalty,
                        "scoreKind": "length-normalized-sequence-log-likelihood",
                        "scoreDomain": score_domain,
                        "durationSeconds": duration_seconds,
                        "language": language,
                        "languageProbability": language_probability,
                        "languagePolicy": language_policy,
                        "noSpeechProbability": getattr(result, "no_speech_prob", None),
                        "initialPromptDigest": _legacy._digest_text(request.initial_prompt),
                        "hotwordsDigest": _legacy._digest_text(hotwords),
                        "fasterWhisperVersion": _legacy._package_version("faster-whisper"),
                        "ctranslate2Version": _legacy._package_version("ctranslate2"),
                        "cpuThreads": self.cpu_threads,
                        "ompNumThreads": os.environ.get("OMP_NUM_THREADS"),
                    },
                )
            )

        by_text: dict[str, CandidateEvidence] = {}
        for candidate in rows:
            current = by_text.get(candidate.text)
            candidate_score = (
                candidate.avg_logprob if candidate.avg_logprob is not None else -math.inf
            )
            current_score = (
                current.avg_logprob
                if current is not None and current.avg_logprob is not None
                else -math.inf
            )
            if current is None or candidate_score > current_score:
                by_text[candidate.text] = candidate
        unique = sorted(
            by_text.values(),
            key=lambda candidate: (
                -(candidate.avg_logprob if candidate.avg_logprob is not None else -math.inf),
                candidate.candidate_id,
            ),
        )
        output = [
            replace(
                candidate,
                candidate_id=f"fw-{index:04d}",
                rank=index,
                hypothesis_count=len(unique),
            )
            for index, candidate in enumerate(unique, 1)
        ]
        if not output:
            raise RuntimeError("faster-whisper returned no non-empty hypothesis")
        return output


# Preserve the public class import path in debug and persisted representations.
DecodeRequest.__module__ = __name__
FasterWhisperAdapter.__module__ = __name__
