from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .candidate_pool import CandidatePath, CandidatePool
from .revisions import (
    FASTER_WHISPER_MODEL_REVISIONS,
    resolve_hugging_face_revision,
    verify_artifact_sha256,
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DecodeVariant:
    variant_id: str
    beam_size: int = 5
    hypotheses: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    sampling_temperature: float = 0.0
    sampling_topk: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def __post_init__(self) -> None:
        if not self.variant_id:
            raise ValueError("variant_id is required")
        if self.beam_size < 1 or self.hypotheses < 1 or self.sampling_topk < 0:
            raise ValueError("beam_size/hypotheses must be positive and topk non-negative")
        if self.patience <= 0 or self.length_penalty <= 0:
            raise ValueError("patience and length_penalty must be positive")
        if self.sampling_temperature < 0 or self.repetition_penalty <= 0:
            raise ValueError("invalid sampling or repetition settings")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class PathDecodeRequest:
    audio_path: str
    variants: tuple[DecodeVariant, ...] = (DecodeVariant("beam-5", beam_size=5, hypotheses=5),)
    language: str | None = "ja"
    start_ms: int | None = None
    end_ms: int | None = None
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    equivalence_policy: str = "exact"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.audio_path:
            raise ValueError("audio_path is required")
        if not self.variants or len({variant.variant_id for variant in self.variants}) != len(
            self.variants
        ):
            raise ValueError("variants must be non-empty with unique IDs")
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if self.end_ms is not None and self.end_ms < 0:
            raise ValueError("end_ms must be non-negative")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")


class FasterWhisperPathAdapter:
    """One-window path-preserving CTranslate2 Whisper adapter.

    Every raw returned decoder path is retained. Surface aggregation happens only
    after cumulative path log likelihoods are recovered from CTranslate2's declared
    length-normalized sequence score.
    """

    name = "faster-whisper-ctranslate2-paths-v2"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        *,
        device: str = "auto",
        device_index: int | list[int] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        model_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        runtime_revision: str | None = None,
    ) -> None:
        local_model = Path(model).expanduser().is_dir()
        if local_model:
            if model_revision is not None:
                raise ValueError("a local model directory cannot claim a Hub revision")
            if model_artifact_sha256 is None:
                raise ValueError("a local model directory requires a verified artifact SHA-256")
            model_artifact_sha256 = verify_artifact_sha256(
                model, model_artifact_sha256, identifier="faster-whisper path model"
            )
        else:
            if model_artifact_sha256 is not None:
                raise ValueError("model artifact SHA-256 is only valid for a local directory")
            model_revision = resolve_hugging_face_revision(
                model,
                model_revision,
                FASTER_WHISPER_MODEL_REVISIONS,
            )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install semantic-asr with the 'asr' extra") from exc
        self.model_name = model
        self.model_revision = model_revision
        self.model_artifact_sha256 = model_artifact_sha256
        self.runtime_revision = runtime_revision or (
            f"faster-whisper@{_package_version('faster-whisper') or 'unknown'}"
            f"+ctranslate2@{_package_version('ctranslate2') or 'unknown'}"
        )
        self.device = device
        self.compute_type = compute_type
        model_kwargs = {
            "device": device,
            "device_index": device_index,
            "compute_type": compute_type,
            "cpu_threads": cpu_threads,
            "num_workers": num_workers,
        }
        if model_revision is not None:
            model_kwargs["revision"] = model_revision
        self.model = WhisperModel(model, **model_kwargs)

    def _language(self, waveform: Any, requested: str | None) -> tuple[str, float | None, str]:
        if requested not in {None, "", "auto"}:
            return str(requested), None, "forced"
        _segments, info = self.model.transcribe(
            waveform,
            language=None,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        language = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)
        if not language:
            raise RuntimeError("faster-whisper returned no detected language")
        return (
            str(language),
            None if probability is None else float(probability),
            "auto",
        )

    def _prompt(self, tokenizer: Any, request: PathDecodeRequest) -> list[int]:
        initial_tokens = tokenizer.encode(request.initial_prompt) if request.initial_prompt else []
        hotwords = "、".join(request.hotwords) or None
        try:
            return list(
                self.model.get_prompt(
                    tokenizer,
                    previous_tokens=initial_tokens,
                    without_timestamps=True,
                    hotwords=hotwords,
                )
            )
        except TypeError:  # pragma: no cover - old faster-whisper compatibility
            try:
                return list(
                    self.model.get_prompt(
                        tokenizer,
                        initial_tokens,
                        True,
                        None,
                        hotwords,
                    )
                )
            except TypeError:
                return list(self.model.get_prompt(tokenizer, initial_tokens, True))

    def decode(self, request: PathDecodeRequest) -> CandidatePool:
        try:
            import numpy as np
            from faster_whisper.audio import decode_audio
            from faster_whisper.tokenizer import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
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
            raise ValueError("path adapter accepts at most one Whisper window")

        language, language_probability, language_policy = self._language(waveform, request.language)
        tokenizer = Tokenizer(
            self.model.hf_tokenizer,
            self.model.model.is_multilingual,
            task="transcribe",
            language=language,
        )
        from .adapters import pad_features_to_window, window_frames

        features = self.model.feature_extractor(waveform)
        if isinstance(features, tuple):
            features = features[0]
        features = pad_features_to_window(np.asarray(features), window_frames(self.model))
        encoded = self.model.encode(features)
        prompt = self._prompt(tokenizer, request)

        paths: list[CandidatePath] = []
        common_metadata = {
            "adapter": self.name,
            "model": self.model_name,
            "modelRevision": self.model_revision,
            "modelArtifactSha256": self.model_artifact_sha256,
            "runtimeRevision": self.runtime_revision,
            "device": self.device,
            "computeType": self.compute_type,
            "durationSeconds": duration_seconds,
            "language": language,
            "languageProbability": language_probability,
            "languagePolicy": language_policy,
            "startMs": request.start_ms,
            "endMs": request.end_ms,
            "initialPromptDigest": (
                hashlib.sha256(request.initial_prompt.encode("utf-8")).hexdigest()
                if request.initial_prompt
                else None
            ),
            "hotwordsDigest": (
                hashlib.sha256("、".join(request.hotwords).encode("utf-8")).hexdigest()
                if request.hotwords
                else None
            ),
            "fasterWhisperVersion": _package_version("faster-whisper"),
            "ctranslate2Version": _package_version("ctranslate2"),
            "requestMetadata": request.metadata,
        }

        for variant in request.variants:
            generated = self.model.model.generate(
                encoded,
                [prompt],
                beam_size=max(variant.beam_size, variant.hypotheses),
                num_hypotheses=variant.hypotheses,
                patience=variant.patience,
                return_scores=True,
                return_no_speech_prob=True,
                sampling_temperature=variant.sampling_temperature,
                sampling_topk=variant.sampling_topk,
                length_penalty=variant.length_penalty,
                repetition_penalty=variant.repetition_penalty,
                no_repeat_ngram_size=variant.no_repeat_ngram_size,
            )
            if len(generated) != 1:
                raise RuntimeError("expected exactly one generated utterance")
            result = generated[0]
            sequences = list(result.sequences_ids)
            scores = list(result.scores)
            if len(sequences) != len(scores):
                raise RuntimeError("CTranslate2 returned mismatched paths and scores")
            for raw_rank, (raw_tokens, normalized_score) in enumerate(
                zip(sequences, scores, strict=True),
                1,
            ):
                token_ids = tuple(int(token) for token in raw_tokens)
                text_tokens = [token for token in token_ids if token < tokenizer.timestamp_begin]
                text = tokenizer.decode(text_tokens).strip()
                if not text or not token_ids:
                    continue
                # CTranslate2 finalizes a beam score as cumulative / length**penalty.
                # Recover the cumulative value before surface-path aggregation.
                cumulative = float(normalized_score) * (len(token_ids) ** variant.length_penalty)
                if not math.isfinite(cumulative):
                    raise RuntimeError("CTranslate2 returned a non-finite path score")
                path_id = (
                    f"fw-{variant.variant_id}-{raw_rank:04d}-"
                    f"{_digest({'tokens': token_ids, 'score': cumulative})[:10]}"
                )
                paths.append(
                    CandidatePath(
                        path_id=path_id,
                        text=text,
                        cumulative_log_likelihood=cumulative,
                        token_ids=token_ids,
                        source=self.name,
                        model=self.model_name,
                        normalized_score=float(normalized_score),
                        rank=raw_rank,
                        metadata={
                            **common_metadata,
                            "variant": asdict(variant),
                            "variantDigest": variant.digest,
                            "averageLogLikelihood": cumulative / len(token_ids),
                            "noSpeechProbability": getattr(result, "no_speech_prob", None),
                        },
                    )
                )
        if not paths:
            raise RuntimeError("faster-whisper returned no non-empty decoder path")
        return CandidatePool.from_paths(
            paths,
            policy=request.equivalence_policy,  # type: ignore[arg-type]
        )


def default_diverse_variants(max_hypotheses: int = 16) -> tuple[DecodeVariant, ...]:
    if max_hypotheses < 1:
        raise ValueError("max_hypotheses must be positive")
    beam_count = min(max_hypotheses, 8)
    variants = [
        DecodeVariant(
            "beam",
            beam_size=beam_count,
            hypotheses=beam_count,
            patience=1.0,
        )
    ]
    if max_hypotheses > beam_count:
        sample_count = min(max_hypotheses - beam_count, 8)
        variants.append(
            DecodeVariant(
                "sample-low-temperature",
                beam_size=1,
                hypotheses=sample_count,
                sampling_temperature=0.25,
                sampling_topk=20,
            )
        )
    return tuple(variants)
