from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .contracts import CandidateEvidence, canonical_json
from .revisions import (
    FASTER_WHISPER_MODEL_REVISIONS,
    QWEN_ASR_MODEL_REVISIONS,
    QWEN_FORCED_ALIGNER_REVISIONS,
    resolve_hugging_face_revision,
    verify_artifact_sha256,
)


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
        if self.beam_size < 1 or self.hypotheses < 1:
            raise ValueError("beam_size and hypotheses must be positive")
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("start_ms must be non-negative")
        if self.end_ms is not None and self.start_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")


class ASRAdapter(Protocol):
    name: str

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]: ...


class MockASRAdapter:
    name = "mock"
    model_name = "mock"

    def __init__(self, candidates: list[CandidateEvidence]) -> None:
        self.candidates = candidates
        self.requests: list[DecodeRequest] = []

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.requests.append(request)
        return list(self.candidates[: request.hypotheses])


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _digest_text(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_domain_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable, auditable identity for one numeric decoder score domain."""

    digest = hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()
    return f"semantic-asr-score-domain-v1:{digest}"


def decode_request_identity(request: DecodeRequest) -> dict[str, Any]:
    """Serialize score-changing request controls without retaining prompt text."""

    return {
        "language": request.language,
        "beamSize": request.beam_size,
        "hypotheses": request.hypotheses,
        "startMs": request.start_ms,
        "endMs": request.end_ms,
        "initialPromptSha256": _digest_text(request.initial_prompt),
        "hotwords": list(request.hotwords),
        "returnTimestamps": request.return_timestamps,
    }


def pad_features_to_window(features: Any, nb_max_frames: int) -> Any:
    """Zero-pad (or trim) log-mel features to one full Whisper window.

    Whisper was trained on 30 s windows. ``faster_whisper.transcribe`` pads every segment
    with ``pad_or_trim`` before encoding; calling the encoder on an unpadded short clip makes
    the decoder hallucinate and loop until ``max_length``. Every direct-generate adapter in
    this package must therefore pad before ``encode``.
    """

    import numpy as np

    array = np.asarray(features)
    if array.ndim == 2:
        array = np.expand_dims(array, 0)
    if nb_max_frames < 1:
        raise ValueError("nb_max_frames must be positive")
    frames = array.shape[-1]
    if frames == nb_max_frames:
        return array
    if frames > nb_max_frames:
        return array[..., :nb_max_frames]
    padding = [(0, 0)] * (array.ndim - 1) + [(0, nb_max_frames - frames)]
    return np.pad(array, padding, mode="constant")


def window_frames(model: Any) -> int:
    extractor = getattr(model, "feature_extractor", None)
    frames = getattr(extractor, "nb_max_frames", None)
    return int(frames) if frames else 3000


def _validate_local_snapshot_revision(path: str, revision: str | None) -> None:
    if revision is None or not Path(path).is_dir():
        return
    raise ValueError(
        "a local model directory cannot claim a Hub revision; use the Hub ID and revision "
        "or record a separately verified artifact digest"
    )


def _local_artifact_digest(
    model: str,
    artifact_sha256: str | None,
    *,
    identifier: str = "model artifact",
    required: bool = False,
) -> str | None:
    """Verify a local directory digest while keeping Hub revisions separate."""

    local = Path(model).expanduser().is_dir()
    if local and artifact_sha256 is None and required:
        raise ValueError(f"a verified local model directory requires {identifier} SHA-256")
    if artifact_sha256 is None:
        return None
    if not local:
        raise ValueError(f"{identifier} is only valid for a local model directory")
    return verify_artifact_sha256(model, artifact_sha256, identifier=identifier)


class FasterWhisperAdapter:
    """One-window CTranslate2 N-best adapter built on faster-whisper internals.

    The adapter deliberately rejects spans longer than one Whisper window. Long-form
    orchestration lives in ``semantic_asr.longform`` so hypotheses from unrelated
    windows are never concatenated into a false global N-best list.
    """

    name = "faster-whisper-ctranslate2-nbest"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "auto",
        compute_type: str = "default",
        *,
        length_penalty: float = 1.0,
        model_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        artifact_sha256: str | None = None,
        runtime_revision: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("install semantic-asr with the 'asr' extra") from exc
        local_model = Path(model).expanduser().is_dir()
        _validate_local_snapshot_revision(model, model_revision)
        if model_revision is not None and not local_model:
            model_revision = resolve_hugging_face_revision(
                model,
                model_revision,
                FASTER_WHISPER_MODEL_REVISIONS,
            )
        elif model_revision is None and not local_model and model in FASTER_WHISPER_MODEL_REVISIONS:
            model_revision = resolve_hugging_face_revision(
                model,
                None,
                FASTER_WHISPER_MODEL_REVISIONS,
            )
        if (
            model_artifact_sha256 is not None
            and artifact_sha256 is not None
            and model_artifact_sha256.lower() != artifact_sha256.lower()
        ):
            raise ValueError("model_artifact_sha256 and artifact_sha256 disagree")
        model_artifact_sha256 = _local_artifact_digest(
            model,
            model_artifact_sha256 or artifact_sha256,
            required=True,
        )
        self.model_name = model
        self.model_revision = model_revision
        self.model_artifact_sha256 = model_artifact_sha256
        self.runtime_revision = (
            None if runtime_revision is None else str(runtime_revision).strip() or None
        )
        self.device = str(device)
        self.compute_type = str(compute_type)
        self.cpu_threads = int(cpu_threads)
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads must be non-negative")
        self.length_penalty = float(length_penalty)
        self.model = WhisperModel(
            model,
            device=self.device,
            compute_type=self.compute_type,
            revision=model_revision,
            cpu_threads=self.cpu_threads,
        )

    def _score_domain(self, request: DecodeRequest, *, language: str) -> str:
        return score_domain_digest(
            {
                "adapter": self.name,
                "model": self.model_name,
                "modelRevision": self.model_revision,
                "modelArtifactSha256": self.model_artifact_sha256,
                "runtimeRevision": self.runtime_revision,
                "device": self.device,
                "computeType": self.compute_type,
                "fasterWhisperVersion": _package_version("faster-whisper"),
                "ctranslate2Version": _package_version("ctranslate2"),
                "ompNumThreads": os.environ.get("OMP_NUM_THREADS"),
                "language": language,
                "request": decode_request_identity(request),
                "lengthPenalty": self.length_penalty,
                "cpuThreads": self.cpu_threads,
                "scoreKind": "length-normalized-sequence-log-likelihood",
            }
        )

    def _language(self, waveform: Any, requested: str | None) -> tuple[str, float | None, str]:
        if requested not in {None, "", "auto"}:
            return str(requested), None, "forced"
        # Use the public high-level detector path instead of guessing an internal API.
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
        try:
            probability_value = None if probability is None else float(probability)
        except (TypeError, ValueError):
            probability_value = None
        return str(language), probability_value, "auto"

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
        features = pad_features_to_window(np.asarray(features), window_frames(self.model))
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
            beam_size=max(request.beam_size, request.hypotheses),
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
                        "initialPromptDigest": _digest_text(request.initial_prompt),
                        "hotwordsDigest": _digest_text(hotwords),
                        "fasterWhisperVersion": _package_version("faster-whisper"),
                        "ctranslate2Version": _package_version("ctranslate2"),
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


_QWEN_LANGUAGE = {
    "ja": "Japanese",
    "jpn": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "日本語": "Japanese",
}


def qwen_language_name(language: str | None) -> str | None:
    if language is None or not str(language).strip():
        return None
    value = str(language).strip()
    if value.lower() == "auto":
        return None
    return _QWEN_LANGUAGE.get(value.lower(), value)


def _torch_dtype(name: str) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("qwen-asr requires torch") from exc
    aliases = {
        "auto": None,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _audio_slice(request: DecodeRequest) -> str | tuple[Any, int]:
    if request.start_ms is None and request.end_ms is None:
        return request.audio_path
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("Qwen span decoding requires librosa") from exc
    offset = (request.start_ms or 0) / 1000
    duration = None if request.end_ms is None else (request.end_ms - (request.start_ms or 0)) / 1000
    waveform, sample_rate = librosa.load(
        request.audio_path,
        sr=None,
        mono=True,
        offset=offset,
        duration=duration,
    )
    if len(waveform) == 0:
        raise ValueError("Qwen span contains no audio")
    return waveform, int(sample_rate)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


class Qwen3ASRAdapter:
    """Official qwen-asr high-level wrapper used as an independent second ear."""

    name = "qwen3-asr"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-ASR-0.6B",
        *,
        model_revision: str | None = None,
        dtype: str = "float16",
        device_map: str = "cuda:0",
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 512,
        return_timestamps: bool = False,
        forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        artifact_sha256: str | None = None,
        forced_aligner_artifact_sha256: str | None = None,
        runtime_revision: str | None = None,
    ) -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("install semantic-asr with the 'qwen' extra") from exc
        local_model = Path(model).expanduser().is_dir()
        local_aligner = Path(forced_aligner).expanduser().is_dir()
        _validate_local_snapshot_revision(model, model_revision)
        if model_revision is not None and not local_model:
            model_revision = resolve_hugging_face_revision(
                model,
                model_revision,
                QWEN_ASR_MODEL_REVISIONS,
            )
        elif model_revision is None and not local_model and model in QWEN_ASR_MODEL_REVISIONS:
            model_revision = resolve_hugging_face_revision(
                model,
                None,
                QWEN_ASR_MODEL_REVISIONS,
            )
        if (
            model_artifact_sha256 is not None
            and artifact_sha256 is not None
            and model_artifact_sha256.lower() != artifact_sha256.lower()
        ):
            raise ValueError("model_artifact_sha256 and artifact_sha256 disagree")
        model_artifact_sha256 = _local_artifact_digest(
            model,
            model_artifact_sha256 or artifact_sha256,
            required=True,
        )

        _validate_local_snapshot_revision(forced_aligner, forced_aligner_revision)
        if forced_aligner_revision is not None and not local_aligner:
            forced_aligner_revision = resolve_hugging_face_revision(
                forced_aligner,
                forced_aligner_revision,
                QWEN_FORCED_ALIGNER_REVISIONS,
            )
        elif (
            forced_aligner_revision is None
            and not local_aligner
            and return_timestamps
            and forced_aligner in QWEN_FORCED_ALIGNER_REVISIONS
        ):
            forced_aligner_revision = resolve_hugging_face_revision(
                forced_aligner,
                None,
                QWEN_FORCED_ALIGNER_REVISIONS,
            )
        forced_aligner_artifact_sha256 = _local_artifact_digest(
            forced_aligner,
            forced_aligner_artifact_sha256,
            identifier="forced aligner artifact",
            required=return_timestamps,
        )
        if (
            return_timestamps
            and forced_aligner_revision is None
            and forced_aligner_artifact_sha256 is None
        ):
            raise ValueError(
                "a Qwen timestamp run requires an exact aligner revision or verified artifact"
            )
        self.model_name = model
        self.model_revision = model_revision
        self.model_artifact_sha256 = model_artifact_sha256
        self.runtime_revision = (
            None if runtime_revision is None else str(runtime_revision).strip() or None
        )
        self.forced_aligner_revision = forced_aligner_revision
        self.forced_aligner_artifact_sha256 = forced_aligner_artifact_sha256
        self.dtype = str(dtype)
        self.device_map = str(device_map)
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        if self.max_inference_batch_size < 1 or self.max_new_tokens < 1:
            raise ValueError("Qwen inference limits must be positive")
        self.return_timestamps = return_timestamps
        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        resolved_dtype = _torch_dtype(dtype)
        if resolved_dtype is not None:
            kwargs["dtype"] = resolved_dtype
        if return_timestamps:
            resolved_aligner = forced_aligner
            if forced_aligner_revision is not None and not Path(forced_aligner).is_dir():
                try:
                    from huggingface_hub import snapshot_download
                except ImportError as exc:  # pragma: no cover - qwen-asr dependency boundary
                    raise RuntimeError(
                        "Qwen forced-aligner revision pinning requires huggingface-hub"
                    ) from exc
                resolved_aligner = snapshot_download(
                    repo_id=forced_aligner,
                    revision=forced_aligner_revision,
                )
            kwargs["forced_aligner"] = resolved_aligner
            kwargs["forced_aligner_kwargs"] = {
                "device_map": self.device_map,
                **({"dtype": resolved_dtype} if resolved_dtype is not None else {}),
            }
        resolved_model = model
        if model_revision is not None and not Path(model).is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:  # pragma: no cover - qwen-asr dependency boundary
                raise RuntimeError("qwen-asr revision pinning requires huggingface-hub") from exc
            # qwen-asr forwards kwargs only to AutoModel, not AutoProcessor. Resolving one
            # immutable local snapshot first binds model and processor to the same revision.
            resolved_model = snapshot_download(repo_id=model, revision=model_revision)
        self.model = Qwen3ASRModel.from_pretrained(resolved_model, **kwargs)

    def _score_domain(self, request: DecodeRequest, *, language: str | None) -> str:
        return score_domain_digest(
            {
                "adapter": self.name,
                "model": self.model_name,
                "modelRevision": self.model_revision,
                "modelArtifactSha256": self.model_artifact_sha256,
                "runtimeRevision": self.runtime_revision,
                "forcedAligner": self.forced_aligner_revision
                or self.forced_aligner_artifact_sha256,
                "qwenAsrVersion": _package_version("qwen-asr"),
                "dtype": self.dtype,
                "deviceMap": self.device_map,
                "maxInferenceBatchSize": self.max_inference_batch_size,
                "maxNewTokens": self.max_new_tokens,
                "returnTimestamps": self.return_timestamps or request.return_timestamps,
                "language": language,
                "request": decode_request_identity(request),
            }
        )

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        context = "\n".join(
            part
            for part in (
                request.initial_prompt,
                ("固有名詞候補: " + "、".join(request.hotwords) if request.hotwords else None),
            )
            if part
        )
        language = qwen_language_name(request.language)
        return_timestamps = request.return_timestamps or self.return_timestamps
        score_domain = self._score_domain(request, language=language)
        results: Any = self.model.transcribe(
            audio=_audio_slice(request),
            context=context,
            language=language,
            return_time_stamps=return_timestamps,
        )
        rows = results if isinstance(results, list) else [results]
        output: list[CandidateEvidence] = []
        for index, row in enumerate(rows):
            text = getattr(row, "text", None)
            language = getattr(row, "language", None)
            timestamps = getattr(row, "time_stamps", None)
            if isinstance(row, dict):
                text = text or row.get("text")
                language = language or row.get("language")
                timestamps = timestamps if timestamps is not None else row.get("time_stamps")
            if not text or not str(text).strip():
                continue
            output.append(
                CandidateEvidence(
                    candidate_id=f"qwen-{index:04d}",
                    text=str(text).strip(),
                    source=self.name,
                    metadata={
                        "adapter": self.name,
                        "model": self.model_name,
                        "modelRevision": self.model_revision,
                        "modelArtifactSha256": self.model_artifact_sha256,
                        "runtimeRevision": self.runtime_revision,
                        "forcedAlignerRevision": self.forced_aligner_revision,
                        "forcedAlignerArtifactSha256": self.forced_aligner_artifact_sha256,
                        "language": getattr(row, "language", None) or language,
                        "returnTimestamps": return_timestamps,
                        "scoreDomain": score_domain,
                        "timeStamps": _jsonable(timestamps),
                        "qwenAsrVersion": _package_version("qwen-asr"),
                        "candidateMultiplicity": "one-transcript-per-input",
                    },
                )
            )
        if not output:
            raise RuntimeError("Qwen3-ASR returned no transcript")
        return output[: request.hypotheses]


@dataclass(frozen=True, slots=True)
class AlignedToken:
    text: str
    start_ms: int
    end_ms: int


class Qwen3ForcedAlignerAdapter:
    name = "qwen3-forced-aligner"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        *,
        dtype: str = "float16",
        device_map: str = "cuda:0",
        model_revision: str | None = None,
        model_artifact_sha256: str | None = None,
        artifact_sha256: str | None = None,
        runtime_revision: str | None = None,
    ) -> None:
        try:
            from qwen_asr import Qwen3ForcedAligner
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("install semantic-asr with the 'qwen' extra") from exc
        local_model = Path(model).expanduser().is_dir()
        _validate_local_snapshot_revision(model, model_revision)
        if model_revision is not None and not local_model:
            model_revision = resolve_hugging_face_revision(
                model,
                model_revision,
                QWEN_FORCED_ALIGNER_REVISIONS,
            )
        elif model_revision is None and not local_model and model in QWEN_FORCED_ALIGNER_REVISIONS:
            model_revision = resolve_hugging_face_revision(
                model,
                None,
                QWEN_FORCED_ALIGNER_REVISIONS,
            )
        if (
            model_artifact_sha256 is not None
            and artifact_sha256 is not None
            and model_artifact_sha256.lower() != artifact_sha256.lower()
        ):
            raise ValueError("model_artifact_sha256 and artifact_sha256 disagree")
        model_artifact_sha256 = _local_artifact_digest(
            model,
            model_artifact_sha256 or artifact_sha256,
            required=True,
        )
        kwargs: dict[str, Any] = {"device_map": device_map}
        resolved_dtype = _torch_dtype(dtype)
        if resolved_dtype is not None:
            kwargs["dtype"] = resolved_dtype
        if model_revision is not None:
            kwargs["revision"] = model_revision
        self.model_name = model
        self.model_revision = model_revision
        self.model_artifact_sha256 = model_artifact_sha256
        self.runtime_revision = (
            None if runtime_revision is None else str(runtime_revision).strip() or None
        )
        self.dtype = str(dtype)
        self.device_map = str(device_map)
        self.model = Qwen3ForcedAligner.from_pretrained(model, **kwargs)

    def align(self, request: DecodeRequest, *, text: str) -> list[AlignedToken]:
        if not text.strip():
            raise ValueError("alignment text must not be empty")
        result: Any = self.model.align(
            audio=_audio_slice(request),
            text=text,
            language=qwen_language_name(request.language),
        )
        rows = result[0] if isinstance(result, list) else result
        output: list[AlignedToken] = []
        for row in rows:
            raw_text = getattr(row, "text", None)
            start = getattr(row, "start_time", None)
            end = getattr(row, "end_time", None)
            if isinstance(row, dict):
                raw_text = raw_text or row.get("text")
                start = start if start is not None else row.get("start_time")
                end = end if end is not None else row.get("end_time")
            if raw_text is None or start is None or end is None:
                continue
            output.append(
                AlignedToken(
                    text=str(raw_text),
                    start_ms=round(float(start) * 1000),
                    end_ms=round(float(end) * 1000),
                )
            )
        if not output:
            raise RuntimeError("Qwen3 Forced Aligner returned no tokens")
        return output
