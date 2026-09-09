"""Local, bounded audio-only phone observations using the existing CTC contract.

No reference or candidate text enters the encoder. Morae are a lossless grouping
of the same decoded phone evidence, not a second acoustic vote. Frame-run times
are encoder-grid estimates, not gold phone boundaries or forced-alignment proof.
Optional model libraries are imported only when an observation is requested.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio import require_integer, validate_audio_span
from .contracts import canonical_json, sha256_json
from .mora_phonology import phones_to_moras
from .phonetic_evidence import (
    CandidatePronunciation,
    PosteriorFrame,
    PosteriorSequence,
    ctc_pronunciation_score,
)
from .revisions import sha256_artifact


def _digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _window_end_sample(frame_count: int, end_ms: int | None) -> int:
    # The long-form duration contract rounds to milliseconds. Only that exact
    # recording-end request may include its remaining fractional millisecond.
    if end_ms is None or end_ms == round(frame_count / 16):
        return frame_count
    return end_ms * 16


def decoded_phone_runs(posterior: PosteriorSequence) -> list[dict]:
    """Collapse adjacent labels before removing blanks; preserve blank-separated repeats."""
    if posterior.kind != "phone":
        raise ValueError("phone decoding requires the phone label domain")
    runs: list[dict] = []
    previous = None
    for frame in posterior.frames:
        symbol, probability = max(frame.probabilities, key=lambda item: (item[1], item[0]))
        if symbol != posterior.blank_symbol:
            if symbol == previous:
                runs[-1]["end_ms"] = frame.end_ms
                runs[-1]["minimum_frame_posterior"] = min(
                    runs[-1]["minimum_frame_posterior"], probability
                )
            else:
                runs.append(
                    {
                        "phone": symbol,
                        "start_ms": frame.start_ms,
                        "end_ms": frame.end_ms,
                        "minimum_frame_posterior": probability,
                    }
                )
        previous = symbol
    return runs


def observe_transcript_phones(observer, audio_path, longform, *, max_windows=16, max_seconds=120):
    """Collect auxiliary evidence without editing the first-pass transcript.

    The wall check is between bounded calls, not an OS-level preemption guarantee.
    Every skipped/failed window stays in the coverage denominator.
    """
    require_integer(max_windows, name="max_windows", minimum=1)
    require_integer(max_seconds, name="max_seconds", minimum=1)
    start = time.monotonic()
    rows = []
    for index, segment in enumerate(longform.segments):
        row = {
            "index": index,
            "start_ms": segment.window.start_ms,
            "end_ms": segment.window.end_ms,
            "status": "provisional",
        }
        try:
            if index >= max_windows or time.monotonic() - start >= max_seconds:
                raise TimeoutError("phone evidence budget exhausted")
            observed = observer.observe(
                audio_path, start_ms=segment.window.start_ms, end_ms=segment.window.end_ms
            )
            if observed.posterior.source_audio_sha256 != longform.source_audio_sha256:
                raise ValueError("phone evidence belongs to another audio source")
            with wave.open(str(audio_path), "rb") as source:
                expected_end = _window_end_sample(source.getnframes(), segment.window.end_ms)
            if (observed.window_start_sample, observed.window_end_sample) != (
                segment.window.start_ms * 16,
                expected_end,
            ):
                raise ValueError("phone evidence belongs to another window")
            row.update(execution="observed", observation=observed.as_dict())
            scorer = getattr(observer, "score_candidates", None)
            if scorer is not None:
                try:
                    scored = scorer(observed, segment)
                    if scored is not None:
                        row["candidate_checks"] = scored
                except (EOFError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    row["candidate_checks"] = {
                        "execution": "unavailable",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
        except (
            EOFError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            wave.Error,
        ) as exc:
            row.update(execution="unavailable", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    payload = {
        "source_audio_sha256": longform.source_audio_sha256,
        "first_pass_evidence_sha256": longform.evidence_sha256,
        "windows": rows,
        "completed_windows": sum(r["execution"] == "observed" for r in rows),
        "total_windows": len(rows),
        "seconds": time.monotonic() - start,
        "max_windows": max_windows,
        "max_seconds": max_seconds,
        "transcript_changed": False,
        "correctness_calibrated": False,
    }
    return {**payload, "digest": sha256_json(payload)}


@dataclass(frozen=True, slots=True)
class AudioPhoneObservation:
    posterior: PosteriorSequence
    window_start_sample: int
    window_end_sample: int
    sample_rate: int
    input_pcm_sha256: str
    preprocessing_sha256: str
    model_artifact_sha256: str
    runtime_revision: str
    frame_stride_samples: int
    receptive_field_samples: int

    def __post_init__(self) -> None:
        for name in (
            "window_start_sample",
            "window_end_sample",
            "sample_rate",
            "frame_stride_samples",
            "receptive_field_samples",
        ):
            require_integer(
                getattr(self, name), name=name, minimum=0 if name == "window_start_sample" else 1
            )
        if self.sample_rate != 16000 or self.window_end_sample <= self.window_start_sample:
            raise ValueError("observation requires a nonempty 16 kHz window")
        for name in ("input_pcm_sha256", "preprocessing_sha256", "model_artifact_sha256"):
            _digest(getattr(self, name), name)
        if not self.runtime_revision:
            raise ValueError("runtime revision is required")
        if self.posterior.kind != "phone":
            raise ValueError("phone observer cannot relabel a mora posterior")
        if self.posterior.encoder_revision != "artifact:" + self.model_artifact_sha256:
            raise ValueError("posterior model differs from observation model")
        start_ms = self.window_start_sample * 1000 / self.sample_rate
        end_ms = self.window_end_sample * 1000 / self.sample_rate
        for i, frame in enumerate(self.posterior.frames):
            if frame.start_ms < math.floor(start_ms) or frame.end_ms > math.ceil(end_ms):
                raise ValueError("posterior frame lies outside the audio window")
            expected = round(start_ms + i * self.frame_stride_samples * 1000 / self.sample_rate)
            expected_end = min(
                round(start_ms + (i + 1) * self.frame_stride_samples * 1000 / self.sample_rate),
                math.ceil(end_ms),
            )
            if frame.start_ms != expected or frame.end_ms != expected_end:
                raise ValueError("posterior frame grid differs from the frozen encoder grid")

    @property
    def digest(self) -> str:
        return sha256_json(self._binding())

    def _binding(self) -> dict:
        return {
            "posterior_digest": self.posterior.digest,
            "window_start_sample": self.window_start_sample,
            "window_end_sample": self.window_end_sample,
            "sample_rate": self.sample_rate,
            "input_pcm_sha256": self.input_pcm_sha256,
            "preprocessing_sha256": self.preprocessing_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "runtime_revision": self.runtime_revision,
            "frame_stride_samples": self.frame_stride_samples,
            "receptive_field_samples": self.receptive_field_samples,
        }

    def as_dict(self, *, include_posterior: bool = False) -> dict:
        runs = decoded_phone_runs(self.posterior)
        moras = []
        for unit in phones_to_moras(tuple(r["phone"] for r in runs)):
            item = asdict(unit)
            item["start_ms"] = runs[unit.phone_start]["start_ms"]
            item["end_ms"] = runs[unit.phone_end - 1]["end_ms"]
            moras.append(item)
        result = {
            "schema": "semantic-asr-audio-phone-observation-v1",
            **self._binding(),
            "observation_digest": self.digest,
            "source_audio_sha256": self.posterior.source_audio_sha256,
            "status": "provisional",
            "phones": runs,
            "moras": moras,
            "mora_evidence": "derived-from-same-phone-posterior-not-independent",
            "timing": "encoder-grid estimate, not verified phone boundaries",
            "posterior_semantics": "frame label posterior, not transcript correctness probability",
            "reference_used_by_encoder": False,
        }
        if include_posterior:
            result["posterior"] = asdict(self.posterior)
        return result


class LocalHubertPhoneObserver:
    """Provision weights separately, then observe one bounded PCM16 WAV window.

    This intentionally reuses the pure-Python posterior/CTC reference contract.
    It neither downloads a model nor replaces an ASR transcript.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        artifact_sha256: str,
        max_audio_seconds: int = 30,
        check_candidates: bool = False,
    ):
        _digest(artifact_sha256, "model artifact")
        require_integer(max_audio_seconds, name="max_audio_seconds", minimum=1)
        if max_audio_seconds > 30:
            raise ValueError("phone observations are bounded to 30 seconds per window")
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(self.model_dir)
        self.artifact_sha256 = artifact_sha256
        self.max_audio_seconds = max_audio_seconds
        if not isinstance(check_candidates, bool):
            raise TypeError("check_candidates must be boolean")
        self.check_candidates = check_candidates
        self._loaded = None

    def score_candidates(self, observation: AudioPhoneObservation, segment) -> dict | None:
        if not self.check_candidates:
            return None
        import pyopenjtalk

        revision = importlib.metadata.version("pyopenjtalk-plus")

        def pronounce(candidate):
            return CandidatePronunciation.create(
                candidate_id=candidate.candidate_id,
                text=candidate.text,
                kind="phone",
                symbols=pyopenjtalk.g2p(candidate.text).split(),
                producer="pyopenjtalk-plus",
                producer_revision=revision,
            )

        return check_candidate_pronunciations(observation, segment, pronounce=pronounce)

    def _load(self):
        if self._loaded is None:
            from transformers import HubertForCTC, Wav2Vec2FeatureExtractor

            if sha256_artifact(self.model_dir) != self.artifact_sha256:
                raise ValueError("phone model artifact changed")
            vocab = json.loads((self.model_dir / "vocab.json").read_text(encoding="utf-8"))
            model = (
                HubertForCTC.from_pretrained(
                    str(self.model_dir), local_files_only=True, use_safetensors=True
                )
                .float()
                .cpu()
                .eval()
            )
            model.requires_grad_(False)
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            if sorted(vocab.values()) != list(range(model.config.vocab_size)):
                raise ValueError("phone model label IDs do not match its head")
            ordered = tuple(label for label, _ in sorted(vocab.items(), key=lambda item: item[1]))
            if extractor.sampling_rate != 16000 or not 0 <= model.config.pad_token_id < len(
                ordered
            ):
                raise ValueError("unsupported phone preprocessing or blank label")
            if sha256_artifact(self.model_dir) != self.artifact_sha256:
                raise ValueError("phone model artifact changed during loading")
            self._loaded = model, extractor, ordered
        return self._loaded

    def observe(
        self, audio_path: str | Path, *, start_ms: int = 0, end_ms: int | None = None
    ) -> AudioPhoneObservation:
        validate_audio_span(start_ms, end_ms)
        path = Path(audio_path)
        source_before = _source_digest(path)
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
                stream.getcomptype(),
            ) != (1, 2, 16000, "NONE"):
                raise ValueError("phone runtime requires mono 16 kHz PCM16 WAV")
            first = start_ms * 16
            last = _window_end_sample(stream.getnframes(), end_ms)
            if not 0 <= first < last <= stream.getnframes():
                raise ValueError("requested phone window is outside the recording")
            if last - first > self.max_audio_seconds * 16000 or last - first < 400:
                raise ValueError("phone window exceeds duration budget or is too short")
            stream.setpos(first)
            pcm = stream.readframes(last - first)
            if len(pcm) != (last - first) * 2:
                raise ValueError("truncated phone window")
        import numpy as np
        import torch

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        model, extractor, vocab = self._load()
        batch = extractor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
        with torch.no_grad():
            logits = model(**batch).logits[0]
            expected = int(model._get_feat_extract_output_lengths(torch.tensor(len(audio))))
            if logits.shape != (expected, len(vocab)) or not torch.isfinite(logits).all():
                raise ValueError("phone encoder output shape or values are invalid")
            probabilities = logits.double().softmax(-1).cpu().numpy()
        stride = 1
        receptive = 1
        for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride, strict=True):
            receptive += (kernel - 1) * stride
            stride *= step
        frames = tuple(
            PosteriorFrame.from_mapping(
                start_ms=round(start_ms + i * stride / 16),
                end_ms=min(round(start_ms + (i + 1) * stride / 16), math.ceil(last / 16)),
                probabilities=dict(zip(vocab, row.tolist(), strict=True)),
            )
            for i, row in enumerate(probabilities)
        )
        source_after = _source_digest(path)
        if source_before != source_after:
            raise ValueError("phone audio source changed during observation")
        posterior = PosteriorSequence(
            "phone",
            vocab[model.config.pad_token_id],
            vocab,
            frames,
            "local-hubert-phone-ctc",
            "artifact:" + self.artifact_sha256,
            sha256_json(vocab),
            source_before,
        )
        preprocessing = sha256_json(
            {
                "pcm": "mono-s16le/32768",
                "sample_rate": 16000,
                "extractor": extractor.to_dict(),
                "padding": False,
            }
        )
        runtime = ";".join(
            f"{name}={importlib.metadata.version(name)}"
            for name in ("torch", "transformers", "numpy")
        )
        return AudioPhoneObservation(
            posterior,
            first,
            last,
            16000,
            hashlib.sha256(pcm).hexdigest(),
            preprocessing,
            self.artifact_sha256,
            runtime,
            stride,
            receptive,
        )


def check_candidate_pronunciations(
    observation, segment, *, pronounce, max_candidates=8, max_seconds=10
) -> dict:
    """Check known whole-window candidates; G2P is a proposal, never an observation.

    Homophones remain unresolved here. A later frozen contextual preference can
    consume the recorded scores; no candidate is selected or transcript changed.
    """
    require_integer(max_candidates, name="max_candidates", minimum=1)
    require_integer(max_seconds, name="max_seconds", minimum=1)
    if max_candidates > 8 or max_seconds > 30:
        raise ValueError("candidate checking exceeds its bounded runtime budget")
    if observation.posterior.source_audio_sha256 != segment.observed.source_audio_sha256:
        raise ValueError("candidate check belongs to another recording")
    if (
        observation.window_start_sample != segment.window.start_ms * 16
        or round(observation.window_end_sample / 16) != segment.window.end_ms
    ):
        raise ValueError("candidate check belongs to another window")
    started = time.monotonic()
    rows = []
    for index, candidate in enumerate(segment.observed.candidates):
        row = {
            "candidate_id": candidate.candidate_id,
            "text_sha256": hashlib.sha256(candidate.text.encode("utf-8")).hexdigest(),
        }
        try:
            if index >= max_candidates or time.monotonic() - started >= max_seconds:
                raise TimeoutError("candidate phone score budget exhausted")
            metadata = candidate.metadata or {}
            if (metadata.get("decodeStartMs"), metadata.get("decodeEndMs")) != (
                segment.window.start_ms,
                segment.window.end_ms,
            ):
                raise ValueError("candidate does not declare matching whole-window coverage")
            if len(candidate.text) > 512:
                raise ValueError("candidate text exceeds phone check budget")
            pronunciation = pronounce(candidate)
            if (pronunciation.candidate_id, pronunciation.text) != (
                candidate.candidate_id,
                candidate.text,
            ):
                raise ValueError("pronunciation belongs to another candidate")
            if len(pronunciation.symbols) > 512:
                raise ValueError("candidate pronunciation exceeds phone check budget")
            score = ctc_pronunciation_score(observation.posterior, pronunciation)
            serialized_score = json.loads(canonical_json(score))
            serialized_score["evidence"] = score.evidence.as_dict()
            row.update(
                execution="scored",
                pronunciation=asdict(pronunciation),
                score=serialized_score,
            )
        except (ImportError, RuntimeError, OSError, TypeError, ValueError) as exc:
            row.update(execution="unavailable", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    payload = {
        "observation_digest": observation.digest,
        "baseline_candidate_id": segment.observed.selected_candidate_id,
        "candidates": rows,
        "total_candidates": len(rows),
        "scored_candidates": sum(r["execution"] == "scored" for r in rows),
        "seconds": time.monotonic() - started,
        "reference_kind": "G2P pronunciation proposal, not acoustic observation",
        "context_resolved": False,
        "text_selection_applied": False,
    }
    return {**payload, "digest": sha256_json(payload)}
