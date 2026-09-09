"""Opt-in Japanese Parakeet CTC adapter with provenance-bound top-one output."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from .adapters import DecodeRequest
from .audio import decode_audio_window, require_integer
from .contracts import CandidateEvidence
from .revisions import verify_artifact_sha256


class ParakeetJapaneseCtcAdapter:
    """Decode one native Parakeet CTC hypothesis from a <=30s Japanese window.

    The adapter is intentionally unscored: CTC text output is evidence from an
    independent engine, not a probability comparable with Whisper or Qwen.
    """

    name = "parakeet-ja-ctc-sherpa-onnx"
    model_name = "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
    model_revision = None
    device = "cpu"
    compute_type = "int8"
    supports_hotwords = False
    supports_initial_prompt = False

    def __init__(self, model_dir: str | Path, *, artifact_sha256: str, cpu_threads: int = 2):
        require_integer(cpu_threads, name="cpu_threads", minimum=1)
        self.cpu_threads = cpu_threads
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.model_artifact_sha256 = verify_artifact_sha256(self.model_dir, artifact_sha256)
        models = list(self.model_dir.glob("model*.onnx"))
        if len(models) != 1:
            raise ValueError("exactly one Parakeet model*.onnx file is required")
        tokens = self.model_dir / "tokens.txt"
        if not tokens.is_file() or not tokens.stat().st_size:
            raise ValueError("non-empty model token labels are required")
        self.model_path = models[0]
        self.tokens_path = tokens
        self.label_sha256 = hashlib.sha256(tokens.read_bytes()).hexdigest()
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "install sherpa-onnx and numpy for the opt-in Parakeet adapter"
            ) from exc
        self.runtime_revision = "sherpa-onnx-" + importlib.metadata.version("sherpa-onnx")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(self.model_path),
            tokens=str(self.tokens_path),
            num_threads=cpu_threads,
            provider="cpu",
        )

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        if request.beam_size != 1 or request.hypotheses != 1:
            raise ValueError("this native Parakeet route requires beam_size=1 and hypotheses=1")
        if request.language not in {"ja", "Japanese"}:
            raise ValueError("the Parakeet route requires an explicit Japanese language")
        if request.initial_prompt or request.hotwords:
            raise ValueError("prompt and hotword conditioning are unsupported by this route")

        def native_only(*args, **kwargs):
            raise ValueError("Parakeet input must be mono PCM16 16kHz WAV")

        audio = decode_audio_window(
            request.audio_path,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            decoder=native_only,
        )
        if not 0 < len(audio) <= 30 * 16000:
            raise ValueError("Parakeet requires a non-empty window of at most 30 seconds")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if not text:
            return []
        return [
            CandidateEvidence(
                candidate_id="parakeet-top1",
                text=text,
                rank=1,
                hypothesis_count=1,
                source=self.name,
                metadata={
                    "model": self.model_name,
                    "modelArtifactSha256": self.model_artifact_sha256,
                    "runtimeRevision": self.runtime_revision,
                    "labelSetSha256": self.label_sha256,
                    "device": self.device,
                    "computeType": self.compute_type,
                    "cpuThreads": self.cpu_threads,
                    "decodingMethod": "ctc-greedy",
                    "availableHypotheses": 1,
                    "requestedHypotheses": request.hypotheses,
                    "scoreKind": "unscored-transcript",
                    "audioWindowSha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
                    "startMs": request.start_ms or 0,
                    "sampleCount": len(audio),
                    "durationSeconds": len(audio) / 16000,
                    "language": "ja",
                    "observedRewriting": False,
                    "numericNormalization": False,
                    "timestampsProvided": False,
                    "timestampsRequested": request.return_timestamps,
                },
            )
        ]
