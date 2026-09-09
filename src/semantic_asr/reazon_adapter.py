# MIT License
#
# Copyright (c) 2026 oboroge0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ---
#
# This license covers the source code of this repository only. Pretrained
# models downloaded into `models/` by `scripts/download_models.py` are
# third-party artifacts distributed under their own licenses -- see
# THIRD_PARTY_NOTICES.md. In particular, the ja->en translation model
# (FuguMT / mojicast-fugumt-ja-en-ct2) is CC BY-SA 4.0 (share-alike), not MIT.

"""Opt-in Japanese ReazonSpeech k2 adapter; no invented confidence or rewriting.

Recognizer settings follow hayamimi's MIT-licensed Japanese route (oboroge0,
commit 35a4d9712dd77bdd9833dbc97eb8209d273af773). See LICENSES/hayamimi-MIT.txt.
The model weights retain their separate Apache-2.0 publisher license.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from .adapters import DecodeRequest
from .audio import decode_audio_window, require_integer
from .contracts import CandidateEvidence
from .revisions import verify_artifact_sha256


class ReazonSpeechK2Adapter:
    """One native top-1 hypothesis per <=30s mono PCM16 16kHz window.

    Construction loads local assets explicitly. Importing the module never loads
    sherpa-onnx, downloads weights or changes the default Whisper profile.
    """

    name = "reazonspeech-k2-sherpa-onnx"
    model_name = "reazon-research/reazonspeech-k2-v2"
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
        files = {}
        for key in ("encoder", "decoder", "joiner"):
            matches = list(self.model_dir.glob(key + "-*.int8.onnx"))
            if len(matches) != 1:
                raise ValueError(f"exactly one {key} INT8 model is required")
            files[key] = str(matches[0])
        tokens = self.model_dir / "tokens.txt"
        if not tokens.is_file() or not tokens.stat().st_size:
            raise ValueError("non-empty model token labels are required")
        self.label_sha256 = hashlib.sha256(tokens.read_bytes()).hexdigest()
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError(
                "install sherpa-onnx and numpy for the opt-in Reazon adapter"
            ) from exc
        self.runtime_revision = (
            "sherpa-onnx-" + importlib.metadata.version("sherpa-onnx") + "+reazon-mbs4-v1"
        )
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            **files,
            tokens=str(tokens),
            num_threads=cpu_threads,
            decoding_method="modified_beam_search",
            max_active_paths=4,
            modeling_unit="cjkchar",
            hotwords_score=2.0,
            provider="cpu",
        )

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        if request.beam_size != 4:
            raise ValueError("this fixed Reazon route requires beam_size=4 (native active paths)")
        if request.language not in {"ja", "Japanese"}:
            raise ValueError("the Reazon route requires an explicit Japanese language")
        if request.initial_prompt or request.hotwords:
            raise ValueError("prompt and hotword conditioning are unsupported by this route")
        if request.return_timestamps:
            raise ValueError("word timestamps are not provided by this adapter")

        def native_only(*args, **kwargs):
            raise ValueError("Reazon input must be mono PCM16 16kHz WAV")

        audio = decode_audio_window(
            request.audio_path,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            decoder=native_only,
        )
        if not 0 < len(audio) <= 30 * 16000:
            raise ValueError("Reazon requires a non-empty window of at most 30 seconds")
        stream = self.recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self.recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if not text:
            return []
        return [
            CandidateEvidence(
                candidate_id="reazon-top1",
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
                    "decodingMethod": "modified_beam_search",
                    "maxActivePaths": 4,
                    "modelingUnit": "cjkchar",
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
                },
            )
        ]
