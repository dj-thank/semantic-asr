"""Keep Whisper's own transcription pipeline as an explicit primary observation."""

from __future__ import annotations

import hashlib

from .adapters import DecodeRequest, _package_version
from .advanced_adapters import PathPreservingFasterWhisperAdapter
from .audio import decode_audio_window
from .contracts import CandidateEvidence


class NativeWhisperAdapter(PathPreservingFasterWhisperAdapter):
    """Reuse the pinned model loader but preserve the native decoder's output.

    Native segment scores are retained as diagnostics. They are not converted to
    a whole-window correctness probability or mixed with N-best path scores.
    """

    name = "faster-whisper-native-observation"

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        if request.hypotheses != 1:
            raise ValueError("native Whisper produces one hypothesis per window")
        if request.return_timestamps:
            raise ValueError("this route exposes utterance times, not word timestamps")
        from faster_whisper.audio import decode_audio

        audio = decode_audio_window(
            request.audio_path,
            start_ms=request.start_ms,
            end_ms=request.end_ms,
            decoder=decode_audio,
        )
        if not 0 < len(audio) <= 480000:
            raise ValueError("native Whisper requires a nonempty window of at most 30 seconds")
        segments, _ = self.model.transcribe(
            audio,
            language=request.language,
            beam_size=request.beam_size,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
            initial_prompt=request.initial_prompt,
            hotwords="、".join(request.hotwords) if request.hotwords else None,
        )
        segments = list(segments)
        text = "".join(segment.text for segment in segments).strip()
        if not text:
            return []
        start = request.start_ms or 0
        spans = [
            {
                "text": segment.text.strip(),
                "startMs": start + round(segment.start * 1000),
                "endMs": start + round(segment.end * 1000),
            }
            for segment in segments
        ]
        return [
            CandidateEvidence(
                candidate_id="native-whisper-top1",
                text=text,
                rank=1,
                hypothesis_count=1,
                source=self.name,
                metadata={
                    "model": self.model_name,
                    "modelRevision": self.model_revision,
                    "modelArtifactSha256": self.model_artifact_sha256,
                    "runtimeRevision": self.runtime_revision,
                    "fasterWhisperVersion": _package_version("faster-whisper"),
                    "ctranslate2Version": _package_version("ctranslate2"),
                    "scoreKind": "unscored-transcript",
                    "nativeOutputPreserved": True,
                    "nativeSegmentScores": [
                        {
                            "avgLogprob": segment.avg_logprob,
                            "noSpeechProbability": segment.no_speech_prob,
                            "tokens": list(segment.tokens),
                        }
                        for segment in segments
                    ],
                    "utteranceSpans": spans,
                    "audioWindowSha256": hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
                    "startMs": start,
                    "sampleCount": len(audio),
                    "sampleRate": 16000,
                    "language": request.language,
                    "beamSize": request.beam_size,
                    "computeType": self.compute_type,
                    "device": self.device,
                    "cpuThreads": self.cpu_threads,
                    "observedRewriting": False,
                },
            )
        ]
