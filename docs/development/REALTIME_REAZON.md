# Realtime Reazon first pass (Issue #64)

Status: **opt-in engineering path**. This is not a promoted default and does not claim a measured Semantic ASR accuracy or latency improvement yet.

## Why this exists

Semantic ASR already has a pinned local `ReazonSpeechK2Adapter` and Hayamimi-derived conservative CJK ITN, but the public runtime is primarily whole-file / long-form. Hayamimi demonstrates a useful realtime shape for local Japanese ASR: Silero VAD, ReazonSpeech on CPU, in-progress partials, fast first-pass finals, and a later refine pass.

This path adopts that **shape**, not Hayamimi's benchmark claims. Hayamimi's reported latency/CER numbers are measurements of its own code, models, data and hardware. They are not Semantic ASR results.

## Contract

```text
PCM16 16 kHz
   |
   +--> caller VAD decision
   |
   +--> RealtimeReazonSession
           |
           +--> speech_start
           +--> partial      (display-only; mutable/provisional)
           +--> final        (first-pass; exact PCM SHA-256 bound)
           +--> refine       (optional child of final; never mutates final)
           +--> warning
           +--> session_summary
```

Three text layers must not be collapsed:

1. `partial`: responsive UI text. It is explicitly not immutable observed evidence.
2. `final`: first-pass acoustic observation for one exact PCM buffer. Its event digest and audio digest remain stable even if later processing disagrees.
3. `refine`: a new event referencing both the same audio digest and the parent final digest. A linguistic/context model cannot silently rewrite the first-pass observation.

This preserves the repository invariant `observedTranscript != normalizedTranscript` and extends it to realtime UI state.

## Defaults and provenance

`RealtimeReazonConfig` uses:

| setting | default | role |
|---|---:|---|
| sample rate | 16,000 Hz | current Reazon adapter contract |
| partial interval | 500 ms | bounded display refresh |
| end silence | 350 ms | endpointing when a caller supplies raw speech flags |
| max speech | 12 s | finite buffer / latency bound |
| pre-roll | 800 ms | protect the utterance head when VAD activates late |
| refine idle | 2 s | defer expensive second pass |
| history | 16 utterances | bound retained PCM/evidence |

The values follow the useful operating shape of `oboroge0/hayamimi`'s Japanese realtime pipeline. They are starting defaults, not a claim that its measurements transfer to this repository.

## Runner

`scripts/realtime_reazon.py` is a deliberately explicit local research entry point. It requires:

- mono PCM16 16 kHz WAV input;
- a local ReazonSpeech k2 model directory plus its exact artifact SHA-256;
- a local Silero VAD ONNX file plus its exact SHA-256;
- `--allow-local-research`;
- the `sherpa` optional dependency.

Example:

```bash
python scripts/realtime_reazon.py sample.wav \
  --model-dir /path/to/reazon-k2 \
  --model-sha256 <64-hex> \
  --vad-model /path/to/silero_vad.onnx \
  --vad-model-sha256 <64-hex> \
  --events-jsonl runs/realtime-events.jsonl \
  --allow-local-research
```

Use `--realtime` to sleep at source-audio pace. Without it, the WAV is driven as fast as the host permits for engineering evaluation.

The first runner intentionally bridges each partial/final PCM buffer through a temporary WAV into the existing file-based `ReazonSpeechK2Adapter`. That is not expected to be the optimal latency path. It avoids creating a second decoder implementation before #39 has measured where the time goes. If profiling shows temporary I/O is material, the next patch should add one tested in-memory decoder seam and prove output/evidence equivalence before replacing this bridge.

## VAD layering

The runner uses sherpa-onnx Silero with the same core shape as Hayamimi: 512-sample windows, threshold 0.5, 0.25 s minimum speech, 0.35 s silence and a 12 s maximum segment. Silero therefore owns the main endpoint. `RealtimeReazonSession` closes on the first non-speech chunk after that transition instead of adding another 350 ms endpoint delay.

No model is downloaded automatically. The runner verifies the exact local VAD file and Reazon model directory before inference.

## Evidence and failure rules

- Partial decodes can change freely and are never reused as final confidence.
- Empty decoder text remains empty; the runtime does not invent speech.
- A final binds the exact PCM bytes, start/end sample indices and SHA-256.
- A refine event requires the matching final ID/digest and matching audio digest.
- Refiner failure emits a warning and leaves the final untouched.
- Chunks larger than the configured bound, malformed PCM, unsupported sample rate and invalid timing/config values fail closed.
- Session reset clears PCM history, final history, refine state and sequence identity.
- No transcript correctness probability is fabricated from native Reazon output.

## Measurement plan

Engineering completion and model promotion are separate.

For #39, use one frozen manifest/model/VAD configuration and record at least:

- cold and warm model load separately;
- speech-end -> first final p50/p95;
- partial decode p50/p95 and revision rate;
- total RTF and peak RSS;
- first-pass strict/lenient CER;
- refine strict/lenient CER;
- semantic-critical error count;
- false-correction / harm / improve / tie;
- exact model, runtime, VAD and source-audio identities.

A latency optimization is accepted only if the same quality/evidence contract still passes. A quality change is promoted only through #29's frozen paired evaluation gates. A successful realtime runner by itself is neither an accuracy result nor a production promotion.

## Next bounded patches

1. Run CI/model-free contract tests for this slice.
2. Add a characterization test for the existing temporary-WAV Reazon bridge.
3. Measure bridge overhead vs decoder time on fixed local audio.
4. Only if material, add an in-memory Reazon decode path with byte-identical preprocessing/evidence tests.
5. Attach an actual Semantic ASR second-pass refiner as an opt-in child event.
6. Evaluate partial/final/refine quality and latency on rights-cleared fixed audio under #29/#39.
7. Add microphone/WebSocket transport only after the event/evidence contract is stable; transport must not become another source of transcript semantics.
