# Japanese backend comparison and training campaign

This extends the fixed-cycle handoff with alternative Japanese recognizers and
public-data training. Personal voice recordings are not campaign input. No
default model, production profile, final evaluation, or publication is promoted
automatically.

## Integrated opt-in Reazon route

`semantic_asr.reazon_adapter.ReazonSpeechK2Adapter` uses local, SHA-256-verified
ReazonSpeech k2 INT8 models through sherpa-onnx. Its configuration follows
[hayamimi](https://github.com/oboroge0/hayamimi/tree/35a4d9712dd77bdd9833dbc97eb8209d273af773).
The adapter produces one native hypothesis and explicitly leaves acoustic
likelihood/confidence unset. It accepts Japanese, mono PCM16 16kHz WAV windows
up to 30 seconds, with an explicit native active-path budget of 4. Requests must use beam_size=4; the CLI also fixes re-listening to this budget. Unsupported prompts, hotwords and word timestamps are rejected.

```bash
python -m pip install sherpa-onnx numpy
python scripts/transcribe_reazon.py recording.wav \
  --model-dir /external/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17 \
  --model-sha256 EXACT_ARTIFACT_SHA256 --output-dir /external/reazon-run \
  --allow-local-research --normalize-numbers
```

This uses the existing `SemanticASRTranscriber` and its observed/normalized
evidence chain. `observed.txt` is unchanged by numeric normalization. The optional
numeric view is saved separately with the parent evidence digest. Its conservative
Japanese/Chinese/Cantonese conversion is ported from hayamimi with the MIT notice
embedded in the module and retained at `LICENSES/hayamimi-MIT.txt`. The upstream
pure-logic tests are retained. Additional regressions fix dropped digits before
万/億 (for example, 二〇二四万円 must become 2024万円) and preserve ambiguous
mixed digit/magnitude expressions instead of changing their numeric value. The model license is separately Apache-2.0; this
does not relicense any dataset as MIT.

The native raw adapter does not yet include hayamimi's conditional head-dropout
retry, live pre-roll, punctuation, language identification, or speaker pipeline.
Those features must keep their own input and evaluation contracts when ported.

## Measured exposed-data comparison

On the same 8 exposed Reazon evaluation clips from the earlier 24-clip smoke run:

| System | Strict corpus CER | Lenient corpus CER |
|---|---:|---:|
| Whisper small | 19.403% | 19.200% |
| ReazonSpeech k2 INT8 | 10.075% | 3.600% |
| Reazon + hayamimi conditional retry | 10.075% | 3.600% |
| Reazon + retry + numeric view | 9.328% | 2.800% |

The conditional retry changed no outputs on these 24 clips, so this run does not
validate its benefit. Raw Reazon inference for all 24 clips took about 5 seconds
on the local CPU, excluding model load. Punctuation/number conventions explain
part of the strict/lenient difference. These are exposed regression clips, not an
unseen quality gate, and not a direct reproduction of hayamimi's published 15-clip
scorecard.

## GPU comparison and fresh adapter training

A fixed 32-clip development cohort was selected from the official `google/fleurs`
Japanese **train** partition at revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Prior source IDs, audio hashes and
reference hashes were excluded where available. No official final test was opened
by this campaign. This remains development data, and speaker independence is not
asserted from clip IDs.

| Frozen recognizer | Strict CER | Lenient CER |
|---|---:|---:|
| Whisper large-v3-turbo | 5.275% | 3.147% |
| Qwen3-ASR 1.7B | 4.265% | 2.910% |
| Qwen3-ASR 0.6B | 7.856% | 6.651% |

These were real V100 16GB FP16 runs. They are not equivalent to the publisher's
vLLM/BF16 benchmark. Qwen 1.7B's interval includes no improvement and it introduced
3 false corrections relative to Whisper, so average CER alone does not authorize
replacement. PyTorch allocator memory does not measure CTranslate2 allocations.

The first fresh training pilot used 96 additional public train clips, 128 optimizer
updates, rank-8 LoRA on q/v projections in the last four Qwen 1.7B text layers, seed
101 and learning rate 1e-4. All 229,376 trainable parameters were tracked; parameter
delta L2 was 1.29276. Frozen base parameters/buffers were byte-identical. Safetensors
weights reloaded in a fresh Python process and reproduced both fixed decode probes.
This is actual adapter training, not optimizer-state resume.

On the development cohort strict edits fell 76→63/1782 (4.265%→3.535%); lenient
edits fell only 49→48/1684 and one lenient false correction appeared. This supports
continuing bounded experiments, not a general accuracy or deployment claim.

## Continuation boundaries

Live compute ownership, budget reservations, source hashes and run paths are in
the task-local `work/campaign/state.json`. Every run has a finite duration, all
failures remain recorded, and the final evaluation must stay isolated from
development-driven model/threshold selection. Public-data licenses and acoustic
reference quality are separate checks. The original #26/#28/#35 dependency and
optimizer-resume requirements remain open; these pilots do not close the full
release roadmap.
