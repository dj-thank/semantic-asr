# Real acoustic-head and LoRA weight feasibility pilot

Related: #23, #26, #35, #36, #37. This is an executed feasibility experiment, not
completion of the full training infrastructure or a publication-quality accuracy result.

## What actually ran

Run `33955411922`, source `5c3a99498e1c8c27ad15874d244e8422e5aba75d`:
https://github.com/dj-thank/semantic-asr/actions/runs/33955411922

FLEURS Japanese **official train only**: 32 train + 8 development recordings,
329.76 seconds total. Text ID modulo five defines development; duplicate audio,
text ID and reference hashes are rejected. All 72 examples exposed in the prior
phonetic studies are excluded. Train/development and prior-exposure intersections
were checked again after download. No official validation/test was opened.
Speaker separation and absence of base-model pretraining contamination are NOT proven.

The fixed protocol was recorded in #35 before execution: seed 17, no search,
128 acoustic-head updates (learning rate 0.003), 16 LoRA updates (0.0002), final-step
checkpoints, maximum 400 audio seconds and 1,800 seconds between-launch wall budget.
Total training-script execution including collection was 533.12 seconds. Synchronous
calls are not preempted by this check. The hosted job has a separate 40-minute cap.

Initial run `33955011038` failed before training because the pinned FLEURS license
was represented as a one-item list. The corrected guard accepts exactly CC-BY-4.0,
as either that list or a string; other licenses still fail. The failed artifact is
retained. The successful inference/training script is subsequently formatting-only
normalized; its AST was compared and is unchanged.

## New trained weights (not a downloaded or coefficient-fit substitute)

| Component | Trainable parameters | Updates | Weight-change L2 | Frozen base |
|---|---:|---:|---:|---|
| Newly initialized phone + mora linear CTC heads | 193,788 | 128 | 41.05544 | HuBERT unchanged |
| Qwen3-0.6B-Base rank-8 LoRA, q/v in layers 26–27 | 81,920 | 16 | 0.296379 | Qwen base unchanged |

HuBERT features come from the pinned audio-only encoder. The new heads use the
existing `SemanticASRMultiTask`; no duplicate training framework is introduced.
Only phone/mora heads update and are exported. Accent, F0, boundary and preservation
heads have no invented supervision and are excluded from the trained artifact.

Phone targets are Open JTalk weak labels from training text. Mora targets are phone
groups, include pauses, and preserve devoicing. They are neither verified spoken
pronunciation nor a standard gold mora benchmark. Both heads share one encoder and
must not count as independent witnesses merely because there are two outputs.

LoRA trains on **10 informative real Whisper candidate pairs** from the training
split. Pair labels use train references; candidate scoring sees only each candidate,
its candidate-derived reading and its recorded acoustic score. Candidate targets
are fully scored after a separately tokenized fixed prefix, capped at 256 tokens
without silent truncation. Candidate G2P is not independent acoustic proof.
No generated negative sentences substitute for real ASR candidates.

## Development observations: do not overclaim

Acoustic weak-label CTC loss: **24.50418 -> 0.42328**. Phone proxy errors:
**1110 -> 19 / 540 labels**. Mora-group proxy errors: **1395 -> 64 / 309 labels**.
The comparison is against **random-initialized new heads**, not an already trained
HuBERT phone head, Whisper or another strong recognizer. Error counts can exceed
the reference length because insertions count. There is no held-out transcription
CER improvement claim from these numbers.

LoRA: **5 -> 5 character errors / 236 reference characters**, with no changed
selection among eight development recordings. Only three recordings have multiple
candidates. Trainable tensors changed and gradients were finite, but no recognition
benefit was demonstrated. A ten-pair, one-seed, 16-step pilot cannot establish whether
LoRA helps this task; a larger controlled comparison is needed. Do not promote it.

## Saved artifacts and independent reload

`research/weight-pilot-20260905/result.json` records the numerical summary, source,
artifact and checkpoint hashes; `exclusions.json` permanently preserves the input
exclusions so replay does not depend on seven-day old Actions artifacts.

- Artifact ID: `9966336524`; ZIP SHA-256:
  `84390dba589dae36c678244a4a87bd714443943b6fdd7cae4c730b1837e7eb8d`.
- `acoustic-heads.safetensors`: 775,480 bytes, SHA-256
  `7055181ff0466330980643843496b56c10894417957426d437b4944a19c58b79`.
- `lora/adapter_model.safetensors`: 328,752 bytes, SHA-256
  `b34e939f287bd67b00b9321739b2f6e98da5f2e29b853cebbbc25196a55ae55f`.
- Training manifest SHA-256:
  `1900caee8a4fde153397667427a87a7d2c7db6610794096a97e28c4bfeb91dce`.

A **fresh Python process** on the runner loaded both exported artifacts: acoustic
logits and LoRA candidate score differences were exactly zero on the fixed probes.
The downloaded acoustic artifact was additionally reloaded in the independent local
CPU environment, also with zero difference. Every artifact file hash was checked.
`execution.json` is the pre-reload training receipt; `reload-verification.json` is
the separate successful completion proof. Neither proves model quality.

Artifacts include vocabulary, protocol, manifest, parameter hashes/deltas, all loss
steps, candidate outputs, before/after scores, runtime versions and notices. There
are no raw recordings or base weights. A single public-derived feature probe is
included for acoustic reload. Thirty-day Actions retention is not archival storage;
preserve the downloaded bundle for long-term reproduction. Do not put weights in git.

## Reproduction and next-agent instructions

The workflow is now **manual-only**: no automatic retraining on every commit and no
source-writing workflow. Use the explicit pinned environment in
`.github/workflows/real-weight-pilot.yml` (its transitive freeze is not a full hashed
lock). Run from the repository root in a separate environment:

```bash
python scripts/train_public_weight_pilot.py --output runs/weight-pilot-new \
  --exclusions research/weight-pilot-20260905/exclusions.json
python scripts/train_public_weight_pilot.py --output runs/weight-pilot-new --verify
python -m pytest -q tests/test_weight_pilot.py
```

The output directory must not already exist. This pilot is not resumable. Exact
randomness across dependency/hardware revisions is not promised. Model cards and
source constants pin data/encoder/tokenizer/model revisions; retain license notices.

For #35: add full checkpoint resume/RNG/data-sampler state and interrupted-vs-uninterrupted
comparison before closing the infrastructure task. Preserve the finite failure semantics.
For #36: compare the new heads to the original trained phone head, use verified pronunciation
annotations, more data and all three seeds before drawing an acoustic-quality conclusion.
For #37: expand real informative train pairs and independently evaluated multi-candidate
examples, test candidate-order/ID shortcuts, and compare no-LoRA/simple-ranker/LoRA at the
same budget. Integrate a frozen acoustic-retention gate before any production use.
For #26/#29: record these 40 newly exposed source IDs and full manifest lineage; they are not
an unseen publication test. Keep gold/proxy labels, developer diagnostics and final evaluation
separate. None of #35/#36/#37 is closed by this pilot alone.

## Sources and rights

FLEURS, Google, Conneau et al. (2022), CC-BY-4.0:
https://huggingface.co/datasets/google/fleurs

Japanese HuBERT phone base, Apache-2.0 (weak-label assumptions in model card):
https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v4

Qwen3-0.6B-Base, Qwen Team, Apache-2.0:
https://huggingface.co/Qwen/Qwen3-0.6B-Base

LoRA: https://arxiv.org/abs/2106.09685

Use is a research-feasibility pilot. It is not a new foundation model, a zero-error
Japanese recognizer, a comparison to world-best systems, or an approved deployment.
