# Training correctness, resume controls and the next scientific experiment

Status: supervision-validation implemented; resume prototype unpublished;
large-scale training, runtime integration and quality promotion remain open.
Related: #23, #35, #36, #37. Baseline: `5eba248c`.

## Why validate the training objective before spending compute?

An acoustic multi-task objective is the weighted sum of its declared losses. With
negative coefficients the optimizer can improve that sum while worsening the
corresponding task. NaN/Inf coefficients invalidate the objective. Silently ignoring
phone labels when no phone head exists creates an apparent run that never trained
the requested task. These are configuration defects, not model-quality experiments.

The published fix centralizes the six weight checks, reuses strict integer checks,
rejects unavailable phone supervision before encoder invocation and validates every
encoder-length tensor even when CTC is absent. Class-index frame targets accept
int32/int64 at the API and are explicitly converted to PyTorch long targets.
A final finite-loss check covers auxiliary losses as well as the existing CTC guard.

Compatibility: valid default weights and heads retain their original initialization
order, objective, tensor names and output structure. Zero weights and entirely ignored
auxiliary labels remain supported. Negative/non-finite/string/bool weights, fractional
lengths and nonexistent phone supervision now fail instead of being coerced or ignored.
This does not fabricate accent/F0 labels or change reference pronunciations.

## Tests and falsification

The new 52-case suite was run before and after repair: 47 failures/5 passes before;
52 passes after. Existing CTC training, mora-label, public-pilot and reload tests also
run. A positive control verifies the same weighted loss and actual gradients for both
trained heads while an unsupervised accent head receives no gradient. A negative
control passes phone labels without a phone head and proves the encoder is not called.
The implementation fails acceptance if those positive behaviors change or malformed
training intent reaches an optimizer as a valid loss. Do not replace failures by skips.

This is engineering evidence. It does not estimate CER, acoustic accuracy, learning
speed or generalization. Existing pilot weights/results are unchanged.

## Resume research: retain the boundary between proof and publication

A separate local I/O prototype compared 12 continuous updates against 5 updates,
checkpoint export, a fresh process and 7 more updates. It exercised the existing CTC
heads and a dropout fixture, all global Python/NumPy/Torch CPU RNGs, scheduler steps
and shuffled epoch boundaries. Full state and history matched in that environment.
Forty local tests checked restoration and rejection. These were synthetic fixtures,
not new public speech training, PEFT/Qwen resume or a GPU reproducibility study.

The connector blocked publication of the new checkpoint codec. Its dependent code
and tests are not included here, and no alternate write route was used. The 588-pass
local prototype count must not be attributed to the published tree. These notes and
Issue #35 preserve the attempted result, limitation and next requirements; there is
no advertised runnable resume command in this publication.

The next implementation must bind parameter names AND optimizer-group order, not just
shapes. PyTorch pairs stored optimizer IDs and live parameters by order without an
extra identity check. Required controls are same-shaped reversed parameters, weights-
only restoration, changed scheduler settings, changed label/data/base identity,
corrupted state and an exception halfway through apply. Rejected loads must preserve
live parameters, optimizer, scheduler, training flags and all tracked randomness.

Use a reviewed tensor/metadata format, a separately trusted artifact digest and explicit
size/type limits. Keep inference-only exports separate from resumable optimizer/RNG/
sampler state. Arbitrary checkpoint hashes do not prove rights or quality. CPU-only
update-boundary proof cannot certify GPU/AMP/distributed or prefetched-worker resumes.
Consult existing `model_io.py`, `training_v2.py` and the real weight pilot before adding
another training/checkpoint abstraction.

## A useful next acoustic/LLM experiment

First resolve data-role and environment prerequisites (#26/#28/#29). Pre-register a
finite training protocol and preserve a final evaluation partition that is not used
for tuning. The previously inspected 72 examples and the pilot's 40 recordings remain
exposed. Unknown speaker independence and pretraining contamination must remain explicit.

For acoustic heads, compare a learned head to the original trained head on the same
audio/label conditions. Include phone-only, mora-only and joint training, a frozen
encoder versus an explicitly scoped adaptation, and a fixed seed list. Separate
human-verified pronunciation labels from G2P weak labels and pause-containing groups.
Two heads sharing one encoder are not independent corroborating acoustic observations.

For the language ranker, first measure how many examples contain a better candidate
than baseline. Report oracle coverage, informative pair count and ties before training.
Compare no-LoRA, a small ranking baseline and LoRA with matched candidate/compute budgets.
Shuffle candidate order and IDs; require the same semantic choices. Add misleading
context and acoustic-retention controls. New parameters without changed/improved
selections are a valid negative result, not proof of benefit.

For each arm, preserve a source/environment/data manifest, actual trainable/frozen
parameter identities, finite-gradient/update evidence, checkpoint/reload receipts and
paired evaluation including harmed cases. Promotion criteria must precede result
inspection. A publication needs those comparisons, not a larger test count alone.

## Primary sources checked 2026-09-06

- PyTorch CTCLoss (target/length constraints and zero-infinity semantics):
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html
- PyTorch CrossEntropyLoss (class-index target dtype and ignored labels):
  https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html
- PyTorch optimizer state mapping:
  https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.state_dict.html
- PyTorch reproducibility limitations:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- PEFT adapter checkpoint content and base-model requirements:
  https://huggingface.co/docs/peft/en/developer_guides/checkpoint

These sources support the engineering constraints. They do not establish novelty or
quality improvement for Semantic ASR. Reused methods, project hypotheses and measured
outcomes must remain distinct in the eventual research report.
