# Semantic ASR v0.2 integration boundary

This branch is the explicit integration of:

- `codex/semantic-asr-v0.2-adaptive-reranking` (runtime and frontier implementation), and
- `codex/semantic-asr-v0.2-research-stack` (typed research, reproducibility, and governance implementation).

Both Git histories are retained by a real merge commit. For paths independently
implemented on both branches, the adaptive-reranking version is the initial source
of truth because it already contains the broader executable runtime and model-free
test surface. Files unique to the research stack are retained and must either be
connected through explicit adapters or marked experimental; silent duplication is
not treated as integration.

## Preserved research-stack surfaces

The merge explicitly retains the complementary typed and reproducibility modules,
including:

- `adapters_v2.py`, `score_types.py`, and `sequence_scorers.py`;
- `experiment.py`, `model_io.py`, and `research_registry.py`;
- `hard_negatives.py`, `reranking.py`, and `training_v2.py`;
- `koemo_bridge.py` and `planner_v2.py`;
- v0.2 schemas, experiment examples, validation scripts, and their tests.

The runtime branch remains authoritative for independently implemented candidate
pooling, MBR, cascade, long-form, N-gram, and risk-control paths. Compatibility
between the retained research modules and runtime modules is enforced by CI rather
than assumed from filenames.

## Evidence and safety boundary

The immutable evidence rule remains:

```text
observedTranscript != normalizedTranscript
```

Text-only models may rank evidence or propose a candidate. They may not directly
author the observed transcript. Generated candidates require acoustic verification.
Scores, preferences, logits, calibrated probabilities, and sequence likelihoods
remain distinct types until an explicitly fitted calibration layer maps them.

## Validation boundary

One-shot mutation and automatic-merge workflows are deliberately removed. CI,
frontier contracts, release gates, and real-audio workflows remain reviewable and
fail closed. Synthetic and model-free tests demonstrate implementation behavior;
they do not constitute a claim of Japanese real-audio CER improvement. Such claims
require the locked rights-approved benchmark protocol, paired confidence intervals,
and recorded hardware/runtime metadata.
