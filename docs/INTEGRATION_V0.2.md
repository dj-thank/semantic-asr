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

The immutable evidence rule remains:

```text
observedTranscript != normalizedTranscript
```

Text-only models may rank evidence or propose a candidate. They may not directly
author the observed transcript. Generated candidates require acoustic verification.

One-shot mutation and automatic-merge workflows are deliberately removed. CI,
frontier contracts, release gates, and real-audio workflows remain reviewable and
fail closed.
