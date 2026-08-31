# Qwen integration

## Qwen3-ASR

Audit target:

```text
QwenLM/Qwen3-ASR@7c6daf77a2421100f5fb066495372c00129d39ff
qwen-asr API family: 0.0.6
```

The official wrapper accepts local paths, URLs, base64 or `(numpy.ndarray, sample_rate)` tuples, can force a language such as `Japanese`, and can return timestamps when a Qwen3 Forced Aligner is configured.

Semantic ASR maps:

```text
ja / jpn / jp / Japanese / 日本語 → Japanese
auto / empty                       → None
```

A requested span is loaded as an array tuple before calling Qwen. The complete original recording is not silently sent when a local island was requested.

The official high-level wrapper normally yields one transcript per input. This is stored as an independent second-ear candidate, not described as true decoder N-best.

## Qwen3 Forced Aligner

The aligner localizes a supplied hypothesis. It must be compared with free ASR/mora evidence before inferring insertions or deletions.

## Qwen3.8-Flash-Next

Research target:

```text
QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526
```

The official release describes:

```text
Gated DeltaNet + Qwen Sparse Attention
four-branch Gated Residual
N-gram Embedding
Muon/AdamW optimization refinements
```

Semantic ASR uses these as design inspiration for selected evidence access, multi-branch decision fusion and lexical memory. It does not claim to implement QSA, GDN or the model's training optimizer.

Qwen3.8 is served separately and locally. The teacher contract permits only existing candidate IDs, probabilities and abstention.

Security:

- loopback HTTP only
- proxy disabled
- redirect blocked
- structured exact ID set
- no free transcript accepted
- no chain-of-thought persisted
- no observed-text overwrite
- cached abstention remains abstention
