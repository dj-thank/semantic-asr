# Edge deployment and quantization

## Goal

A deployment artifact is accepted because it preserves task quality under a declared operating
point, not merely because it is smaller or faster.

Supported artifact classes in the deployment manifest are:

```text
PyTorch
TorchAO INT8 / INT4 weight-only
bitsandbytes INT8 / INT4
ONNX
CTranslate2
GGUF
```

## Artifact manifest

Every artifact records:

- source model and immutable revision;
- artifact SHA-256;
- tokenizer SHA-256;
- runtime and version;
- quantization configuration;
- calibration digest;
- build-manifest digest.

A tokenizer or source revision change is a new artifact family and requires a new comparison.

## Comparison requirements

Baseline and candidate must use:

- the same locked test-manifest digest;
- the same sample and group counts;
- the same hardware description;
- the same source model/revision;
- compatible calibration provenance;
- repeated measurements where backend nondeterminism is possible.

KV-cache reuse, batching, and backend-specific kernels can change floating-point reduction order.
Therefore deterministic replay rate is measured rather than assumed.

## Metrics

The deployment gate compares:

```text
candidate top-1 accuracy
pairwise accuracy
semantic loss
meaning-critical error rate
calibration error
AURC
real-time factor
peak memory
artifact size
deterministic replay rate
```

By default, any increase in meaning-critical error rejects the candidate even when aggregate
accuracy is unchanged.

## CLI

Provide baseline and candidate `DeploymentEvaluation` JSON files:

```bash
semantic-asr deployment-gate baseline.json candidate-int4.json \
  --output deployment-decision.json
```

Exit status is `0` when accepted and `2` when rejected. This makes the gate suitable for release
workflows.

A stricter or device-specific policy may be supplied:

```bash
semantic-asr deployment-gate baseline.json candidate.json \
  --policy pixel9a-policy.json
```

## Recommended edge tiers

### CPU-only

1. character and mora N-gram;
2. Semantic MBR;
3. deterministic linear/listwise student;
4. optional small encoder reranker only for ambiguous spans.

### Small GPU / integrated accelerator

1. CPU frontier above;
2. quantized 130M–600M reranker;
3. acoustic verifier on contradiction islands;
4. Qwen3-ASR second ear only when expected information gain justifies the cost.

### Offline teacher

8B–12B models should normally create candidate-locked preferences or probability-cache entries
offline. They are not required in the edge runtime.

## Runtime pressure

```bash
semantic-asr throttle-policy \
  --effort edge-gpu \
  --latency-ratio 1.4 \
  --memory-pressure 0.8 \
  --thermal-pressure 0.7
```

The response resolves candidate count, evidence budget, and enabled model families. Hysteresis is
used when a previous throttle state is supplied, preventing rapid unload/reload oscillation.

## Transformers and TorchAO

Modern Transformers releases expose weight-only TorchAO configurations such as INT4 groups. The
repository deliberately does not hard-code one backend as universally best. Conversion scripts
must preserve raw ranking logits and must write the exact configuration into the artifact manifest.

## CTranslate2 and Whisper

CTranslate2 supports quantized Whisper inference and path-level generation settings. Decoder score
domains, length penalty, beam settings, and runtime revision are part of candidate provenance. A
change in any of these invalidates the old calibration profile.

## llama.cpp / GGUF

A local causal teacher or reranker may be served through llama.cpp. Prompt caching improves
throughput, but backend documentation notes that cache reuse and different batching paths can make
logits non-bit-identical. Deployment evaluation therefore includes repeated runs and deterministic
replay rate.

## Claim boundary

No quantized artifact is shipped as superior until its actual conversion, immutable hash, and
locked-test evaluation have passed `deployment-gate` on the target hardware.
