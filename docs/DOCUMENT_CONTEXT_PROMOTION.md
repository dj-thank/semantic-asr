# Document-context scorer registration and promotion gate

## Why a separate registration object exists

An experiment protocol that records only `scorer_key="model"` is not frozen. The runtime could load
a different checkpoint under the same key and still claim to have executed the registered arm.

`FrozenScorerRegistry` binds every scorer key to its exact `profile_digest`. A
`DocumentExperimentRegistration` then binds three identities before evaluation:

```text
protocol digest
manifest digest
scorer-registry digest
```

`run_registered_document_context_experiment()` rejects any changed protocol, dataset manifest,
scorer key, or scorer profile before opening reference metrics.

For a neural scorer, the profile digest should include at least:

- architecture and tokenizer identifiers;
- immutable model revision or weight SHA-256;
- prompt/input serialization revision;
- context-window and truncation policy;
- score extraction semantics;
- held-out normalization artifact digest;
- runtime implementation revision.

A mutable model alias such as `latest` is not an acceptable experiment identity.

## Privacy-safe reports

The in-memory result retains selected text so metrics and local inspection remain possible. The
report writer defaults to hashes for selected document and window text. Raw selected text is emitted
only when `include_text=True` is explicitly requested and the operator has confirmed the output
rights and destination.

References are never written into the report by this package. Their text is represented by the
frozen reference digest and aggregate metrics.

## Promotion is conjunctive

`DocumentContextPromotionPolicy` is deliberately fail-closed. The target arm is promoted only when
all preregistered checks pass:

1. minimum absolute strict-CER reduction over the base arm;
2. paired bootstrap upper bound below the allowed delta;
3. ordered context beats the shuffled-window control by the required margin;
4. semantic-critical token errors do not regress beyond allowance;
5. context-induced false corrections do not regress beyond allowance;
6. introduced error characters do not regress beyond allowance;
7. accepted coverage remains above the minimum;
8. mean target-arm latency remains below the limit;
9. no case/arm evaluation failed, unless failures were explicitly allowed before evaluation.

A single mean-CER improvement cannot override a failed safety condition.

## Recommended initial gate

The exact threshold depends on the locked test set. A conservative initial preregistration is:

```python
from semantic_asr.document_experiment.promotion import (
    DocumentContextPromotionPolicy,
)

policy = DocumentContextPromotionPolicy(
    target_arm="ordered-bidirectional",
    baseline_arm="acoustic-only",
    shuffled_control_arm="shuffled-bidirectional",
    minimum_absolute_strict_cer_reduction=0.002,
    maximum_bootstrap_upper_delta=0.0,
    minimum_ordered_advantage_over_shuffled=0.001,
    maximum_critical_error_regression=0,
    maximum_false_correction_regression=0,
    maximum_introduced_error_regression=0,
    minimum_coverage=0.70,
    maximum_mean_latency_ms=30_000.0,
)
```

These numbers are an example, not a repository default. They must be committed with the registration
before test references are evaluated.

## Interpretation

A failed promotion decision is a result, not an infrastructure failure. The complete list of checks
and observed values is digest-bound and should be published together with the experiment report.

Particularly important outcomes include:

- ordered and shuffled arms improve equally: likely generic language prior rather than discourse;
- CER improves but false corrections rise: unsafe context overreach;
- accepted CER improves while coverage collapses: abstention trade-off, not a universal gain;
- critical errors rise despite lower CER: reject for high-stakes transcription;
- target arm wins only with mutable external context: insufficiently reproducible;
- latency exceeds the registered device budget: research result only, not deployment candidate.
