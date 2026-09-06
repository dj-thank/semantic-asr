# Paired evaluation integrity and an audio-grounding research protocol

Related: #23, #29, focused implementation #52, proposed experiment #53.
Baseline: `3d998afdfff460c3abcebedb9a4aec44a677147a`, tree
`5bfef190a98b043614a6a4a9f1df6e651833650f`. This is an evaluation repair and
retrospective replay, not new acoustic/LLM training or a model promotion.

## Why this change is necessary

Three concrete failures were reproduced in the existing `experiment.py`:

1. Pairing used a set intersection: a candidate system missing a difficult recording
   could be compared on only the remaining easy recordings, without an error.
2. Duplicate `(system_id, sample_id)` records were overwritten by input order.
3. `evaluate_claim_gate` could return `passed=True` with NaN critical-error change
   and NaN latency ratio. NaN comparison is not a substitute for validation.

The initial 37 regression cases yielded 30 failures and seven passes on the baseline.
The implementation now rejects these conditions rather than imputing favorable
values, selecting an intersection or hiding failed predictions. Reference text is
never passed into the selector to make a replay succeed.

## Metric contract and compatibility

`paired_bootstrap_comparison()` still estimates the **utterance mean**. Optional
`group_ids` resamples complete speaker/session groups, but does not change the
estimand into an equally weighted mean of group means. Every target-system sample
must have exactly one counterpart and the requested metric. `accepted=False` rows
remain included; selective evaluation needs separate, predeclared coverage reporting.

Pass `expected_sample_ids` from a frozen evaluation manifest. Mutual equality alone
cannot detect a recording missing from BOTH systems. Exact expected-cohort matching
also rejects unexpected extra rows and duplicate expected IDs. A caller-supplied
ID list is not proof of its licensing, independence or unseen status; #26 must bind
that list to the real manifest before publication-quality evaluation.

`paired_error_rate_comparison()` accepts `PairedErrorCounts` with integer reference
length and baseline/candidate edit counts. It estimates:

```text
sum(candidate errors) / sum(reference units)
  - sum(baseline errors) / sum(reference units)
```

This **corpus error rate** differs from averaging each utterance's error rate. For
example, correcting a one-character clip while introducing ten errors in a
100-character clip can improve the utterance mean and worsen corpus CER. Both
aggregations are reported explicitly, never chosen after seeing which one wins.

Counts may exceed reference length because insertions count. Empty-reference rows
retain insertion counts within a positive-denominator group. A silence-only group
makes a resampled error rate undefined; it is rejected with an explicit message,
not discarded or assigned zero error. Use a separately specified silence insertion
metric, and retain its coverage, before claiming robustness on silent recordings.

Both systems' numerators and the reference denominator use the same group draws.
An independent row-level reference implementation checks the optimized group-total
implementation. Splitting one session into more clips must not increase the reported
independent group count. Group IDs alone do not establish speaker independence.

Compatibility details:

- `experiment.py` retains candidate-minus-baseline differences and its historical
  floor/ceil percentile indices. Lower CER is an improvement when the difference
  is negative.
- `benchmark.py` retains left-minus-right differences and its historical interpolated
  percentiles. These conventions are documented rather than silently unified.
- `BootstrapComparison` appends `group_count`, `resampling_unit`, and `aggregation`.
  Old scalar constructors remain supported; group resampling requires an explicit
  group count. Serializers with exact field allowlists must accept these new fields.
- `BootstrapInterval` adds eligible sample, group and explicitly excluded sample
  counts. Jointly undefined annotated-reference metrics remain undefined; a metric
  defined on only one side is an error, not an exclusion.
- The legacy name `probability_candidate_better` is retained. It is the fraction of
  bootstrap draws favoring the candidate, **not** a posterior probability of model
  superiority, a transcript correctness probability, or a p-value.
- Numeric controls and receipts reject bool, text, NaN, infinity, reversed intervals,
  inconsistent paired means and invalid counts. Legacy mutable metrics are validated
  again at consumption. This does not make a receipt cryptographically authentic.
- A one-group comparison can be descriptive, but the existing claim gate cannot pass
  it. Two groups are only a structural minimum, not sufficient statistical power.
- The claim gate is not a publication authorization: unseen data, preregistration,
  rights, label quality, critical errors, risk and compute limits remain required.

## Smaller evaluation work, not faster speech inference

`benchmark.py::paired_group_bootstrap()` previously reevaluated metric callbacks and
rebuilt all sampled row lists on every bootstrap draw. It now evaluates both metrics
once per row and resamples precomputed group sums/counts. The numeric validator is
shared with `experiment.py`; no second score type, dataset registry or training
framework was added.

A fixed 200-row/20-group, 1,000-draw characterization reduced metric callback calls
from **400,800 to 400**, retaining estimate `0.09699999999999999` and interval
`[0.081, 0.1125]`. This measures synthetic in-process bookkeeping. It is not a claim
about model latency, recognition accuracy, or production speedup. Callbacks must
represent pure metrics, not stateful model inference.

## Complete retrospective public-decision audit

The prior two-wave study bundle contains all 96 stored decisions, representing 72
unique source-audio hashes. Its `complete-decision-inputs.jsonl` has SHA-256:

```text
a29f5accb9f617c473d1ab9415b00cd1eecb2fdb806708091f24c72e3d3fc6da
```

`scripts/audit_public_decisions.py` verifies that identity, replays the already-frozen
policy on candidate text/phone/language scores, checks every selected candidate ID,
then evaluates the reference. All 96 selections and saved lenient error counts
reproduced. No model, tokenizer or audio inference runs and no weights are updated.

Each wave/split is reported separately, with strict and lenient corpus CER and
utterance means, harm/improvement/tie counts, changed text, baseline-exact counts
and false-correction numerators/denominators. The report is in
`research/evaluation-integrity-20260906/public-paired-audit.json`; it records input,
policy, normalization, evaluator and script hashes, Python/Unicode versions,
2,000 draws, seed 17 and a 95% interval. No raw references or weights are added to
that Git-tracked report. Public reference attribution remains FLEURS, Google,
Conneau et al. (2022), CC-BY-4.0, as recorded in the original study.

| Original cohort (now exposed) | Lenient errors, baseline -> candidate | Reference units | Corpus-CER change 95% interval |
|---|---:|---:|---:|
| Wave 1 development, 24 recordings | 49 -> 36 | 1,157 | -2.8090 to -0.0982 percentage points |
| Wave 1 test, 24 recordings | 63 -> 60 | 1,183 | -0.9267 to +0.3413 percentage points |
| Wave 2 development, 24 recordings | 45 -> 34 | 1,157 | -2.5339 to 0 percentage points |
| Wave 2 test, 24 recordings | 55 -> 55 | 1,140 | 0 to 0 percentage points |

These are **retrospective clip-group** intervals, not speaker-disjoint publication
results. Small differences from historical interval endpoints reflect this explicitly
recorded resampling implementation, not new predictions. The wave-1 test interval
still includes no improvement; wave 2 remains unchanged. Development results are not
independent evidence because those data informed policy fitting. The two test cohorts
are different and cannot be subtracted from one another as a paired improvement.

Zero observed false corrections are not a zero-risk guarantee: there are only eight
baseline-exact lenient test examples in wave 1 and six in wave 2. Gold phonetic and
semantic labels, session independence and latency observations are absent here.
`promotion='not-evaluated'` and `fresh_publication_test=False` are explicit, even if
a descriptive comparison looks favorable. Previously inspected data stay exposed.

Reproduce from the repository root after extracting the historical study JSONL:

```bash
python scripts/audit_public_decisions.py /path/to/complete-decision-inputs.jsonl \
  --sha256 a29f5accb9f617c473d1ab9415b00cd1eecb2fdb806708091f24c72e3d3fc6da \
  --output runs/retrospective-paired-audit.json
python -m pytest -q tests/test_evaluation_pairing_contract.py \
  tests/test_evaluation_counts_contract.py tests/test_public_paired_audit.py
python scripts/replay_phonetic_decisions.py
```

The output path must not already exist. The eight checked-in historical fixtures
remain lightweight regression coverage; the full 96-row replay is a separate artifact
reproduction, not simulated model inference. Missing input files are an error, not
an invitation to regenerate convenient substitutes.

## Primary methods and a new falsifiable experiment

Koehn (2004), *Statistical Significance Tests for Machine Translation Evaluation*,
https://aclanthology.org/W04-3250/, is the original paired-resampling reference for
language-system comparison. That setting is machine translation; a Japanese speech
study must separately specify its metric and dependence unit.

Dror et al. (2018), *The Hitchhiker's Guide to Testing Statistical Significance in
Natural Language Processing*, https://aclanthology.org/P18-1128/, emphasizes choosing
statistical tests to match the task, setup and evaluation measure. Here the explicit
choice between utterance mean and ratio of corpus counts is part of that specification,
not an interchangeable display preference.

Hume/Hugging Face (2026-08-21), *Measuring benchmark optimization in speech recognition*,
https://huggingface.co/blog/asr-benchmark-optimization, introduces reference-disagreement,
masked-entity and orthographic-switching probes on public ASR benchmarks. In particular,
target words absent from modified audio should not automatically be rewarded for
matching the original transcript. Its reported experiments concern other systems and
English benchmark data; this repository has not reproduced those findings in Japanese.

Issue #53 records an **unexecuted Japanese adaptation**: original audio, target-word
silence/noise, matched non-target edits; no/proper/unrelated/incorrect context; fixed
source boundaries; human-checked annotations; source-recording-level paired groups.
Measure unsupported recovery, abstention/coverage and damage outside the target,
not merely CER against the unchanged original transcript. A recovered word by itself
does not prove memorization; linguistic inference and other explanations need controls.
The first PR should implement only transformation/manifest/negative tests, then a
frozen development probe, then an independently held-out evaluation.

## Next-agent work and completion boundaries

Read the live Issues, not only this dated snapshot. #19/#24/#25 were already completed;
#26 has an active rights/lineage branch. The focus here is #52, not completion of #29.

1. #26/#28: bind expected IDs, grouping and derivation history to the validated rights
   manifest and locked runtime. Include prior 72 phonetic-study recordings and 40
   training-pilot recordings in exposure lineage; do not overwrite old reproduction
   manifests. Unknown speakers remain unknown, not fabricated independent IDs.
2. #29: add gold versus G2P-proxy label metrics, silence policy, semantic-critical slices,
   coverage/risk and false-correction uncertainty; predeclare the primary estimand and
   cluster unit. Add explicit permutation/missing-arm tests. Multiple trials/seeds need
   a preregistered analysis rather than cherry-picking the best test interval.
3. #35/#36: finish the separately tracked resume contract, compare new heads against
   the original TRAINED phone head, and run phone-only/mora-only/joint ablations with
   fixed seeds. Save actual updates, frozen-tensor identities, reload results and all
   failures. A lower weak-label loss from random initialization is not an ASR gain.
4. #37: count informative real candidate pairs and oracle headroom before more LoRA
   updates. Compare no-LoRA/simple-ranker/LoRA on the same candidates and compute.
   Keep exact expected evaluation IDs even for failed or abstaining predictions;
   record false corrections and order/ID/context controls. Do not run an easier
   intersection after failures. Existing weights are unchanged by this increment.
5. #38/#40/#53: use real ordered recordings to assess document context. Keep intervention
   variants together by original recording. Publish negative results, complete records,
   protocol and code hashes. Engineering-complete, experiment-complete and
   promotion-approved remain separate outcomes.

There is no newly trained checkpoint, perpetual agent, enabled Discussion board,
validated end-to-end phonetic deployment, or world-best claim in this change.
