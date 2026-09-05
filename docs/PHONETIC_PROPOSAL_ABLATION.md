# Frozen phonetic proposal ablation

## Question

A phone or mora head is useful only if it improves the final transcript decision. Lower standalone
phone error rate or mora error rate is insufficient. This experiment asks:

1. can source-audio-only phone/mora evidence recover the correct local surface when it is absent
   from the first-pass ASR candidates;
2. how often does the same evidence replace an already-correct first-pass surface with a wrong one;
3. how much of the candidate oracle is provided by a frozen, exogenous pronunciation lexicon;
4. how much latency is spent generating the shared evidence pool and selecting each arm.

## Reference separation

Each case has two identities:

```text
planning_digest = audio + crop + first-pass candidates + exogenous lexicon + rights/provenance
case_digest     = planning_digest + frozen reference digest
```

`PlanningCaseView` does not contain the reference or reference digest. The acoustic runtime runs
once against this reference-free view. The returned `FrozenPhoneticCandidatePool` is then reused by
every ablation arm. References are opened only by `evaluate_prepared_phonetic_ablation()` after the
pool and its digest are fixed.

Changing a reference, protocol, runtime profile, utility artifact, lexicon, candidate posterior, or
crop after preparation invalidates the run.

## Shared candidate pool

The planner requires:

- exact source-audio SHA-256 and crop bounds;
- one frozen dual CTC runtime profile;
- one runtime-bound phone/mora utility artifact;
- one frozen pronunciation lexicon prepared before evaluation;
- the complete first-pass local candidate surfaces and posterior mass.

Phone and mora posteriors are inferred once. Every lexicon surface is scored once, and the resulting
candidate pool records:

- phone utility;
- audio-derived mora utility;
- optional future discrete-unit utility;
- first-pass membership and posterior;
- whether the surface was the selected first-pass surface;
- pronunciation and proposal digests;
- generation latency.

All arms use the exact same pool. An arm may mask channels or forbid candidates outside the
first-pass set, but it may not regenerate candidates.

## Initial arms

A minimal registered protocol should include:

```text
first-pass       first-pass posterior only; outside candidates forbidden
phone            phone utility only
mora             audio-derived mora utility only
phone+mora       both independent CTC utilities
```

A later protocol may add `discrete_unit`, but only after every pool candidate has a frozen,
source-audio-bound utility for that channel. Missing channel evidence makes a candidate ineligible;
it is never silently interpreted as zero.

## Selection and abstention

Each arm scores eligible candidates with its frozen non-negative channel weights. The selected
first-pass surface may receive a registered retention bonus. A decision below `minimum_margin` is
`provisional`; unless `apply_provisional=True`, its effective output falls back to the original
first-pass selection.

The report therefore distinguishes:

```text
proposed candidate  what the arm preferred
effective candidate what the declared application policy would emit
```

This prevents a low-margin exploratory proposal from being counted as a deployed correction.

## Metrics

Per case and arm:

- first-pass, proposed, and effective edit counts;
- proposed and effective exact match;
- candidate-pool oracle membership;
- reference outside the first-pass candidate set;
- successful recovery outside first pass;
- false correction of a correct first-pass surface;
- successful correction of an incorrect first-pass surface;
- introduced and corrected error characters;
- critical-token case identity;
- acceptance, change, margin, generation latency, and selection latency.

Aggregate false-correction rate is conditioned on cases where the first-pass surface was correct.
Paired confidence intervals resample frozen speaker/session/source groups rather than treating
multiple utterances from one speaker as independent observations.

Reports contain reference and candidate text hashes by default, not raw text.

## Promotion

`evaluate_phonetic_promotion()` is conjunctive. The target arm must satisfy every registered
threshold:

- exact-accuracy gain over first pass;
- paired bootstrap upper bound on character-error delta;
- candidate-pool oracle coverage;
- recovery rate when the reference is outside first pass;
- conditional false-correction rate;
- introduced error characters;
- critical-case exact accuracy;
- accepted coverage;
- mean generation latency.

Failure of any check blocks promotion. Thresholds must be registered before test references are
opened.

## Statistical boundary

Cases from the same speaker, session, or source recording may be correlated. The protocol therefore
names the bootstrap grouping field and samples groups with replacement. The point estimate remains
computed over all cases; only uncertainty estimation is grouped.

This harness does not establish independence merely by hashing rows. Dataset construction must
still prevent train/validation/calibration/test leakage and must keep lexicon construction
exogenous to the evaluation reference.

## Claim boundary

Passing synthetic tests validates contracts, not ASR quality. A real claim requires:

- a locked speaker-disjoint Japanese test manifest;
- immutable runtime and utility artifacts;
- frozen candidate posteriors and lexicons;
- negative and distractor cases, including correct first-pass examples;
- published failure cases;
- paired grouped confidence intervals;
- end-to-end document deliberation evaluation after local proposal evaluation passes.
