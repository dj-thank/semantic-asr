# Context × phonetic factorial experiment

## Purpose

Semantic ASR now has two independent ways to challenge a locally plausible first-pass transcript:

1. complete-path and document context;
2. source-audio phone and mora posteriors.

Testing each one separately cannot show whether they complement one another or merely repeat the
same preference. This protocol evaluates both factors over **one frozen candidate/evidence pool**.
It also includes shuffled-context controls so general language fluency is not mistaken for useful
conversation context.

## Core invariant

```text
context preference != acoustic proof
candidate-derived mora_shadow != audio-derived mora
all factorial arms reuse one frozen candidate pool
```

No context arm may generate a new surface. Candidate generation happens once from the source audio,
frozen pronunciation lexicon, first-pass candidates, runtime profile, and held-out phone/mora
utility artifact. Context only ranks those frozen surfaces.

## Factor matrix

The complete registered design is:

```text
                         context
phonetic          none   ordered   shuffled
------------------------------------------------
first-pass          ×       ×          ×
phone               ×       ×          ×
mora                ×       ×          ×
phone + mora        ×       ×          ×
```

The default required identities are:

```text
baseline          first-pass:none
target            phone+mora:ordered
shuffled control  phone+mora:shuffled
```

A protocol may add discrete-unit arms after every candidate in every frozen pool has a
source-audio-bound discrete-unit utility. Missing channel evidence makes a candidate ineligible; it
is never treated as a zero score.

## Two-phase execution

### Phase A — reference-free preparation

`prepare_context_phonetic_experiment()` receives the registered manifest but passes only
`PlanningCaseView` to the phonetic planner. That view contains:

- exact audio path, crop, and SHA-256;
- first-pass local surfaces and posterior mass;
- exogenous frozen pronunciation lexicon;
- rights and provenance digest.

It does not contain the reference.

The dual CTC runtime runs once per case. Every lexicon surface is scored once for phone and mora.
The context scorer then runs once for ordered context and once for shuffled context. All twelve arms
reuse those exact values.

### Phase B — reference-opened evaluation

After pool, context-score, shuffle-assignment, protocol, and scorer digests are frozen, references
are opened. The evaluator computes proposed and effective decisions for every registered arm.
Low-margin proposals are `provisional`; unless registered otherwise, the effective output falls
back to the original first-pass selection.

## Ordered and shuffled context

Each case contains a caller-owned `FrozenContextSnapshot` with left context, right context, topic,
entity IDs, source case ID, and revision. The snapshot must be fixed before reference evaluation.

Shuffled context uses a deterministic derangement under a registered seed. A donor may be rejected
when it shares the receiver's speaker, session, or source recording. If no derangement satisfies the
registered exclusions, preparation fails; it does not silently relax the control.

The scorer identity is identical for ordered and shuffled conditions. Every score is bound to:

- candidate ID and candidate-text SHA-256;
- context-snapshot digest;
- scorer source;
- immutable scorer-profile digest.

## Selection

For candidate `c` under arm `a`:

```text
score(c, a)
  = Σ phonetic_weight[channel] × frozen_utility(c, channel)
  + context_weight × frozen_context_score(c, condition)
  + registered retention bonus when c is the selected first-pass surface
```

An outside-first-pass candidate is never eligible under an arm with no independent phone, mora, or
discrete-unit channel. This keeps context-only evaluation from turning language preference into
acoustic evidence.

## Metrics

Per case and arm:

- first-pass, proposed, and effective edit counts;
- proposed and effective exact match;
- frozen-pool oracle coverage;
- reference outside the first-pass candidate set;
- successful outside-first-pass recovery;
- false correction of a correct first-pass surface;
- successful correction of an incorrect first-pass surface;
- introduced and corrected error characters;
- critical-case identity;
- acceptance, change, margin, and latency;
- ordered or shuffled context donor identity.

False-correction rate is conditioned on first-pass-correct cases. It is not divided by all test
cases, because that would hide destructive behavior on the only cases that could be harmed.

Reports contain text hashes by default, not raw candidate, reference, or context text.

## Factorial contrasts

The report produces:

```text
combined-vs-baseline
  phone+mora:ordered - first-pass:none

ordered-vs-shuffled
  phone+mora:ordered - phone+mora:shuffled

phonetic-main-effect
  phone+mora:none - first-pass:none

context-main-effect
  first-pass:ordered - first-pass:none

context-by-phonetic-error-interaction
  [CER(phone+mora, ordered) - CER(phone+mora, none)]
  - [CER(first-pass, ordered) - CER(first-pass, none)]
```

Negative character-error deltas favor the target. A negative interaction means ordered context
reduces error more when independent phonetic evidence is active than it does on the first-pass-only
candidate set.

## Grouped paired bootstrap

All uncertainty estimates resample registered speaker, session, or source groups. Multiple
utterances from one speaker may remain in the point estimate but are never treated as independent
bootstrap units.

The same sampled groups are used across every arm in a contrast. The random seed and number of
resamples are part of the protocol digest.

## Preregistration

`ContextPhoneticExperimentRegistration` binds:

- the full manifest digest, including reference hashes;
- the reference-free planning-manifest digest;
- factorial and lower phonetic protocol digests;
- phonetic planner profile digest;
- context scorer source and profile digest;
- promotion-policy digest.

Any change to a reference, context, candidate pool, runtime, utility profile, shuffle seed, arm,
statistical grouping, or threshold is rejected before audio inference or context scoring begins.

## Promotion

Promotion is conjunctive. The ordered phone+mora target must pass every registered check:

- exact-accuracy gain over first pass;
- paired-bootstrap upper bound for combined vs baseline CER delta;
- ordered vs shuffled context specificity;
- context × phonetic interaction bound;
- candidate-pool oracle coverage;
- outside-first-pass recovery;
- conditional false-correction rate and delta;
- introduced error-character limit;
- critical-case accuracy;
- accepted coverage;
- total generation + context-scoring + selection latency.

A failure in any one dimension blocks promotion.

## Synthetic contract fixture

The permanent synthetic fixture intentionally contains:

1. a correct surface outside first-pass N-best that phone/mora recovers;
2. an already-correct first-pass surface that must be retained;
3. a phone/mora error that ordered context must repair;
4. a second outside-N-best recovery;
5. shuffled contexts that prefer the wrong surface.

The expected target is 4/4 exact, while first-pass is 2/4 and phone+mora without context contains
one false correction. This validates the measurement logic; it is not a Japanese model-quality
claim.

## Real-data boundary

A real experiment additionally requires:

- the four-way train/validation/calibration/test manifest;
- immutable dual CTC and utility artifacts;
- frozen pronunciation lexicons constructed without test-reference access;
- frozen first-pass local candidate posteriors;
- context snapshots captured before reference opening;
- enough independent groups for confidence intervals;
- negative, distractor, correct-first-pass, homophone, number, negation, entity, filler, and repair
  slices;
- target-device latency and memory measurements.

Nothing in this package changes a measured v0.2 runtime default.
