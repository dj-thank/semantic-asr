# Grouped realtime refine scheduler (Issue #66)

Status: **main-based engineering slice in PR #68, after PR #65**. This document specifies grouping and evidence binding only. It does not promote a second-pass model or claim a Semantic ASR accuracy improvement.

## Motivation

Hayamimi's realtime pipeline keeps the fast final visible and later re-decodes a larger utterance group once enough silence arrives. Its desktop refiner uses a 2 second group gap and a 25 second maximum group span, forces the last group at shutdown, and keeps expensive refine work off the immediate-final hot path.

Semantic ASR needs the same useful latency/quality separation, but with a stronger provenance rule: a refine result may be a child of first-pass evidence; it may not silently replace that evidence.

## Why grouped audio is not `b"".join(final.pcm)`

The realtime first pass can retain pre-roll and endpoint silence in each final buffer. Two neighboring final buffers can therefore overlap or contain duplicated context. Concatenating them would manufacture an audio sequence that never existed on the input timeline.

`BoundedPcmHistory` instead stores the raw PCM stream once with absolute sample positions. A grouped request reads exactly `[first_parent.start_sample, last_parent.end_sample)` from that continuous timeline. Silence and context between finals are therefore represented exactly once.

## Scheduler defaults

| control | default | purpose |
|---|---:|---|
| sample rate | 16,000 Hz | current realtime Reazon contract |
| idle gap | 2,000 ms | close group after a real VAD-confirmed pause |
| max group | 25,000 ms | prevent unbounded context/refine work |
| minimum group | 500 ms | skip second-pass work that is too short to justify |
| PCM history | 30,000 ms | covers 25 s group + 2 s idle detection with margin |
| pending finals | 64 | finite metadata bound |

The 2 s / 25 s values are Hayamimi-inspired starting points, not measured Semantic ASR optima. They stay behind an engineering boundary until #39/#29 paired evaluation proves a better setting or validates these defaults.

## Evidence model

Every parent final contributes:

- `utterance_id`
- immutable `final_digest`
- first-pass `audio_sha256`
- absolute start/end samples
- observed text for downstream display/debug context

Every `GroupedRefineRequest` contributes:

- ordered parent final proofs
- group/session identity and monotonic sequence
- exact continuous PCM bytes
- absolute group start/end samples
- `group_audio_sha256`
- close trigger (`idle-gap`, `max-duration`, `parent-limit`, `eof`, `reset`, `manual`)
- whether the group is eligible for decode, with explicit skip reason otherwise
- canonical request evidence digest

The request evidence digest binds the parent final digests, not a second unaudited copy of parent text. The parent final digest is the authoritative link back to the original first-pass event.

## VAD activity is part of the idle contract

Elapsed samples alone are not evidence of silence. A new utterance may remain active for longer than the 2 second refine gap, and closing the previous group merely because two seconds elapsed would make grouped refinement race the live speech path.

`GroupedRefineScheduler.feed_pcm16()` therefore accepts the caller's `speech` decision for each raw PCM chunk. Live integration must pass the **same serialized VAD decision** already used by the realtime Reazon session:

- `speech=True` resets accumulated idle silence and never closes the group;
- only consecutive `speech=False` PCM counts toward the 2 second idle threshold;
- a new first-pass final resets the idle accumulator;
- malformed/non-boolean activity fails before PCM history advances.

The default non-speech value exists for deterministic/offline scheduler fixtures. Production/live callers must pass the VAD decision explicitly; wall-clock or sample distance must not be substituted for VAD-confirmed silence.

## Closing rules

1. Close with `idle-gap` only after consecutive VAD-confirmed non-speech PCM reaches 2 seconds. Any speech chunk resets that counter.
2. If adding the next final would make the group exceed 25 s, close the existing group first with `max-duration`, then start the next group with the new final.
3. If a group lands exactly on the 25 s bound, close immediately.
4. If 64 parents are already pending, close before accepting another parent.
5. EOF/manual close never fabricates additional silence.
6. A group shorter than 500 ms is emitted as an explicit `group-too-short` skip instead of disappearing.
7. Reset requires the caller to choose `flush_pending=True` or `False`. A non-flushing reset emits a `GroupedRefineDiscard` receipt binding the discarded parent final digests.

## Failure policy

Fail closed when:

- PCM history is discontinuous;
- a supplied activity flag is not boolean;
- requested parent audio has already fallen outside retained continuous history;
- parent final order moves backward or its end does not strictly advance;
- SHA-256 fields are malformed;
- grouped PCM length/digest and sample bounds disagree;
- the configured PCM history cannot cover `max_group + idle_gap`;
- numeric bounds are invalid.

No reference transcript, gold text, candidate-derived dictionary, network lookup, or model download is involved in this module.

## Next integration slice

Once this scheduler is green on the exact PR #68 source:

1. bind PR #65 `final` events to `RefineParentFinal` without changing their immutable evidence;
2. feed each raw 16 kHz PCM chunk into `BoundedPcmHistory` once and pass the same serialized VAD `speech` decision used by `RealtimeReazonSession`;
3. create a single FIFO refine worker so grouped re-decodes cannot reorder output or stall fast finals;
4. make the worker consume `GroupedRefineRequest`, not mutable scheduler state;
5. reuse a warm Semantic ASR transcriber / pinned Reazon adapter rather than loading a new model per group;
6. emit a new refine revision binding both the grouped-request digest and all parent final digests;
7. force the final pending group at EOF and cleanly drain/close the worker;
8. prove queue/history/model lifecycle under reset/close tests;
9. profile speech-end -> fast-final independently from group-close -> refine latency;
10. run paired fixed-audio CER / semantic-critical / improve-tie-harm evaluation before any default-profile change.

## Promotion boundary

Engineering completion of the scheduler says only that grouping and evidence binding are deterministic and bounded. It does **not** establish that grouped re-decoding improves Japanese recognition. Any such claim requires fixed real audio, exact model/runtime identities, paired measurements, and the existing Semantic ASR quality/safety gates.
