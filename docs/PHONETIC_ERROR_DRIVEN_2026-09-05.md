# Audio-first mora/phone improvement: two measured public-speech cycles

## Result and claim boundary

This work fixes executable mora/CTC defects and performs **two actual model-inference / error-analysis cycles**, not only synthetic demonstrations. It does not establish a zero-error recognizer or a deployable quality improvement.

The public FLEURS Japanese pilot contains **72 distinct recordings and 96 utterance passes**: wave 1 has 24 development + 24 test clips; wave 2 reuses those 24 development clips and holds out 24 different test clips. Model inputs never include the reference transcription. Models, dataset, source revision, runtime, candidate outputs, frame-level posteriors and score provenance were recorded. Text-ID, audio and reference-surface exclusions are not a proof of speaker-disjointness or absence of pretraining contamination.

| Experiment | Split | Reference characters | Baseline errors | Selected errors | Improved / harmed utterances |
|---|---|---:|---:|---:|---:|
| Wave 1, beam-5 + phone/context guard | Development 24 | 1,157 | 49 | 36 | 4 / 0 |
| Wave 1, frozen policy | Held-out 24 | 1,183 | 63 | 60 | 3 / 1 |
| Wave 2, retain beam-5 + add beam-12 paths | Development 24 | 1,157 | 45 | 34 | 3 / 0 |
| Wave 2, separately frozen policy | Fresh held-out 24 | 1,140 | 55 | 55 | 0 / 0 |

Wave 1 held-out CER is **5.3254% -> 5.0719%**. Paired utterance bootstrap (2,000 repetitions, seed 17) gives a 95% interval for CER change of **-0.9315 to +0.3500 percentage points**, which includes no improvement. One already-wrong utterance became worse under the text metric; no previously exact test utterance became wrong. Wave 2 held-out CER remains **4.8246%**; its only changed output has equal character error count. Neither policy is promoted to default inference.

The two test cohorts differ. Do not subtract their error counts to claim wave-2 improvement. Even the repeated development baseline changed from 49 to 45 errors across inference runs: artifacts reproduce the exact measured outputs, but CPU/sampling execution is not claimed bit-identical. Each improvement claim must use its own paired baseline.

## What was actually implemented

`mora_phonology.py` adds an explicit Open-JTalk-style kana/phone inventory, including yoon and foreign combinations. `japanese.py` now requires a legal **adjacent** combination: `き、ゃ` and `き ゃ` no longer collapse into `キャ`; malformed `カィ` and `キャャ` are not silently accepted as one normal mora. Timed punctuation/unknown rows break composition. No timing-only segmentation accuracy is claimed.

`phones_to_moras()` preserves the original phone sequence exactly while giving a mora-level view. It retains voiced/devoiced `i/I`, `u/U`, gemination `cl`, moraic nasal `N`, pauses and unresolved phones. Homophonous kana remain alternatives; a repeated vowel is only marked as possible lengthening. It does not pretend `/j i/` proves ジ rather than ヂ, or that `/o o/` proves a particular kanji/long-vowel spelling. `kana_to_phone_moras()` is a reading proposal, never independent acoustic evidence. It does not blindly rewrite every エイ/オウ into /ee/ or /oo/.

`training.py` now rejects invalid CTC labels, blank targets, non-integer lengths and targets, out-of-vocabulary IDs and insufficient frames for repeated labels. The previous `[1,1]` target over two frames could reach `zero_infinity=True` and silently contribute zero loss. It now fails explicitly. Non-finite supervision cannot masquerade as successful training. Existing valid forward/backward tests remain green.

The existing pure-Python `phonetic_evidence.py` scorer now keeps zero-probability paths impossible rather than introducing artificial epsilon support. This is a research API behavior change: `probability_floor` must be zero. Immutable sequence storage and integer frame-time validation were tightened.

`phonetic_refinement.py` provides a standalone, reference-blind frozen selector. It combines candidate-specific acoustic likelihood and full-candidate LM preference with an acoustic-regression guard, an edit-size bound, stable baseline retention and explicit provisional decisions. Scores must share a model/profile, recording and posterior identity. Same-phone surface changes are marked context-resolved, not acoustically proven spelling. No text generation or first-pass evidence mutation is performed.

## Actual models and reproducibility

All model inference ran on GitHub-hosted CPU runners, not local mock adapters:

- Dataset: `google/fleurs`, `ja_jp`, revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`; reference text CC-BY-4.0. Attribution: FLEURS, Google, Conneau et al. (2022). https://huggingface.co/datasets/google/fleurs
- Existing Whisper: large-v3-turbo, revision `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf`, `cpu-ja-v1`.
- Independent phoneme model: `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`, revision `f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99`, Apache-2.0. Its model card documents training-time pronunciation substitutions and exclusions; it is not a gold phonetic annotator. https://huggingface.co/prj-beatrice/japanese-hubert-base-phoneme-ctc-v4
- Full-candidate language scoring: `Qwen/Qwen3-0.6B-Base`, revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Apache-2.0. https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Acoustic/LM runner: Python 3.12.14, torch 2.14.0+cpu, transformers 4.57.6, faster-whisper 1.2.1, CTranslate2 4.8.2, pyopenjtalk-plus 0.4.1.post9, NumPy 2.5.2, pyarrow 25.0.1. Resolved dependencies are in the inference artifacts.

The LM uses teacher-forced whole-candidate likelihood with a fixed neutral prefix, not generated corrections. These isolated FLEURS clips have no reliable shared document context, so unrelated adjacent dataset rows are not used as surrounding conversation. The repository's earlier left/right document-context deliberation interface remains available, but **long-meeting context quality was not measured here**.

Wave 1 acoustic run: https://github.com/dj-thank/semantic-asr/actions/runs/33946834870

Wave 1 context run: https://github.com/dj-thank/semantic-asr/actions/runs/33947448420

Wave 2 expanded acoustic/context run: https://github.com/dj-thank/semantic-asr/actions/runs/33947793936

All three inference workflows completed successfully. Artifact retention was seven days; preserve the downloaded study bundle for long-term reproduction. No audio or model weights were uploaded as evidence artifacts.

For a new bounded collection in a separate environment (the `rerank` extra targets a different Transformers major version):

```bash
python -m pip install -e '.[phonetic]' pyarrow soundfile
python scripts/collect_phonetic_public_probe.py --output runs/phone-probe --per-split 24 \
  --dataset-revision 70bb2e84b976b7e960aa89f1c648e09c59f894dd \
  --phone-revision f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99
python scripts/score_public_probe_context.py --probe runs/phone-probe --output runs/context-probe \
  --revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd
python scripts/collect_expanded_phonetic_probe.py --prior runs/phone-probe \
  --output runs/expanded-probe --per-split 24
```

Collection scripts are executable and published. The fresh-cohort collector propagates prior exclusions for later manual rounds. There is no perpetual background agent, automatic model promotion or automatic push loop.

## Frozen selection and error ledger

Each round fitted a finite grid on development only, then saved a policy hash before opening held-out references. The fitted object is a **fusion policy**, not newly fine-tuned HuBERT, Whisper or Qwen weights. A no-change policy was included. Phone likelihood uses mean-frame normalization; LM preference uses mean-token normalization.

Wave 1 frozen policy file SHA-256: `45e3b28910dcdbf50388edfef67bc0ebb775f8bb5e2fd9afb3583fdb3c8248ec`.

Wave 2 frozen policy file SHA-256: `a2e85cddceba25d64915c359c49758e73163dbcbc9e40a4b6f49e36fa5ee6e81`.

The local exploratory grid also tested pause-tolerant and devoicing-marginalized scores. Both winning policies use strict CTC. All **252 strict candidate scores** across the 96 passes were independently checked against `torch.nn.functional.ctc_loss` with float64 and no zeroing: maximum absolute mean-frame difference `1.1102230246251565e-16`. This checks the arithmetic, not the recognizer's correctness.

`research/phonetic-20260905/errors.json` stores **all 59 baseline-or-final erroneous rows** from both development/test rounds, as lossless edits against attributed public reference text. The count includes repeated development observations, not 59 unique recordings. Tests reconstruct the exact baseline/final strings and their error counts. The complete numeric/text study bundle retains original posteriors, phone strings, candidates, every development trial and full per-error analysis.

Eight regression fixtures cover every changed held-out decision plus two unchanged controls. They replay the stored scalar evidence with the published selector, not the unpublished dense runtime:

```bash
python scripts/replay_phonetic_decisions.py
python -m pytest -q tests/test_public_phonetic_records.py
```

These selected regression fixtures are not a substitute for the complete evaluation; the full error ledger and measured denominators remain authoritative.

## What the failures say about the architecture

1. **Candidate coverage is a real bottleneck.** Wave 1's 16 residual erroneous test rows all lack an exact reference surface in the original candidate pool. Ranking cannot select a sentence absent from its choices. In wave 2, the same-run test candidate oracle improves from 49 errors (retained beam-5 pool) to 44 (expanded pool). Average pool size grows to 3.54. This is an oracle diagnostic, not achieved recognition quality.
2. **A better pool does not guarantee a better selector.** Wave 2 retains 55 actual errors, while seven cases have a better available candidate. A rigid guard can preserve a mistaken pronunciation because the phone model or G2P reading is wrong. Simply trusting the phone argmax is worse on development: 49 errors versus baseline 45, with four previously exact utterances spoiled.
3. **Homophones and spelling variants require different evaluation.** Wave 1 includes `50m -> 50マイル` (improved but not entirely corrected), `ガサブランカ -> カサブランカ`, and `失礼の人 -> 失礼な人`. The harmed row includes `つながっています -> 繋がっています`; text CER can worsen for a valid spelling variant, so it is not automatically a semantic error. Do not hide the metric regression, but do not call every character edit a changed meaning.
4. **G2P is fallible supervision.** Reference pronunciations here are text-derived proxies. For example, the frontend's reading of 微表情 contains an implausible prefix in the recorded phone sequence. An audio-model/reference-G2P disagreement cannot identify which side is wrong without phonetic annotation or listening. The public phone model itself also inherits weak-label choices.
5. **Audio units are not separate independent witnesses.** A mora sequence grouped from the same phoneme posterior is another representation of that evidence, not a second vote. Future phone/mora heads need explicit shared-encoder correlation handling, calibrated local acoustic evidence and pronunciation alternatives rather than duplicated confidence.

The next defensible research increment is pronunciation alternatives plus independently checked local evidence, document-context evaluation on actual long recordings, and locked speaker-disjoint training/calibration/test manifests. Those are outstanding research requirements, not work claimed completed by this commit.

## Publication and verification boundary

The connector rejected the new dense `phone_ctc.py` write. Its dependent dense-posterior runtime, full local fitting/evaluation driver and runtime smoke script are therefore **not included or advertised as runnable from this branch**. No alternate upload route was used. The existing scorer fixes, standalone mora/phone mapping, guarded score selector, public model collectors and stored-result regression replay are independent and published.

The published-source subset passes **473 tests, with three existing strict xfails** for legacy `DecodeRequest` (issue #19). Ruff format/lint and compileall pass. The larger local experimental stack's 490-pass count is not attributed to the published source. CI includes actual CPU backward tests and Ubuntu/Windows integrity checks. No newly trained acoustic/LM checkpoint, streaming finalization, accent accuracy, zero-error guarantee or general Japanese CER improvement is claimed.
