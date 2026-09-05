# Deterministic Japanese mora and phone labels

## Boundary

This component converts an **explicit reading** into a closed phonetic label system. It does not
infer a reading from Kanji. A source row such as:

```json
{"transcript": "学校へ行く", "reading": "がっこうへいく"}
```

is accepted because the reading is supplied independently. A row with only `学校へ行く` is
rejected. This avoids silently treating a morphological analyzer's mutable dictionary output as
acoustic ground truth.

## Label conventions

The v1 profile uses:

```text
moraic nasal  ン -> N
gemination    ッ -> q
long vowel    ー -> :
```

Palatalized mora and common foreign katakana are represented explicitly:

```text
キョ -> ky o
シャ -> sh a
チャ -> ch a
ファ -> f a
ティ -> t i
ヴュ -> vy u
```

The onset cluster is one label in this inventory (`ky`, `sh`, `ch`, and so on). This is a frozen
engineering convention, not a claim that each label is a universal linguistic phoneme. Replacing
it with decomposed phones, X-SAMPA, Julius labels, or an articulatory inventory creates a new
profile and new digest.

## Fail-closed normalization

The labeler:

- applies Unicode NFKC;
- converts hiragana to katakana;
- removes only a declared punctuation set;
- combines supported small kana with the preceding base mora;
- rejects Kanji, iteration marks, unsupported symbols, and unsupported combinations;
- rejects an initial long-vowel mark;
- rejects a terminal or vowel-leading geminate marker.

No unknown-phone fallback is enabled in the default profile. Unsupported material must be reviewed
or added under a new mapping revision.

## Identity

`JapanesePhoneticLabelProfile.digest` includes:

- profile and schema revisions;
- special phone conventions;
- punctuation policy and exact punctuation set;
- the complete sorted mora-to-phone mapping digest.

The generated phone and mora inventories include that mapping digest in their revisions. Training,
utility calibration, pronunciation lexicons, and evaluation therefore cannot silently use a
different kana-to-phone table.

## Pronunciation lexicons

`build_japanese_pronunciation_lexicon()` converts caller-owned `(surface, reading)` entries into the
existing frozen P2G lexicon contract. It is appropriate for:

- first-pass candidate surfaces with verified readings;
- frozen context-catalog names with caller-supplied readings;
- domain terms prepared before evaluation;
- controlled homophone sets.

It must not read the evaluation reference after transcription. Two homographs or duplicate surface
strings are rejected because they would make the local proposal identity ambiguous.

## Manifest materialization

```bash
python scripts/prepare_japanese_phonetic_manifest.py \
  /data/source-readings.jsonl \
  --output-dir /data/materialized-ja-phonetic-r1 \
  --name ja-phonetic-corpus \
  --revision r1 \
  --allow-derived-phonetic-labels
```

The input contains absolute PCM16 WAV paths, explicit readings, rights, license, speaker, session,
source, and four-way split identities. The materializer writes:

```text
manifest.jsonl
manifest.jsonl.metadata.json
materialization.json
phone_inventory.json
mora_inventory.json
```

The training manifest does not copy raw transcript text. When a transcript is supplied, only its
SHA-256 is retained. `materialization.json` binds the source input SHA-256, label profile, mapping,
inventory digests, row digests, and output file hashes.

Output is create-only, atomic, and must be outside the repository checkout. Train, validation,
calibration, and test splits are checked for speaker, session, and source-recording leakage before
publication.

## Accuracy boundary

Rule-based labels are supervision conventions. They do not prove that the audio contains every
canonical phone exactly as written; spontaneous reductions, devoicing, dialect, repairs, and
pronunciation variants remain real acoustic phenomena. Evaluation should include:

- label-audit samples by native speakers;
- variant-pronunciation lexicons;
- phone and mora error rates;
- error slices for long vowels, gemination, moraic nasal, palatalization, and loanwords;
- end-to-end proposal recovery and false-correction rates.

A lower CTC loss against these labels is not by itself an ASR improvement claim.
