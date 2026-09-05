# Frozen ContextCatalog

`ContextCatalog` supplies proper nouns and domain terms to the decoder without allowing a
language model to rewrite the observed transcript. It is deliberately **exogenous**: freeze
the catalog before processing or evaluating audio. Never construct it from the evaluation
reference transcript.

## Safety contract

- A catalog has a required schema version, name, revision, deterministic digest, and unique
  entry IDs and canonical phrases.
- Retrieval uses caller-owned `context_query` text, aliases, optional readings, and required
  tags. An empty query or no match records an abstention and injects no catalog phrase.
- Only selected canonical phrases become Whisper hotwords. The catalog never rewrites a
  candidate after decoding, but hotwords can change candidate generation; measure distractor
  false positives before promotion.
- The transcript provenance stores the catalog digest, query hash, selected entry IDs, phrase
  hashes, scores, and reasons. It does not retain the raw query or selected phrase strings.
- Use opaque entry IDs when identifiers themselves are sensitive. Context queries are capped
  at 1,024 characters to bound deterministic retrieval cost.
- The catalog receipt is included in decode cache identity, so evidence from another catalog
  revision or query cannot be replayed silently.

## JSON format

```json
{
  "schemaVersion": 1,
  "name": "project-meeting",
  "revision": "agenda-2026-09-04",
  "entries": [
    {
      "id": "person:moriwaki",
      "phrase": "森脇翔太",
      "aliases": ["森脇さん"],
      "reading": "モリワキショウタ",
      "tags": ["person", "project-a"],
      "priority": 2.0
    }
  ]
}
```

`priority` only breaks equal-score ties; it cannot turn an unrelated entry into a match.
Multiple `context_tags` are an AND filter.

## Python

```python
from semantic_asr import ContextCatalog, transcribe

catalog = ContextCatalog.from_json("examples/context_catalog.example.json")
result = transcribe(
    "meeting.wav",
    profile="cpu-ja-v1",
    catalog=catalog,
    context_query="森脇さんとSemantic ASRの進捗確認",
    context_tags=("person",),
)
print(result.observed_text)
print(result.provenance["contextCatalog"])
```

A warm transcriber must be loaded with the same profile:

```python
from semantic_asr import load_transcriber, transcribe

warm = load_transcriber("cpu-ja-quality-v1")
result = transcribe(
    "meeting.wav",
    profile="cpu-ja-quality-v1",
    transcriber=warm,
    catalog=catalog,
    context_query="Semantic ASR",
)
```

## CLI

```bash
semantic-asr run meeting.wav \
  --catalog examples/context_catalog.example.json \
  --context-query "森脇さんとSemantic ASRの進捗確認" \
  --context-tag person \
  --output-dir transcripts
```

Omitting `--context-query` intentionally produces `empty-query` abstention rather than
injecting the whole catalog.

## Promotion gate

This module establishes the interface, provenance, no-bias path, and deterministic retrieval.
It does not by itself claim an entity-error-rate improvement. Before enabling a catalog by
default, compare frozen relevant-context, distractor-only, homophone, catalog-missing, and
no-context arms. Report entity recall/error, non-entity CER, distractor false positives, and
abstention separately on speaker/domain-disjoint data.
