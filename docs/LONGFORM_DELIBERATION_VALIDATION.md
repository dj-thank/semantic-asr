# Long-form deliberation validation ledger

## 2026-09-05 — implementation audit

This ledger records software-contract validation only. It is not an accuracy claim and does not
promote long-form deliberation into a default runtime profile.

Validated invariants:

- every first-pass candidate is reconstructed exactly through one `SourcePath`;
- deletions use explicit epsilon arcs rather than malformed empty text;
- projected whole-hypothesis evidence has a finite factor budget instead of being repeated per
  character or contradiction span;
- candidate-derived reading evidence is `mora_shadow`, while only audio-derived `phone`, `mora`,
  or `discrete_unit` evidence may authenticate a generated proposal;
- observed-eligible proposals are bound to the exact source-audio SHA-256;
- complete-path context scores are bound to path, context, scorer source, and immutable scorer
  profile digests;
- an applied second-pass result retains the complete first-pass candidate and ranking evidence;
- first-pass confidence and timestamp spans are not reused after a changed path;
- provisional edits remain unapplied by default;
- invariant or scorer failures retain the first-pass text and are recorded as attempted
  fail-closed decisions;
- standard JSON, observed TXT, SRT, Markdown, and VTT output paths consume the final observed
  transcript without mutating the original `LongformResult`.

The one-shot audit environment ran Ruff formatting and lint plus the complete dependency-free suite:

```text
353 passed, 8 skipped
```

The skips were the existing optional NumPy and PyTorch paths. The permanent repository CI remains
responsible for Python 3.11, Python 3.12, Windows, clean-wheel installation, optional CPU PyTorch,
and frontier-contract validation on the final non-bot head.

## Accuracy boundary

No Japanese CER, mora error, semantic-critical error, context false-correction, risk-coverage,
latency, or memory improvement is claimed here. Promotion requires locked speaker-disjoint Japanese
train/calibration/test manifests, a frozen global scorer, independently trained audio-to-phone and
audio-to-mora heads, and paired evaluation against the measured v0.2 first pass.
