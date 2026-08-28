# Contributing

Preserve the invariant:

```text
observedTranscript != normalizedTranscript
```

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src tests
ruff check src tests
pytest -q
semantic-asr demo --output runs/demo.json
python -m build --wheel
```

Requirements:

- New evidence scores need calibration and missing-evidence behaviour.
- Semantic classes need deterministic tests and criticality rationale.
- Local-LLM changes must retain loopback-only, no-proxy, no-redirect and exact-ID checks.
- Mora changes need contracted-sound and special-mora fixtures.
- Schema changes must update examples and validation tests.
- Recognition-quality claims require a named held-out corpus, exact model revisions and reproducible metrics.
- Do not commit recordings, weights, credentials, absolute private paths or transcripts containing personal information.
