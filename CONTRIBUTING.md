# Contributing

Preserve the invariant:

```text
observedTranscript != normalizedTranscript
```

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src tests scripts
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
semantic-asr demo --output runs/demo.json
python -m build --wheel
```

The base environment intentionally has no PyTorch and skips the optional training module.
Also validate the auxiliary heads in a separate CPU environment with `torch>=2.4` installed:
`python -m pytest -q tests/test_training_optional.py`.
CI checks Linux/Python 3.11 and 3.12, Windows/Python 3.12, and the CPU training path.
Workflows validate the event revision without formatting, committing, or pushing source changes.

Requirements:

- New evidence scores need calibration and missing-evidence behaviour.
- Semantic classes need deterministic tests and criticality rationale.
- Local-LLM changes must retain loopback-only, no-proxy, no-redirect and exact-ID checks.
- Mora changes need contracted-sound and special-mora fixtures.
- Schema changes must update examples and validation tests.
- Recognition-quality claims require a named held-out corpus, exact model revisions and reproducible metrics.
- Do not commit recordings, weights, credentials, absolute private paths or transcripts containing personal information.
