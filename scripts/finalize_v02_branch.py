#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _write(relative: str, content: str) -> None:
    (ROOT / relative).write_text(content, encoding="utf-8")


def _replace_if_present(relative: str, old: str, new: str) -> bool:
    content = _read(relative)
    if old not in content:
        return False
    updated = content.replace(old, new, 1)
    _write(relative, updated)
    return True


def _regex_replace(relative: str, pattern: str, replacement: str) -> bool:
    content = _read(relative)
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if count:
        _write(relative, updated)
        return True
    return False


def _append_once(relative: str, marker: str, section: str) -> bool:
    content = _read(relative)
    if marker in content:
        return False
    separator = "" if content.endswith("\n\n") else "\n" if content.endswith("\n") else "\n\n"
    _write(relative, content + separator + section.strip() + "\n")
    return True


def _prepend_changelog_once() -> bool:
    path = "CHANGELOG.md"
    content = _read(path)
    marker = "## [0.2.0] - Unreleased"
    if marker in content:
        return False
    section = """## [0.2.0] - Unreleased

### Added

- path-preserving CTranslate2/faster-whisper candidate aggregation with log-sum-exp surface mass;
- typed score semantics and held-out Platt/isotonic calibration contracts;
- character/mora/subword n-gram baselines and existing-candidate semantic MBR;
- constrained CPU reranker, sparse specialist neural reranker and multi-objective MWER/listwise training;
- finite-sample adaptive-K risk control and learned evidence gain/cost planning;
- proper causal sequence likelihood and dedicated local reranker scoring;
- compact acoustic-text verifier and guarded generated-candidate acceptance;
- leakage-safe manifests, paired bootstrap comparisons and quality/cost claim gates;
- evidence-safe Koemo integration contracts;
- revision-pinned research registry with provisional-source enforcement.

### Changed

- v0.2 keeps `observedTranscript != normalizedTranscript` and strengthens it by prohibiting generated candidates from becoming observed evidence before acoustic verification.
- numeric values written by a chat model are now preferences, never implicit probabilities.

### Claim boundary

- This release adds executable research and training infrastructure. It does not claim measured Japanese ASR improvement until rights-approved real-model benchmarks are completed.

"""
    heading_match = re.search(r"(?m)^# .+\n", content)
    if heading_match is None:
        _write(path, section + content)
    else:
        insertion = heading_match.end()
        _write(path, content[:insertion] + "\n" + section + content[insertion:])
    return True


def main() -> int:
    changed: list[str] = []

    import_fixes = (
        (
            "src/semantic_asr/risk_control.py",
            "from collections.abc import Iterable, Mapping, Sequence",
            "from collections.abc import Iterable, Sequence",
        ),
        (
            "src/semantic_asr/reranking.py",
            "from collections.abc import Iterable, Mapping, Sequence",
            "from collections.abc import Mapping, Sequence",
        ),
        (
            "src/semantic_asr/training_v2.py",
            "from typing import Any\n",
            "",
        ),
        (
            "src/semantic_asr/planner_v2.py",
            "from dataclasses import asdict, dataclass, field",
            "from dataclasses import dataclass, field",
        ),
    )
    for path, old, new in import_fixes:
        if _replace_if_present(path, old, new):
            changed.append(path)

    for path in ("pyproject.toml", "src/semantic_asr/__init__.py"):
        if not (ROOT / path).is_file():
            continue
        if _regex_replace(path, r'^(\s*version\s*=\s*)"0\.1\.0"\s*$', r'\1"0.2.0"'):
            changed.append(path)
        if _regex_replace(path, r'^(\s*__version__\s*=\s*)"0\.1\.0"\s*$', r'\1"0.2.0"'):
            changed.append(path)

    readme_section = """
## Semantic ASR v0.2 research stack

The v0.2 stack extends the evidence-preserving foundation with a falsifiable CPU-to-small-GPU cascade:

```text
path-preserving ASR candidates
  -> n-gram and MBR baselines
  -> compact constrained/learned reranking
  -> held-out calibration and adaptive risk control
  -> selective re-listening, second ear or acoustic verifier
  -> immutable observed transcript
  -> separate normalized derivative
```

The implementation adds typed score semantics, proper candidate sequence likelihood, duplicate-path probability mass, adaptive hypothesis count, sparse specialist routing, MWER/listwise training, guarded generative proposals, leakage-safe manifests, paired statistics and an evidence-safe Koemo bridge.

Start with:

```bash
python examples/v02_cpu_demo.py
python scripts/train_reranker_v2.py \
  examples/v02-ranking-groups.jsonl \
  --output reranker.json \
  --epochs 250
python scripts/validate_v02_manifest.py examples/v02-experiment-manifest.json
```

Design and research boundaries:

- [`docs/ARCHITECTURE_V0.2.md`](docs/ARCHITECTURE_V0.2.md)
- [`docs/RESEARCH_2026-08-31_V0.2.md`](docs/RESEARCH_2026-08-31_V0.2.md)
- [`docs/TRAINING_V0.2.md`](docs/TRAINING_V0.2.md)
- [`docs/EXPERIMENT_MATRIX_V0.2.md`](docs/EXPERIMENT_MATRIX_V0.2.md)
- [`docs/KOEMO_INTEGRATION.md`](docs/KOEMO_INTEGRATION.md)

The code and model-free tests do not constitute a measured accuracy claim. Real-model improvements require the locked held-out protocol and paired confidence intervals.
"""
    if _append_once(
        "README.md",
        "## Semantic ASR v0.2 research stack",
        readme_section,
    ):
        changed.append("README.md")

    if _prepend_changelog_once():
        changed.append("CHANGELOG.md")

    print("Updated:" if changed else "No changes required.", *changed, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
