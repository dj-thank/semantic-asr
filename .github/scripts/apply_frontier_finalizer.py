from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one patch anchor in {path}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def main() -> None:
    replace_once(
        Path("src/semantic_asr/advanced_adapters.py"),
        "        self.ranker = ranker\n",
        "        if ranker is not None:\n"
        "            from .ranker_guard import AcousticGuardedRanker\n\n"
        "            if not isinstance(ranker, AcousticGuardedRanker):\n"
        "                ranker = AcousticGuardedRanker(ranker)\n"
        "        self.ranker = ranker\n",
    )
    append_once(
        Path("docs/ARCHITECTURE.md"),
        "## 15. Acoustic-guarded reranking",
        """
## 15. Acoustic-guarded reranking

Every text-only candidate ranker used by the adaptive runtime is wrapped by the
acoustic Honeytrap guard. The guard compares relative language preference with
relative Whisper/CTranslate2 path mass and subtracts a penalty when preference
exceeds acoustic support beyond a deadband. A ranker calibration profile is
valid only when it was fitted on this guarded output.

The guard is a safety constraint, not an accuracy claim. Its strength and
thresholds require calibration and ablation on speaker-disjoint Japanese audio.
""",
    )
    append_once(
        Path("CHANGELOG.md"),
        "### Final frontier hardening",
        """
### Final frontier hardening

- added group-aware finite-sample Learn-Then-Test risk control for adaptive K;
- added correct full-sequence causal-LM candidate likelihoods;
- added decoupled Top-K and length-normalized pairwise distillation objectives;
- preserved representative path fields while aggregating exact-surface path mass;
- enabled acoustic Honeytrap guarding for runtime text rerankers;
- registered LFM2/LFM2.5 edge, Japanese, and continuous-audio research tiers.
""",
    )


if __name__ == "__main__":
    main()
