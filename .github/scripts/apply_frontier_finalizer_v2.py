from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


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


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def patch_runtime() -> None:
    replace_once(
        ROOT / "src/semantic_asr/advanced_adapters.py",
        "        self.ranker = ranker\n",
        "        if ranker is not None:\n"
        "            from .ranker_guard import AcousticGuardedRanker\n\n"
        "            if not isinstance(ranker, AcousticGuardedRanker):\n"
        "                ranker = AcousticGuardedRanker(ranker)\n"
        "        self.ranker = ranker\n",
    )


def patch_candidate_pool() -> None:
    path = ROOT / "src/semantic_asr/candidate_pool.py"
    replace_once(
        path,
        "        token_count = max(_token_count(path, text) for path in paths)\n"
        "        aggregate_average = aggregate_logprob / (token_count + 1)\n"
        "        metadata[\"aggregateCumulativeLogprob\"] = aggregate_logprob\n"
        "        metadata[\"aggregateAverageLogprob\"] = aggregate_average\n"
        "        metadata[\"pathProbabilityMassAggregated\"] = aggregate_path_mass\n",
        "        representative = _representative(candidates)\n"
        "        token_count = max(1, len(representative.token_ids))\n"
        "        aggregate_average = aggregate_logprob / (token_count + 1)\n"
        "        metadata[\"aggregateCumulativeLogprob\"] = aggregate_logprob\n"
        "        metadata[\"aggregateAverageLogprob\"] = aggregate_average\n"
        "        metadata[\"pathProbabilityMassAggregated\"] = aggregate_path_mass\n"
        "        metadata[\"representativePathId\"] = representative.candidate_id\n"
        "        metadata[\"representativeSequenceScore\"] = representative.sequence_score\n"
        "        metadata[\"representativeAverageLogprob\"] = representative.avg_logprob\n",
    )
    replace_once(
        path,
        "        metadata = _aggregate_metadata(\n"
        "            candidates,\n"
        "            paths,\n"
        "            aggregate_path_mass=aggregate_path_mass,\n"
        "        )\n"
        "        aggregate_logprob = _finite(metadata.get(\"aggregateCumulativeLogprob\"))\n"
        "        aggregate_avg = _finite(metadata.get(\"aggregateAverageLogprob\"))\n"
        "        merged.append(\n"
        "            CandidateEvidence(\n"
        "                candidate_id=representative.candidate_id,\n"
        "                text=text,\n"
        "                token_ids=representative.token_ids,\n"
        "                acoustic=(\n"
        "                    aggregate_avg if aggregate_avg is not None else representative.acoustic\n"
        "                ),\n"
        "                mora=representative.mora,\n"
        "                lexical=representative.lexical,\n"
        "                preservation=representative.preservation,\n"
        "                cross_model=representative.cross_model,\n"
        "                teacher=None,\n"
        "                reading=representative.reading,\n"
        "                mora_units=representative.mora_units,\n"
        "                rank=representative.rank,\n"
        "                hypothesis_count=len(by_surface),\n"
        "                sequence_score=(\n"
        "                    aggregate_logprob\n"
        "                    if aggregate_logprob is not None\n"
        "                    else representative.sequence_score\n"
        "                ),\n"
        "                avg_logprob=(\n"
        "                    aggregate_avg\n"
        "                    if aggregate_avg is not None\n"
        "                    else representative.avg_logprob\n"
        "                ),\n"
        "                beam_confidence=representative.beam_confidence,\n"
        "                source=representative.source,\n"
        "                metadata=metadata,\n"
        "            )\n"
        "        )\n",
        "        metadata = _aggregate_metadata(\n"
        "            candidates,\n"
        "            paths,\n"
        "            aggregate_path_mass=aggregate_path_mass,\n"
        "        )\n"
        "        merged.append(\n"
        "            CandidateEvidence(\n"
        "                candidate_id=representative.candidate_id,\n"
        "                text=text,\n"
        "                token_ids=representative.token_ids,\n"
        "                acoustic=representative.acoustic,\n"
        "                mora=representative.mora,\n"
        "                lexical=representative.lexical,\n"
        "                preservation=representative.preservation,\n"
        "                cross_model=representative.cross_model,\n"
        "                teacher=None,\n"
        "                reading=representative.reading,\n"
        "                mora_units=representative.mora_units,\n"
        "                rank=representative.rank,\n"
        "                hypothesis_count=len(by_surface),\n"
        "                sequence_score=representative.sequence_score,\n"
        "                avg_logprob=representative.avg_logprob,\n"
        "                beam_confidence=representative.beam_confidence,\n"
        "                source=representative.source,\n"
        "                metadata=metadata,\n"
        "            )\n"
        "        )\n",
    )


def patch_project() -> None:
    replace_once(
        ROOT / "pyproject.toml",
        "[project.scripts]\nsemantic-asr = \"semantic_asr.cli:main\"\n",
        "[project.scripts]\n"
        "semantic-asr = \"semantic_asr.cli:main\"\n"
        "semantic-asr-risk-control = \"semantic_asr.risk_control:main\"\n",
    )


def write_docs() -> None:
    append_once(
        ROOT / "docs/ARCHITECTURE.md",
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
        ROOT / "CHANGELOG.md",
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
    write(
        ROOT / "docs/FRONTIER_EXTENSIONS_V0.2.md",
        r'''
# Semantic ASR v0.2 final frontier extensions

This note records the final hardening applied before the v0.2 merge.

## Finite-sample adaptive-K risk control

`adaptive.py` selects K from candidate mass, entropy, margin, and semantic
criticality. `risk_control.py` adds a conservative Learn-Then-Test calibration
layer. It averages repeated segments within a declared independent `group_id`,
uses a Bonferroni-adjusted Hoeffding upper bound over every evaluated K, and
selects the least-cost K only when the bound meets the declared target risk.

The confidence statement is conditional on bounded loss, independent groups,
and a fixed evaluated K family. It is not a universal guarantee under arbitrary
dataset or microphone shift.

## Path-mass evidence semantics

Multiple Whisper/CTranslate2 paths may decode to the same exact Japanese surface
string. Their cumulative log probability is aggregated with log-sum-exp, but the
representative path's native `avg_logprob`, `sequence_score`, and `acoustic`
fields remain unchanged. Aggregate values live in explicit metadata and are used
for candidate-set mass and the acoustic guard. This avoids inventing a synthetic
path while preserving all available probability mass.

## Acoustic guard for text rerankers

`ranker_guard.py` provides an acoustic Honeytrap guard. The guard compares
relative ranker preference with relative acoustic path mass and penalizes
language preference that exceeds acoustic support beyond a deadband. Any
held-out calibration must be fitted on the guarded output, not on the raw inner
ranker.

## Full-sequence causal-LM scoring

`sequence_scoring.py` sums the assigned next-token log probability for every
candidate token and exposes the per-token average separately. It explicitly does
not use a final-step maximum vocabulary probability and does not call an
uncalibrated likelihood a correctness probability.

## Edge-student distillation

`distillation_objectives.py` adds:

- decoupled Top-K KL, separating retained candidate mass from the conditional
  distribution inside Top-K;
- length-normalized pairwise alignment with an absolute chosen-candidate anchor.

These objectives complement the existing multi-teacher consensus and listwise
semantic-MWER training paths.

## Repository governance

Model presets remain disabled as runtime defaults until an immutable revision,
rights approval, and a speaker-disjoint held-out Japanese benchmark are recorded.
No synthetic smoke result is presented as a real-audio accuracy claim.
''',
    )
    write(
        ROOT / "docs/LFM_INTEGRATION.md",
        r'''
# LFM integration and research plan

Liquid Foundation Models are treated as a first-class research family in
Semantic ASR, not as a generic chat-model afterthought.

## Verified model roles

The repository registry includes the following candidates, all disabled as
runtime defaults until an immutable revision, rights decision, and held-out
Japanese evaluation are recorded:

| Preset | Intended role |
|---|---|
| `LiquidAI/LFM2-350M` | ultra-compact full-sequence candidate likelihood and edge student |
| `LiquidAI/LFM2-700M` | higher-quality CPU/small-GPU sequence scorer |
| `LiquidAI/LFM2.5-1.2B-JP` | Japanese-specialized scorer or offline teacher |
| `LiquidAI/LFM2.5-Audio-1.5B-JP` | continuous-audio second ear and acoustic-verifier comparison |

Model size, general text benchmarks, or a Japanese language tag do not establish
ASR reranking quality. Each tier must be measured on the same locked Japanese
speech corpus and deployment hardware as the other candidates.

## Architecture translations

### Hardware-in-the-loop design

LFM2's hardware-aware design principle maps to Semantic ASR's measured Pareto
frontier. Candidate cascades are selected from held-out quality, p50/p95 latency,
peak host/GPU memory, and optional energy on the actual target device. Parameter
count alone does not choose the runtime default.

### Compact recurrent/local computation

LFM-style efficient recurrent/local processing motivates a compact acoustic
candidate verifier that reuses continuous encoder features instead of always
running a second complete ASR model. This is a research translation, not a
reproduction of Liquid model layers or kernels.

### Decoupled Top-K distillation

`distillation_objectives.py` adapts decoupled Top-K knowledge transfer to ASR
candidate distributions. The student separately learns:

1. how much teacher mass belongs inside the retained candidate set;
2. how that mass is distributed among the retained candidates.

This avoids pretending that every omitted N-best tail candidate has zero mass.

### Continuous audio features

`LFM2.5-Audio-1.5B-JP` is evaluated as an independent second ear and as a teacher
for the proposed compact acoustic verifier. It may never directly author the
observed transcript. Any generated or corrected text must pass the same acoustic,
mora, critical-edit, and insertion checks as every other generated proposal.

## Required LFM ablations

```text
KenLM char/mora/subword
ModernBERT-Ja-130M
LFM2-350M full-sequence likelihood
LFM2-700M full-sequence likelihood
Qwen3-Reranker-0.6B
LFM2.5-1.2B-JP
Qwen3-ASR-0.6B second ear
LFM2.5-Audio-1.5B-JP second ear
compact distilled acoustic verifier
```

Report CER, mora error, semantic-critical error, unsupported insertion,
calibration/risk coverage, RTF, latency, memory, energy where available, and
invocation rate. Publish negative results: LFM is valuable only where measured
quality per unit cost improves.

## Primary sources

- LFM2 technical report: `arXiv:2511.23404`
- official LiquidAI model repositories for the exact model IDs above

No model weights or vendor implementation are copied into Semantic ASR.
''',
    )


def write_tests() -> None:
    write(
        ROOT / "tests/test_v02_final_frontier.py",
        r'''
import math

import pytest

from semantic_asr.candidate_pool import build_candidate_pool, candidate_distribution
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.distillation_objectives import (
    decoupled_topk_kl,
    length_normalized_pairwise_alignment,
)
from semantic_asr.model_presets import PRESETS, preset_by_name, recommended_edge_cascade
from semantic_asr.ranker_guard import AcousticGuardedRanker
from semantic_asr.risk_control import (
    LearnThenTestConfig,
    ParetoPoint,
    RiskObservation,
    learn_then_test_k,
    pareto_frontier,
)
from semantic_asr.score_semantics import ScoreKind
from semantic_asr.sequence_scoring import (
    sequence_logprob_from_logits,
    sequence_scores_to_preferences,
)


def test_group_level_ltt_selects_least_cost_safe_k():
    observations = []
    for index in range(1_200):
        observations.append(RiskObservation(f"3-{index}", f"g-{index}", 3, 0.015, 20.0))
        observations.append(RiskObservation(f"5-{index}", f"g-{index}", 5, 0.005, 34.0))
    result = learn_then_test_k(
        observations,
        config=LearnThenTestConfig(target_risk=0.08, delta=0.05, minimum_groups=100),
    )
    assert result.selected_k == 3
    assert all(estimate.accepted for estimate in result.estimates)


def test_repeated_segments_do_not_inflate_group_count():
    rows = [RiskObservation(f"s-{i}", "same", 5, 0.0, 10.0) for i in range(500)]
    result = learn_then_test_k(rows, config=LearnThenTestConfig(minimum_groups=2))
    assert result.estimates[0].sample_count == 500
    assert result.estimates[0].group_count == 1
    assert result.selected_k is None


def test_pareto_frontier_removes_dominated_point():
    result = pareto_frontier(
        [
            ParetoPoint("fast", 0.8, 10, 100),
            ParetoPoint("dominated", 0.7, 12, 120),
            ParetoPoint("quality", 0.9, 20, 150),
        ]
    )
    assert {row.name for row in result} == {"fast", "quality"}


def test_identical_topk_distribution_has_zero_loss():
    result = decoupled_topk_kl(
        {"a": 0.6, "b": 0.3, "c": 0.1},
        {"a": 0.6, "b": 0.3, "c": 0.1},
        k=2,
    )
    assert result.total == pytest.approx(0.0, abs=1e-10)


def test_topk_membership_tracks_tail_mass():
    result = decoupled_topk_kl(
        {"a": 0.6, "b": 0.3, "c": 0.1},
        {"a": 0.4, "b": 0.2, "c": 0.4},
        k=2,
    )
    assert result.membership_kl > 0
    assert result.student_topk_mass < result.teacher_topk_mass


def test_length_normalized_alignment_prefers_better_candidate():
    good = length_normalized_pairwise_alignment(
        chosen_logprob=-2,
        rejected_logprob=-8,
        chosen_length=4,
        rejected_length=8,
    )
    bad = length_normalized_pairwise_alignment(
        chosen_logprob=-8,
        rejected_logprob=-2,
        chosen_length=8,
        rejected_length=4,
    )
    assert good.total < bad.total


def test_full_sequence_logprob_scores_every_token():
    score = sequence_logprob_from_logits(
        [[3.0, 1.0], [0.0, 2.0]], [0, 1], candidate_id="a"
    )
    expected = (3 - math.log(math.exp(3) + math.exp(1))) + (
        2 - math.log(math.exp(0) + math.exp(2))
    )
    assert score.sum_logprob == pytest.approx(expected)
    evidence = score.as_evidence()
    assert evidence.kind == ScoreKind.LOG_LIKELIHOOD
    assert not evidence.usable_as_probability


def test_likelihoods_become_relative_mass_not_probability_claims():
    a = sequence_logprob_from_logits([[2, 0]], [0], candidate_id="a")
    b = sequence_logprob_from_logits([[0, 2]], [0], candidate_id="b")
    mass = sequence_scores_to_preferences([a, b])
    assert mass["a"] > mass["b"]
    assert sum(mass.values()) == pytest.approx(1.0)


class FluencyOnly:
    name = "fluency"

    def score(self, candidates, **kwargs):
        del kwargs
        return {
            candidate.candidate_id: (5.0 if candidate.candidate_id == "fluent" else 1.0)
            for candidate in candidates
        }


def test_guard_blocks_fluent_acoustically_unsupported_candidate():
    supported = CandidateEvidence(
        "supported", "明日は行きません", avg_logprob=-0.05, acoustic=-0.05
    )
    fluent = CandidateEvidence(
        "fluent", "明日は行きます", avg_logprob=-2.0, acoustic=-2.0
    )
    ranker = AcousticGuardedRanker(FluencyOnly())
    scores = ranker.score([supported, fluent])
    assert scores["supported"] > scores["fluent"]
    assert ranker.last_diagnostics["fluent"]["penalty"] > 0


def test_lfm_roles_are_explicit_and_not_runtime_defaults():
    assert preset_by_name("lfm2-350m").role == "causal-scorer"
    assert preset_by_name("lfm2.5-audio-1.5b-jp").role == "audio-second-ear"
    assert any(row.name == "qwen3-reranker-0.6b" for row in PRESETS)
    assert not any(row.ready_for_runtime_default for row in PRESETS)
    cascade = recommended_edge_cascade()
    assert cascade[0].role == "ngram"
    assert cascade[-1].role == "audio-second-ear"


def test_surface_aggregation_preserves_representative_path_fields():
    best = CandidateEvidence(
        "a",
        "東京に行きます",
        token_ids=(1, 2, 3),
        acoustic=-0.2,
        sequence_score=-0.5,
        avg_logprob=-0.2,
        rank=1,
        metadata={"cumulativeLogprob": -0.8},
    )
    alternate = CandidateEvidence(
        "b",
        "東京に行きます",
        token_ids=(4, 5, 6),
        acoustic=-0.8,
        sequence_score=-1.2,
        avg_logprob=-0.8,
        rank=2,
        metadata={"cumulativeLogprob": -1.4},
    )
    pool = build_candidate_pool([[best, alternate]])
    merged = pool.candidates[0]
    assert merged.avg_logprob == best.avg_logprob
    assert merged.sequence_score == best.sequence_score
    assert merged.acoustic == best.acoustic
    assert merged.metadata["surfacePathCount"] == 2
    assert merged.metadata["pathProbabilityMassAggregated"] is True
    assert merged.metadata["aggregateCumulativeLogprob"] > -0.8
    assert candidate_distribution(pool)[merged.candidate_id] == pytest.approx(1.0)
''',
    )


def remove_temporary_automation() -> None:
    for relative in (
        ".github/scripts/apply_frontier_finalizer.py",
        ".github/workflows/apply-frontier-finalizer-once.yml",
        ".github/scripts/apply_frontier_finalizer_v2.py",
        ".github/workflows/apply-frontier-finalizer-v2-once.yml",
    ):
        (ROOT / relative).unlink(missing_ok=True)


def main() -> None:
    patch_runtime()
    patch_candidate_pool()
    patch_project()
    write_docs()
    write_tests()
    remove_temporary_automation()


if __name__ == "__main__":
    main()
