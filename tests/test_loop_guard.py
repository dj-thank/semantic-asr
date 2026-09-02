from __future__ import annotations

import pytest

from semantic_asr.advanced_adapters import (
    LoopGuardConfig,
    apply_loop_guard,
    compression_ratio,
    repeated_ngram_fraction,
)
from semantic_asr.contracts import CandidateEvidence


def test_compression_ratio_flags_repetition_loops() -> None:
    loop = "そこで木行きしたのは、" * 20
    assert compression_ratio(loop) > 2.4
    assert compression_ratio("そこで目撃したのは。") < 2.4
    assert compression_ratio("") == 1.0


def test_repeated_ngram_fraction() -> None:
    assert repeated_ngram_fraction([1, 2, 3, 4, 5, 6, 7, 8]) == 0.0
    assert repeated_ngram_fraction([1, 2, 3, 4] * 6) > 0.8
    assert repeated_ngram_fraction([1, 2, 3]) == 0.0
    with pytest.raises(ValueError):
        repeated_ngram_fraction([1, 2], order=0)


def test_token_budget_scales_with_duration_and_is_disabled_when_off() -> None:
    guard = LoopGuardConfig()
    assert guard.max_new_tokens(1.4) == 32 + int(1.4 * 14.0)
    assert guard.max_new_tokens(60.0) == 440
    assert LoopGuardConfig(enabled=False).max_new_tokens(1.4) is None
    assert LoopGuardConfig(enabled=False).stages == (("beam", 0.0),)
    assert [stage for stage, _ in guard.stages][:2] == ["beam", "sample-t0.2"]


def test_enrichment_stage_is_optional() -> None:
    assert LoopGuardConfig().enrichment_stage is None
    stage = LoopGuardConfig(extra_samples=8, extra_sample_temperature=0.7).enrichment_stage
    assert stage == ("mbr-sample-t0.7", 0.7)
    with pytest.raises(ValueError):
        LoopGuardConfig(extra_samples=-1)


def test_degeneracy_reasons() -> None:
    guard = LoopGuardConfig()
    loop = "ここで、" * 30
    verdict = guard.degeneracy(loop, list(range(4)) * 30, -0.05)
    assert verdict["degenerate"] is True
    assert "compression-ratio" in verdict["degenerateReasons"]
    assert "repeated-ngram" in verdict["degenerateReasons"]
    healthy = guard.degeneracy("そこで目撃したのは。", [1, 2, 3, 4, 5, 6], -0.05)
    assert healthy["degenerate"] is False
    assert guard.degeneracy("短い", [1, 2], -3.0)["degenerateReasons"] == ["low-logprob"]
    over_budget = guard.degeneracy(
        "".join(chr(0x3042 + index) for index in range(40)),
        list(range(40)),
        -0.1,
        duration_seconds=1.0,
    )
    assert over_budget["degenerateReasons"] == ["character-budget"]
    assert over_budget["characterBudget"] == 8 + 12
    assert LoopGuardConfig(enabled=False).degeneracy(loop, [1] * 40, -5.0)["degenerate"] is False


def _candidate(
    identifier: str, *, degenerate: bool, reasons: tuple[str, ...] = ("compression-ratio",)
) -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id=identifier,
        text=identifier,
        acoustic=-0.1,
        metadata={
            "degenerate": degenerate,
            "degenerateReasons": list(reasons) if degenerate else [],
        },
    )


def test_apply_loop_guard_keeps_low_logprob_only_paths() -> None:
    rows = [
        _candidate("quiet", degenerate=True, reasons=("low-logprob",)),
        _candidate("loop", degenerate=True, reasons=("low-logprob", "repeated-ngram")),
        _candidate("clean", degenerate=False),
    ]
    kept = apply_loop_guard(rows, config=LoopGuardConfig())
    assert [row.candidate_id for row in kept] == ["clean", "quiet"]
    assert kept[0].metadata["rejectedDegeneratePaths"] == 1
    assert kept[0].metadata["demotedLowLogprobPaths"] == 1


def test_apply_loop_guard_drops_or_demotes_but_never_empties() -> None:
    rows = [_candidate("loop", degenerate=True), _candidate("clean", degenerate=False)]
    dropped = apply_loop_guard(rows, config=LoopGuardConfig())
    assert [row.candidate_id for row in dropped] == ["clean"]
    assert dropped[0].metadata["rejectedDegeneratePaths"] == 1
    demoted = apply_loop_guard(rows, config=LoopGuardConfig(drop_degenerate=False))
    assert [row.candidate_id for row in demoted] == ["clean", "loop"]
    only_loops = apply_loop_guard([_candidate("loop", degenerate=True)], config=LoopGuardConfig())
    assert [row.candidate_id for row in only_loops] == ["loop"]


def test_generate_candidates_cli_exposes_loop_guard_flags() -> None:
    from semantic_asr.frontier_cli import build_parser

    args = build_parser().parse_args(
        ["generate-candidates", "manifest.jsonl", "--output", "out.jsonl", "--no-loop-guard"]
    )
    assert args.no_loop_guard is True
    assert args.fallback_temperatures == "0.2,0.4,0.6,0.8,1.0"
