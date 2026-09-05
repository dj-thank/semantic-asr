from __future__ import annotations

from dataclasses import replace

from _context_phonetic_factorial_fixture import (
    factorial_manifest,
    factorial_protocol,
    utility_artifact,
)

from semantic_asr.context_phonetic_experiment.planner import (
    prepare_context_phonetic_experiment,
)
from semantic_asr.context_phonetic_experiment.selection import (
    select_context_phonetic_arm,
)
from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner


def test_prepared_and_decision_identity_exclude_wall_clock_latency(tmp_path) -> None:
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    prepared = prepare_context_phonetic_experiment(
        manifest,
        protocol,
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
        scorer,
    )
    case = prepared.cases[0]
    changed_case = replace(
        case,
        pool=replace(
            case.pool,
            generation_latency_ms=case.pool.generation_latency_ms + 100.0,
        ),
        ordered=replace(
            case.ordered,
            scoring_latency_ms=case.ordered.scoring_latency_ms + 100.0,
        ),
        shuffled=replace(
            case.shuffled,
            scoring_latency_ms=case.shuffled.scoring_latency_ms + 100.0,
        ),
    )

    assert changed_case.digest == case.digest

    arm = protocol.arm("phone+mora:ordered")
    decision = select_context_phonetic_arm(case, arm, protocol)
    changed_decision = replace(
        decision,
        selection_latency_ms=decision.selection_latency_ms + 100.0,
    )
    assert changed_decision.digest == decision.digest
