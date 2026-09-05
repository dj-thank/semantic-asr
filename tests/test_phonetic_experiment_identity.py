from __future__ import annotations

from dataclasses import replace

from _phonetic_experiment_fixture import manifest, protocol, utility_artifact

from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner
from semantic_asr.phonetic_experiment.selection import select_phonetic_arm


def test_candidate_pool_digest_excludes_generation_latency(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())
    pool = planner.plan(
        __import__(
            "semantic_asr.phonetic_experiment.planner",
            fromlist=["PlanningCaseView"],
        ).PlanningCaseView.from_case(experiment.cases[0]),
        protocol=protocol(),
    )

    changed_timing = replace(pool, generation_latency_ms=pool.generation_latency_ms + 100.0)

    assert changed_timing.digest == pool.digest


def test_decision_digest_excludes_selection_latency(tmp_path) -> None:
    experiment, runtime = manifest(tmp_path)
    plan = protocol()
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())
    pool = planner.plan(
        __import__(
            "semantic_asr.phonetic_experiment.planner",
            fromlist=["PlanningCaseView"],
        ).PlanningCaseView.from_case(experiment.cases[0]),
        protocol=plan,
    )
    decision = select_phonetic_arm(pool, plan.arm("phone+mora"))

    changed_timing = replace(
        decision,
        selection_latency_ms=decision.selection_latency_ms + 50.0,
    )

    assert changed_timing.digest == decision.digest
