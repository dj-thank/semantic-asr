"""Actual orchestrator budget regressions, not a mocked planner-only contract."""

from __future__ import annotations

import pytest
from test_longform import FakeAdapter

from semantic_asr.longform import SemanticASRTranscriber
from semantic_asr.planner import EvidenceBudget
from semantic_asr.teachers import DelayedTeacherPolicy, TeacherResult


class CountingTeacher:
    model = "counting-fixture"
    allow_legacy_cache_identity = True

    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def probabilities(self, candidates, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return TeacherResult(
            {c.candidate_id: 1 / len(candidates) for c in candidates},
            self.model,
            "fixture",
            "fixture",
            1.0,
            False,
        )


@pytest.mark.parametrize("cost,actions", [(0, 0), (0, 4), (1000, 0), (1, 1)])
def test_unfunded_teacher_never_executes(tmp_path, cost, actions):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture adapter does not read samples")
    teacher = CountingTeacher()
    result = SemanticASRTranscriber(
        FakeAdapter(),
        teacher=teacher,
        teacher_policy=DelayedTeacherPolicy(minimum_entropy=0),
        evidence_budget=EvidenceBudget(total_cost_ms=cost, max_actions=actions),
    ).transcribe(audio, duration_ms=1000)
    assert teacher.calls == 0
    assert result.segments[0].diagnostics["teacherUsed"] is False


def make_action(identifier="one", kind="local-teacher", cost=100):
    from semantic_asr.planner import EvidenceAction

    return EvidenceAction(identifier, kind, 0, 1000, cost, 0.5, 0.5, 0.005, (), False)


def make_execution(actions, *, cost=1000, count=2):
    from semantic_asr.evidence_execution import EvidenceExecution
    from semantic_asr.planner import EvidencePlan

    return EvidenceExecution(
        EvidencePlan(
            tuple(actions), (), cost, sum(a.estimated_cost_ms for a in actions), 0.5, "fixture"
        ),
        EvidenceBudget(cost, count),
    )


def test_actual_teacher_obeys_competing_action_count_and_cache(tmp_path):
    from semantic_asr.cache import EvidenceCache

    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    teacher, adapter = CountingTeacher(), FakeAdapter()
    with EvidenceCache(tmp_path / "cache.sqlite3") as cache:
        transcriber = SemanticASRTranscriber(
            adapter,
            teacher=teacher,
            cache=cache,
            teacher_policy=DelayedTeacherPolicy(minimum_entropy=0),
            evidence_budget=EvidenceBudget(1000, 2),
        )
        first = transcriber.transcribe(audio, duration_ms=1000)
        assert teacher.calls == 1 and len(adapter.requests) == 2
        execution = first.segments[0].diagnostics["evidenceExecution"]
        assert execution["attemptedActionCount"] == 2
        assert execution["admittedEstimatedCostMs"] <= 1000
        assert execution["completedUncachedActionCount"] == 2
        second = transcriber.transcribe(audio, duration_ms=1000)
        assert teacher.calls == 1 and len(adapter.requests) == 2
        cached = second.segments[0].diagnostics["evidenceExecution"]
        assert cached["cacheHitCount"] == 2
        assert cached["completedUncachedActionCount"] == 0
        assert first.observed_text == second.observed_text
        assert first.evidence_sha256 == second.evidence_sha256
        second.verify()


@pytest.mark.parametrize(
    "error", [TimeoutError("private path"), RuntimeError("secret"), ValueError("bad response")]
)
def test_optional_teacher_failure_retains_verified_observation(tmp_path, error):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    teacher = CountingTeacher(error)
    result = SemanticASRTranscriber(
        FakeAdapter(),
        teacher=teacher,
        teacher_policy=DelayedTeacherPolicy(minimum_entropy=0),
        evidence_budget=EvidenceBudget(1000, 2),
    ).transcribe(audio, duration_ms=1000)
    assert teacher.calls == 1
    assert "学校を" in result.observed_text
    assert result.segments[0].normalized.mode == "deterministic"
    ledger = result.segments[0].diagnostics["evidenceExecution"]
    assert ledger["failedActionCount"] == 1
    assert ledger["actions"][-1]["errorType"] == type(error).__name__
    assert str(error) not in str(ledger)
    result.verify()


def test_single_action_budget_cannot_also_run_teacher(tmp_path):
    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    teacher, adapter = CountingTeacher(), FakeAdapter()
    result = SemanticASRTranscriber(
        adapter,
        teacher=teacher,
        teacher_policy=DelayedTeacherPolicy(minimum_entropy=0),
        evidence_budget=EvidenceBudget(1000, 1),
    ).transcribe(audio, duration_ms=1000)
    assert len(adapter.requests) - 1 + teacher.calls == 1
    assert result.segments[0].diagnostics["evidenceExecution"]["attemptedActionCount"] == 1


def test_elapsed_overrun_prevents_next_call_without_claiming_preemption(monkeypatch):
    import semantic_asr.evidence_execution as execution_module

    clock = iter((1.0, 1.201))
    monkeypatch.setattr(execution_module.time, "monotonic", lambda: next(clock))
    one, two = make_action("one", cost=10), make_action("two", cost=10)
    run = make_execution([one, two], cost=100)
    assert run.run(one, lambda: ("slow result", False)) == "slow result"
    assert run.run(two, lambda: pytest.fail("overrun must prevent next launch")) is None
    report = run.diagnostics()
    assert report["elapsedMs"] == pytest.approx(201)
    assert report["admittedEstimatedCostMs"] == 10
    assert report["hardDeadlineEnforced"] is False
    assert report["actions"][1]["status"] == "budget-not-executed"


def test_duplicate_unapproved_and_cost_altered_actions_do_not_execute():
    from dataclasses import replace

    one = make_action()
    run = make_execution([one])
    with pytest.raises(ValueError, match="approved"):
        run.run(make_action("other"), lambda: pytest.fail("unapproved"))
    with pytest.raises(ValueError, match="approved"):
        run.run(replace(one, estimated_cost_ms=0), lambda: pytest.fail("altered"))
    assert run.run(one, lambda: ("cached", True)) == "cached"
    assert run.run(one, lambda: pytest.fail("duplicate")) is None
    report = run.diagnostics()
    report["actions"][0]["status"] = "tampered"
    assert run.diagnostics()["actions"][0]["status"] == "cache-hit"
    with pytest.raises(ValueError, match="duplicate"):
        make_execution([one, one])


@pytest.mark.parametrize("cost,count", [(0, 2), (1000, 0), (50, 2)])
def test_executor_checks_budget_even_for_malformed_external_plan(cost, count):
    one = make_action(cost=100)
    run = make_execution([one], cost=cost, count=count)
    assert run.run(one, lambda: pytest.fail("unfunded external plan")) is None
    assert run.diagnostics()["attemptedActionCount"] == 0


@pytest.mark.parametrize("bad", [True, 1.5, float("nan"), float("inf"), -1])
def test_budget_rejects_invalid_limits(bad):
    for field in ("total_cost_ms", "max_actions"):
        with pytest.raises((ValueError, TypeError)):
            EvidenceBudget(**{field: bad})
    if bad != 1.5:
        with pytest.raises((ValueError, TypeError)):
            EvidenceBudget(minimum_utility=bad)


def test_planner_requests_whole_pool_teacher_at_most_once():
    from dataclasses import replace

    from semantic_asr.adapters import DecodeRequest
    from semantic_asr.fusion import fuse_candidates
    from semantic_asr.planner import plan_evidence
    from semantic_asr.semantic_lattice import build_semantic_lattice

    candidates = FakeAdapter().decode(DecodeRequest("fixture"))
    ranked = fuse_candidates(candidates)
    lattice = build_semantic_lattice(candidates, segment_start_ms=0, segment_end_ms=1000)
    assert lattice.contradiction_islands
    lattice = replace(lattice, contradiction_islands=lattice.contradiction_islands * 3)
    plan = plan_evidence(
        ranked, lattice, enabled=("local-teacher",), budget=EvidenceBudget(10000, 8)
    )
    assert len(plan.selected) == 1
    assert plan.selected[0].kind == "local-teacher"


def test_empty_primary_candidates_fail_without_querying_teacher(tmp_path):
    class EmptyAdapter(FakeAdapter):
        def decode(self, request):
            return []

    audio = tmp_path / "fixture.wav"
    audio.write_bytes(b"fixture")
    teacher = CountingTeacher()
    with pytest.raises(RuntimeError, match="no candidates"):
        SemanticASRTranscriber(EmptyAdapter(), teacher=teacher).transcribe(audio, duration_ms=1000)
    assert teacher.calls == 0
