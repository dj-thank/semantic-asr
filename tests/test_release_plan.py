"""Offline structural checks for the dated execution plan, not completion claims."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "development" / "release-plan.json"


def load_plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_release_plan_ids_dependencies_and_priority_are_consistent():
    plan = load_plan()
    assert plan["schema"] == "semantic-asr-release-plan-v1"
    tasks = plan["tasks"]
    by_id = {task["issue"]: task for task in tasks}
    assert len(by_id) == len(tasks)
    assert set(by_id) == {19, *range(24, 41)}
    assert plan["epic"] == 23 and plan["research_parent"] == 21
    visited, active = set(), set()

    def visit(number):
        assert number in by_id, f"unknown dependency: {number}"
        assert number not in active, f"cyclic dependency: {number}"
        if number in visited:
            return
        active.add(number)
        task = by_id[number]
        assert task["priority"] in {"P0", "P1", "P2"}
        assert task["state"] in {"planned", "partial", "blocked", "completed"}
        assert task["issue_url"] == (f"https://github.com/{plan['repository']}/issues/{number}")
        assert len(task["depends_on"]) == len(set(task["depends_on"]))
        for dependency in task["depends_on"]:
            visit(dependency)
        assert set(task.get("promotion_requires", ())) <= set(by_id)
        active.remove(number)
        visited.add(number)

    for number in by_id:
        visit(number)
    for number in plan["parallel_start"]:
        assert not by_id[number]["depends_on"]


def test_source_paths_exist_and_new_tests_are_explicitly_proposed():
    for task in load_plan()["tasks"]:
        assert task["role"] and task["source_paths"]
        for value in task["source_paths"] + task["new_test_paths"]:
            path = Path(value)
            assert not path.is_absolute() and ".." not in path.parts
        for value in task["source_paths"]:
            assert (ROOT / value).is_file(), value
        for value in task["new_test_paths"]:
            assert value.startswith("tests/test_") and value.endswith(".py")
            # A planned test is not required to exist before its implementation.


def test_training_tasks_require_real_weight_evidence_and_keep_promotion_separate():
    tasks = {task["issue"]: task for task in load_plan()["tasks"]}
    required = {
        "optimizer_updates",
        "changed_trainable_tensors",
        "unchanged_frozen_tensors",
        "checkpoint_sha256",
        "fresh_process_reload",
        "separate_evaluation",
    }
    for number in (36, 37):
        assert set(tasks[number]["required_weight_evidence"]) == required
        assert {26, 28, 29, 35} <= set(tasks[number]["depends_on"])
    assert tasks[40]["promotion_requires"]
    assert not tasks[40]["depends_on"]  # Documentation may begin before release gates.


def test_execution_entrypoints_and_templates_are_present():
    for value in (
        "AGENTS.md",
        "docs/development/README.md",
        "docs/development/TEMPLATES.md",
        "docs/development/AUDIT_2026-09-05.md",
        "docs/research/METHODS_2026-09-05.md",
        ".github/ISSUE_TEMPLATE/implementation.yml",
        ".github/ISSUE_TEMPLATE/research.yml",
        ".github/pull_request_template.md",
    ):
        assert (ROOT / value).is_file(), value
