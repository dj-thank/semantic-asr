"""Retired source editors must not return as supported execution dependencies."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETIRED = {
    "apply_pr_review_fixes_v3",
    "apply_v02_post_review",
    "repair_pr_review_test",
}


def test_retired_scripts_are_removed_and_historical_plan_paths_are_explicit():
    for name in RETIRED:
        assert not (ROOT / "scripts" / f"{name}.py").exists()
    plan = json.loads((ROOT / "docs/development/release-plan.json").read_text())
    cleanup = next(row for row in plan["tasks"] if row["issue"] == 27)
    assert set(cleanup["historical_paths"]) == {f"scripts/{name}.py" for name in RETIRED}
    assert all((ROOT / path).is_file() for path in cleanup["source_paths"])
    assert cleanup["state"] == "partial"


def test_runtime_and_workflows_do_not_depend_on_retired_source_editors():
    for directory in ("src", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    imported = [row.name for row in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or "", *(row.name for row in node.names)]
                else:
                    continue
                assert not any(part in RETIRED for name in imported for part in name.split("."))
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert not any(f"{name}.py" in text for name in RETIRED)
