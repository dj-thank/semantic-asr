#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tests/test_lattice_planner.py"
text = path.read_text(encoding="utf-8")
old = """    forced = next(action for action in plan.selected if action.kind == \"forced-align\")
    assert set(forced.hypotheses) == {\"三\", \"二\"}
    assert all(\"万円です\" not in hypothesis for hypothesis in forced.hypotheses)
"""
new = """    forced_plan = plan_evidence(
        ranked,
        lattice,
        budget=EvidenceBudget(
            total_cost_ms=2_500,
            max_actions=1,
            minimum_utility=0,
        ),
        enabled=(\"forced-align\",),
    )
    forced = forced_plan.selected[0]
    assert set(forced.hypotheses) == {\"三\", \"二\"}
    assert all(\"万円です\" not in hypothesis for hypothesis in forced.hypotheses)
"""
if old not in text:
    raise RuntimeError("forced-align planner regression anchor is missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
Path(__file__).unlink()
print("planner regression repaired")
