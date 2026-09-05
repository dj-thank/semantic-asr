"""Admission and receipts for a single window's optional evidence calls.

Planning estimates are not elapsed time. Cache lookups consume one admitted action
and its conservative estimate, but are distinguished from uncached completions.
A synchronous call cannot be preempted here: elapsed overrun prevents the NEXT
call. Primary ASR/preprocessing and its enrichers are outside this optional budget.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from .planner import EvidenceAction, EvidenceBudget, EvidencePlan

T = TypeVar("T")


class EvidenceExecution:
    def __init__(self, plan: EvidencePlan, budget: EvidenceBudget) -> None:
        self.budget = budget
        self.approved = {action.action_id: action for action in plan.selected}
        if len(self.approved) != len(plan.selected):
            raise ValueError("evidence plan contains duplicate action IDs")
        self.records: list[dict[str, object]] = []
        self.attempted = 0
        self.estimated_ms = 0
        self.elapsed_ms = 0.0
        self.seen: set[str] = set()

    def run(self, action: EvidenceAction, callback: Callable[[], tuple[T, bool]]) -> T | None:
        if self.approved.get(action.action_id) != action:
            raise ValueError("evidence action was not approved by the planner")
        if isinstance(action.estimated_cost_ms, bool) or not isinstance(
            action.estimated_cost_ms, int
        ):
            raise TypeError("estimated action cost must be an integer")
        if action.estimated_cost_ms < 0:
            raise ValueError("estimated action cost must be non-negative")
        row: dict[str, object] = {
            "actionId": action.action_id,
            "kind": action.kind,
            "estimatedCostMs": action.estimated_cost_ms,
            "elapsedMs": 0.0,
            "cacheHit": False,
        }
        self.records.append(row)
        if action.action_id in self.seen:
            row["status"] = "duplicate-not-executed"
            return None
        self.seen.add(action.action_id)
        if (
            self.budget.total_cost_ms == 0
            or self.attempted >= self.budget.max_actions
            or self.estimated_ms + action.estimated_cost_ms > self.budget.total_cost_ms
            or self.elapsed_ms >= self.budget.total_cost_ms
        ):
            row["status"] = "budget-not-executed"
            return None
        self.attempted += 1
        self.estimated_ms += action.estimated_cost_ms
        started = time.monotonic()
        try:
            result, cached = callback()
            row["cacheHit"] = cached
            row["status"] = "cache-hit" if cached else "completed"
            return result
        except (OSError, RuntimeError, ValueError) as exc:
            # Retain first-pass evidence on backend failure. Do not publish arbitrary
            # backend messages (they may contain paths, prompts or credentials).
            row["status"] = "failed"
            row["errorType"] = type(exc).__name__
            return None
        finally:
            elapsed = max(0.0, (time.monotonic() - started) * 1000)
            row["elapsedMs"] = elapsed
            self.elapsed_ms += elapsed

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema": "semantic-asr-evidence-execution-v1",
            "scope": "per-window-optional-calls",
            "attemptedActionCount": self.attempted,
            "completedUncachedActionCount": sum(r["status"] == "completed" for r in self.records),
            "cacheHitCount": sum(r["status"] == "cache-hit" for r in self.records),
            "failedActionCount": sum(r["status"] == "failed" for r in self.records),
            "admittedEstimatedCostMs": self.estimated_ms,
            "elapsedMs": self.elapsed_ms,
            "hardDeadlineEnforced": False,
            "cacheAdmission": "one-action-and-conservative-estimate",
            "actions": [dict(row) for row in self.records],
        }
