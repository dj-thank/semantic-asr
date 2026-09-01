#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    content = target.read_text(encoding="utf-8")
    if old not in content:
        if new in content:
            return
        raise RuntimeError(f"expected review target was not found in {path}")
    target.write_text(content.replace(old, new, 1), encoding="utf-8")


def _replace_between(path: str, start: str, end: str, replacement: str) -> None:
    target = ROOT / path
    content = target.read_text(encoding="utf-8")
    if replacement in content:
        return
    start_index = content.find(start)
    end_index = content.find(end, start_index)
    if start_index < 0 or end_index < 0:
        raise RuntimeError(f"review range was not found in {path}")
    target.write_text(
        content[:start_index] + replacement + content[end_index:],
        encoding="utf-8",
    )


def patch_mbr() -> None:
    _replace(
        "src/semantic_asr/mbr.py",
        "import math\nimport unicodedata\n",
        "import math\nimport re\nimport unicodedata\n",
    )
    _replace(
        "src/semantic_asr/mbr.py",
        """from .evaluation import (
    critical_entity_sequence,
    date_time_error_rate,
    filler_sequence,
    negation_error_rate,
    number_error_rate,
)
""",
        """from .evaluation import (
    DATE_TIME_PATTERN,
    NEGATION_PATTERN,
    NUMBER_PATTERN,
    critical_entity_sequence,
    filler_sequence,
)
""",
    )
    _replace(
        "src/semantic_asr/mbr.py",
        """def _optional_rate(value: float | None) -> float:
    return 0.0 if value is None else min(1.0, max(0.0, float(value)))


""",
        """def _pattern_rate(
    pattern: re.Pattern[str], reference: str, hypothesis: str
) -> float:
    normalized_reference = unicodedata.normalize("NFKC", reference)
    normalized_hypothesis = unicodedata.normalize("NFKC", hypothesis)
    return _rate(
        tuple(pattern.findall(normalized_reference)),
        tuple(pattern.findall(normalized_hypothesis)),
    )


""",
    )
    _replace(
        "src/semantic_asr/mbr.py",
        """    number = _optional_rate(number_error_rate(truth_text, hypothesis_text))
    date_time = _optional_rate(date_time_error_rate(truth_text, hypothesis_text))
    negation = _optional_rate(negation_error_rate(truth_text, hypothesis_text))
""",
        """    number = _pattern_rate(NUMBER_PATTERN, truth_text, hypothesis_text)
    date_time = _pattern_rate(DATE_TIME_PATTERN, truth_text, hypothesis_text)
    negation = _pattern_rate(NEGATION_PATTERN, truth_text, hypothesis_text)
""",
    )


def patch_risk_control() -> None:
    _replace(
        "src/semantic_asr/risk_control.py",
        """    correction_count = len(policies)
    bounds: list[PolicyRiskBound] = []
""",
        """    missing_outcomes = [
        policy.policy_id for policy in policies if not by_policy[policy.policy_id]
    ]
    if missing_outcomes:
        raise ValueError(
            "every policy requires held-out outcomes before risk control: "
            + ", ".join(sorted(missing_outcomes))
        )

    correction_count = len(policies)
    bounds: list[PolicyRiskBound] = []
""",
    )
    _replace(
        "src/semantic_asr/risk_control.py",
        """        empirical = (
            sum(row.bounded_loss for row in rows) / samples if samples else 1.0
        )
        mean_cost = (
            sum(row.measured_cost_ms for row in rows) / samples if samples else math.inf
        )
""",
        """        empirical = sum(row.bounded_loss for row in rows) / samples
        mean_cost = sum(row.measured_cost_ms for row in rows) / samples
""",
    )


def patch_model_io() -> None:
    _replace(
        "src/semantic_asr/model_io.py",
        """    model = ConstrainedLinearReranker(
        schema=schema,
        normalizer=normalizer,
        weights=weights,
        bias=float(payload.get("bias")),
        objective=str(payload.get("objective")),  # type: ignore[arg-type]
        training_digest=str(payload.get("trainingDigest") or ""),
""",
        """    objective = str(payload.get("objective") or "")
    if objective not in {"pairwise", "listwise", "hybrid"}:
        raise ValueError("unsupported constrained reranker objective")
    model = ConstrainedLinearReranker(
        schema=schema,
        normalizer=normalizer,
        weights=weights,
        bias=float(payload.get("bias")),
        objective=objective,  # type: ignore[arg-type]
        training_digest=str(payload.get("trainingDigest") or ""),
""",
    )


def patch_planner() -> None:
    start = "    selected: list[ActionPrediction] = []\n"
    end = "    if selected and used >= cost_budget_ms:\n"
    replacement = """    selected: list[ActionPrediction] = []
    rejected: list[ActionPrediction] = []
    selected_ids: set[str] = set()
    exclusive_groups: set[str] = set()
    used = 0.0
    remaining = list(predictions)
    while remaining and len(selected) < maximum_actions:
        made_progress = False
        next_remaining: list[ActionPrediction] = []
        for index, prediction in enumerate(remaining):
            action = prediction.action
            if not set(action.dependencies).issubset(selected_ids):
                next_remaining.append(prediction)
                continue
            if action.exclusive_group and action.exclusive_group in exclusive_groups:
                rejected.append(prediction)
                continue
            if prediction.expected_gain < minimum_expected_gain and not action.mandatory:
                rejected.append(prediction)
                continue
            if used + prediction.expected_cost_ms > cost_budget_ms:
                rejected.append(prediction)
                continue
            selected.append(prediction)
            selected_ids.add(action.action_id)
            if action.exclusive_group:
                exclusive_groups.add(action.exclusive_group)
            used += prediction.expected_cost_ms
            made_progress = True
            if len(selected) >= maximum_actions:
                next_remaining.extend(remaining[index + 1 :])
                break
        if not made_progress:
            rejected.extend(next_remaining)
            remaining = []
            break
        remaining = next_remaining
    rejected.extend(remaining)
"""
    _replace_between(
        "src/semantic_asr/planner_v2.py",
        start,
        end,
        replacement,
    )


def patch_experiment_duplicates() -> None:
    _replace(
        "src/semantic_asr/experiment.py",
        """    by_system: dict[str, dict[str, SampleResult]] = {}
    for result in results:
        by_system.setdefault(result.system_id, {})[result.sample_id] = result
""",
        """    by_system: dict[str, dict[str, SampleResult]] = {}
    seen_results: set[tuple[str, str]] = set()
    for result in results:
        key = (result.system_id, result.sample_id)
        if key in seen_results:
            raise ValueError(
                f"duplicate benchmark result for system/sample: {key[0]}/{key[1]}"
            )
        seen_results.add(key)
        by_system.setdefault(result.system_id, {})[result.sample_id] = result
""",
    )


def main() -> int:
    patch_mbr()
    patch_risk_control()
    patch_model_io()
    patch_planner()
    patch_experiment_duplicates()
    print("Applied Semantic ASR v0.2 post-review corrections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
