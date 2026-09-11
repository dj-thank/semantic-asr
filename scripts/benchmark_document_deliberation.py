#!/usr/bin/env python3
"""Evaluate paired first-pass and document-deliberated JSONL rows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from semantic_asr.document_deliberation_benchmark import (
    DocumentEvaluationCase,
    DocumentPromotionGate,
    apply_document_promotion_gate,
    evaluate_document_deliberation,
)


def load_cases(path: Path) -> tuple[DocumentEvaluationCase, ...]:
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                output.append(
                    DocumentEvaluationCase(
                        case_id=str(row["caseId"]),
                        reference=str(row["reference"]),
                        first_pass=str(row["firstPass"]),
                        final=str(row["final"]),
                        final_status=str(row.get("finalStatus", "first-pass")),
                        first_pass_segments=tuple(row.get("firstPassSegments", ())),
                        final_segments=tuple(row.get("finalSegments", ())),
                        critical_tokens=tuple(row.get("criticalTokens", ())),
                        changed_window_count=int(row.get("changedWindowCount", 0)),
                        metadata=dict(row.get("metadata", {})),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid evaluation row at line {line_number}: {exc}") from exc
    if not output:
        raise ValueError("evaluation JSONL contains no cases")
    return tuple(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument("--minimum-reference-characters", type=int, default=10_000)
    parser.add_argument("--minimum-accepted-coverage", type=float, default=0.20)
    parser.add_argument("--maximum-false-correction-rate", type=float, default=0.005)
    parser.add_argument("--maximum-regressed-case-rate", type=float, default=0.10)
    parser.add_argument("--maximum-critical-error-delta", type=float, default=0.0)
    parser.add_argument("--require-cer-delta-upper-below", type=float, default=0.0)
    args = parser.parse_args()

    cases = load_cases(args.input)
    report = evaluate_document_deliberation(
        cases,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    gate = DocumentPromotionGate(
        minimum_cases=args.minimum_cases,
        minimum_reference_characters=args.minimum_reference_characters,
        minimum_accepted_coverage=args.minimum_accepted_coverage,
        maximum_false_correction_rate=args.maximum_false_correction_rate,
        maximum_regressed_case_rate=args.maximum_regressed_case_rate,
        maximum_critical_error_delta=args.maximum_critical_error_delta,
        require_cer_delta_upper_below=args.require_cer_delta_upper_below,
    )
    decision = apply_document_promotion_gate(report, gate)
    payload = {
        "report": asdict(report),
        "reportDigest": report.digest,
        "gate": asdict(gate),
        "gateDigest": gate.digest,
        "promotion": asdict(decision),
        "promotionDigest": decision.digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
