from __future__ import annotations

from semantic_asr.document_deliberation_benchmark import (
    DocumentEvaluationCase,
    DocumentPromotionGate,
    apply_document_promotion_gate,
    evaluate_document_deliberation,
)


def cases():
    return (
        DocumentEvaluationCase(
            case_id="improved",
            reference="計画はまだ保留です。承認後に実行します。",
            first_pass="計画はまた保留です。保留です。承認後に実行します。",
            final="計画はまだ保留です。承認後に実行します。",
            final_status="accepted",
            first_pass_segments=("計画はまた保留です。", "保留です。承認後に実行します。"),
            final_segments=("計画はまだ保留です。", "承認後に実行します。"),
            critical_tokens=("まだ", "実行"),
            changed_window_count=1,
        ),
        DocumentEvaluationCase(
            case_id="retained-exact",
            reference="三千円です。",
            first_pass="三千円です。",
            final="三千円です。",
            final_status="accepted",
            critical_tokens=("三千円",),
            changed_window_count=0,
        ),
        DocumentEvaluationCase(
            case_id="provisional-retained",
            reference="今日は晴れです。",
            first_pass="今日は晴れです。",
            final="今日は晴れです。",
            final_status="first-pass",
            critical_tokens=("晴れ",),
            changed_window_count=0,
        ),
    )


def test_report_separates_correction_and_overlap_effects() -> None:
    report = evaluate_document_deliberation(
        cases(),
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert report.final_corpus_cer < report.first_corpus_cer
    assert report.improved_case_rate > 0.0
    assert report.regressed_case_rate == 0.0
    assert report.false_correction_rate_on_first_exact == 0.0
    assert report.critical_error_delta < 0.0
    assert report.overlap_duplicate_reduction == 1
    assert report.digest


def test_gate_fails_closed_on_insufficient_evidence() -> None:
    report = evaluate_document_deliberation(
        cases(),
        bootstrap_samples=100,
    )
    gate = DocumentPromotionGate(
        minimum_cases=100,
        minimum_reference_characters=10_000,
    )

    decision = apply_document_promotion_gate(report, gate)

    assert not decision.passed
    assert "insufficient-case-count" in decision.reasons
    assert "insufficient-reference-characters" in decision.reasons


def test_gate_detects_false_correction_of_already_correct_text() -> None:
    rows = (*cases(),)
    corrupted = DocumentEvaluationCase(
        case_id="false-correction",
        reference="三千円です。",
        first_pass="三千円です。",
        final="三万円です。",
        final_status="accepted",
        critical_tokens=("三千円",),
        changed_window_count=1,
    )
    report = evaluate_document_deliberation(
        (*rows, corrupted),
        bootstrap_samples=100,
    )
    gate = DocumentPromotionGate(
        minimum_cases=1,
        minimum_reference_characters=1,
        minimum_accepted_coverage=0.0,
        maximum_false_correction_rate=0.0,
        maximum_regressed_case_rate=1.0,
        maximum_critical_error_delta=1.0,
        require_cer_delta_upper_below=1.0,
    )

    decision = apply_document_promotion_gate(report, gate)

    assert not decision.passed
    assert "false-correction-rate-regression" in decision.reasons
