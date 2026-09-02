from itertools import product

import pytest

from semantic_asr.evaluation import (
    best_contiguous_alignment,
    cer,
    critical_entity_error_rate,
    currency_error_rate,
    date_time_error_rate,
    disfluency_preservation_rate,
    edit_distance,
    evaluate_confidence,
    evaluate_transcript,
    negation_error_rate,
    number_error_rate,
    oracle_cer,
    punctuation_f1,
    unsupported_correction_rate,
)


def test_cer_uses_reference_character_count() -> None:
    assert cer("今日は3人です", "今日は2人です") == pytest.approx(1 / 7)


def test_best_contiguous_alignment_reports_boundary_overrun_without_changing_text() -> None:
    diagnostic = best_contiguous_alignment("中央区です", "前置き中央区です後")
    assert diagnostic is not None
    assert diagnostic.edits == 0
    assert diagnostic.prefix_overrun_characters == 3
    assert diagnostic.suffix_overrun_characters == 1
    assert diagnostic.retained_characters == 5
    assert diagnostic.aligned_cer == 0.0


def test_best_contiguous_alignment_is_undefined_for_an_empty_reference() -> None:
    assert best_contiguous_alignment(" \t", "文字") is None


def test_best_contiguous_alignment_matches_bruteforce_for_short_sequences() -> None:
    alphabet = "ab"
    for reference_size in range(1, 4):
        for hypothesis_size in range(4):
            for reference_items in product(alphabet, repeat=reference_size):
                reference = "".join(reference_items)
                for hypothesis_items in product(alphabet, repeat=hypothesis_size):
                    hypothesis = "".join(hypothesis_items)
                    expected = min(
                        (
                            edit_distance(reference, hypothesis[start:end]),
                            -(end - start),
                            start,
                            end,
                        )
                        for start in range(len(hypothesis) + 1)
                        for end in range(start, len(hypothesis) + 1)
                    )
                    actual = best_contiguous_alignment(reference, hypothesis)
                    assert actual is not None
                    assert (
                        actual.edits,
                        -actual.retained_characters,
                        actual.retained_start,
                        actual.retained_end,
                    ) == expected


def test_semantic_critical_metrics() -> None:
    assert number_error_rate("3人で1000円", "2人で1000円") == pytest.approx(0.5)
    assert date_time_error_rate("2026/08/29 10:30", "2026/08/28 10:30") > 0
    assert currency_error_rate("料金は3000円", "料金は30000円") == 1
    assert negation_error_rate("明日は行きません", "明日は行きます") == 1
    assert critical_entity_error_rate("Qwen3.8を東京で使う", "Qwen3.7を京都で使う") > 0


def test_disfluency_and_unsupported_corrections() -> None:
    reference = "えっと今日は、あの、学校へ"
    assert disfluency_preservation_rate(reference, "今日は学校へ") == 0
    assert disfluency_preservation_rate(reference, reference) == 1
    assert unsupported_correction_rate("学校を行きました", "学校に行きました") > 0
    assert unsupported_correction_rate("学校を行きました", "学校を行きました") == 0


def test_punctuation_and_oracle_metrics() -> None:
    assert punctuation_f1("はい、行きます。", "はい、行きます。") == 1
    assert punctuation_f1("はい、行きます。", "はい行きます") == 0
    assert oracle_cer("今日は三人です", ["今日は二人です", "今日は三人です"]) == 0


def test_confidence_metrics_and_full_result() -> None:
    metrics = evaluate_confidence([0.95, 0.8, 0.45, 0.2], [1, 1, 0, 0])
    assert 0 <= metrics.expected_calibration_error <= 1
    assert 0 <= metrics.brier <= 1
    assert metrics.negative_log_likelihood >= 0
    assert metrics.aurc >= 0
    result = evaluate_transcript(
        reference="明日は行きません。料金は3000円です。",
        observed="明日は行きます。料金は30000円です。",
        normalized="明日は行きます。料金は30000円です。",
    )
    assert result.negation_error_rate == 1
    assert result.currency_error_rate == 1
