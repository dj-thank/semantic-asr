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
    exact_cer,
    filler_event_score,
    negation_error_rate,
    number_error_rate,
    oracle_cer,
    punctuation_f1,
    reference_annotation_counts,
    spoken_reference_surface,
    unsupported_correction_rate,
)


def test_cer_uses_reference_character_count() -> None:
    assert cer("今日は3人です", "今日は2人です") == pytest.approx(1 / 7)


def test_exact_cer_is_undefined_for_an_empty_reference() -> None:
    assert exact_cer(" ", "") is None


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


def test_spoken_reference_surface_preserves_fillers_and_repairs() -> None:
    annotated = "(F えー) (D 東京に)大阪へ行きます(? 舞い)(?)"
    assert spoken_reference_surface(annotated) == "えー 東京に大阪へ行きます舞い"
    counts = reference_annotation_counts(annotated + "[PERSON_01]")
    assert counts.filler_events == 1
    assert counts.disfluency_events == 1
    assert counts.uncertain_spans == 2
    assert counts.masked_spans == 1


def test_filler_event_score_matches_variant_families_without_deleting_them() -> None:
    score = filler_event_score("(F えー)そうですね(F あのー)", "えっとそうですね")
    assert score is not None
    assert score.expected_events == 2
    assert score.observed_events == 1
    assert score.matched_events == 1
    assert score.precision == 1.0
    assert score.recall == 0.5
    assert score.f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize("filler", ("ええと", "その", "まあ", "うーん", "んー"))
def test_filler_event_score_keeps_each_declared_family_distinct(filler: str) -> None:
    score = filler_event_score(f"(F {filler})", filler)
    assert score is not None
    assert score.expected_events == 1
    assert score.observed_events == 1
    assert score.matched_events == 1
    assert score.f1 == 1.0


def test_filler_event_score_does_not_match_different_declared_families() -> None:
    score = filler_event_score("(F その)", "まあ")
    assert score is not None
    assert score.matched_events == 0
    assert score.f1 == 0.0


def test_filler_event_score_is_multiplicity_aware_within_one_utterance() -> None:
    score = filler_event_score("(F その)(F その)", "そのその")
    assert score is not None
    assert score.expected_events == 2
    assert score.observed_events == 2
    assert score.matched_events == 2
    assert score.f1 == 1.0


def test_filler_event_score_matches_unknown_annotated_family_without_collapsing_it() -> None:
    score = filler_event_score("(F ほら)", "ほら")
    assert score is not None
    assert score.expected_events == 1
    assert score.observed_events == 1
    assert score.matched_events == 1
    assert score.f1 == 1.0


def test_full_evaluation_uses_spoken_surface_and_reports_filler_events() -> None:
    result = evaluate_transcript(
        reference="そうですね",
        annotated_reference="(F えー)そうですね",
        observed="えっとそうですね",
        normalized="えっとそうですね",
    )
    assert result.cer == pytest.approx(2 / 7)
    assert result.filler_events is not None
    assert result.filler_events.f1 == 1.0


def test_full_evaluation_refuses_exact_cer_for_uncertain_annotated_spans() -> None:
    result = evaluate_transcript(
        reference="明日は舞い上がる",
        annotated_reference="明日は(? 舞い)上がる",
        observed="明日は舞い上がる",
        normalized="明日は舞い上がる",
    )
    assert result.cer is None
    assert result.critical_entity_error_rate is None
    assert result.reference_annotations is not None
    assert result.reference_annotations.exact_cer_safe is False


@pytest.mark.parametrize(
    "annotated",
    ("明日は(?)上がる", "明日は[PERSON_01]上がる", "明日は[MASK]上がる"),
)
def test_full_evaluation_refuses_exact_cer_for_inaudible_or_masked_spans(
    annotated: str,
) -> None:
    result = evaluate_transcript(
        reference="明日は舞い上がる",
        annotated_reference=annotated,
        observed="明日は舞い上がる",
        normalized="明日は舞い上がる",
    )
    assert result.cer is None
    assert result.reference_annotations is not None
    assert result.reference_annotations.exact_cer_safe is False


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
