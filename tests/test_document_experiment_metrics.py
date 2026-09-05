from __future__ import annotations

from _document_experiment_fixture import AUDIO, fake_paths

from semantic_asr.document_experiment.metrics import (
    CaseArmMetrics,
    aggregate_arm_metrics,
    paired_bootstrap_cer_delta,
    text_error_metrics,
    window_revision_metrics,
)
from semantic_asr.document_experiment.protocol import (
    CriticalReferenceToken,
    FrozenReference,
)


def reference() -> FrozenReference:
    windows = (
        "レビュー完了まではまだマージしません。",
        "承認後に統合します。",
    )
    return FrozenReference(
        reference_id="reference",
        source_audio_sha256=AUDIO,
        text="".join(windows),
        window_texts=windows,
        critical_tokens=(
            CriticalReferenceToken(kind="negation", text="ません"),
            CriticalReferenceToken(kind="entity", text="マージ"),
        ),
    )


def case_metrics(path_index: int, *, accepted: bool = True) -> CaseArmMetrics:
    paths = fake_paths()
    selected = paths[path_index]
    first_pass_windows = tuple(option.text for option in paths[0].options)
    return CaseArmMetrics(
        text=text_error_metrics(selected.text, reference()),
        windows=window_revision_metrics(
            first_pass_windows,
            selected,
            reference().window_texts,
        ),
        accepted=accepted,
        latency_ms=1.0,
        python_peak_bytes=100,
        scored_characters=10,
        scorer_calls=1,
    )


def test_corrected_path_improves_cer_and_one_window() -> None:
    baseline = case_metrics(0)
    corrected = case_metrics(1)

    assert corrected.text.strict_edits < baseline.text.strict_edits
    assert corrected.windows.improved_windows == 1
    assert corrected.windows.worsened_windows == 0
    assert corrected.windows.corrected_characters == 1


def test_harmful_path_counts_false_correction_on_previously_correct_window() -> None:
    harmful = case_metrics(2)

    assert harmful.windows.worsened_windows == 1
    assert harmful.windows.false_correction_windows == 1
    assert harmful.windows.introduced_error_characters > 0


def test_aggregate_exposes_coverage_risk_and_revision_rate() -> None:
    rows = (case_metrics(1, accepted=True), case_metrics(2, accepted=False))

    aggregate = aggregate_arm_metrics("ordered", rows)

    assert aggregate.case_count == 2
    assert aggregate.accepted_case_count == 1
    assert aggregate.coverage == 0.5
    assert aggregate.accepted_strict_cer is not None
    assert aggregate.revision_rate > 0.0


def test_paired_bootstrap_is_deterministic_and_directional() -> None:
    baseline = (case_metrics(0), case_metrics(0))
    improved = (case_metrics(1), case_metrics(1))

    first = paired_bootstrap_cer_delta(
        "improved",
        "baseline",
        improved,
        baseline,
        resamples=200,
        seed=7,
    )
    second = paired_bootstrap_cer_delta(
        "improved",
        "baseline",
        improved,
        baseline,
        resamples=200,
        seed=7,
    )

    assert first == second
    assert first.point_delta < 0.0
    assert first.upper < 0.0
