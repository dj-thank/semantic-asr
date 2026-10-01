from dataclasses import FrozenInstanceError, replace
from itertools import repeat

import pytest

from semantic_asr.span_candidates import (
    SpanAnchor,
    SpanBudget,
    SpanDecode,
    SpanDraft,
    SpanExpansion,
    SpanWindow,
    generate_span_drafts,
)

AUDIO = "a" * 64
ALIGNMENT = "b" * 64
DECODER = "c" * 64
EVIDENCE = "d" * 64


def window(text="えー、歯垢原品、歯垢原品を売る、いや売らない。"):
    return SpanWindow("first", text, AUDIO, 10_000, 18_000)


def anchor(parent, start=3, end=7, audio_start=11_000, audio_end=12_000):
    return SpanAnchor(parent.digest, start, end, audio_start, audio_end, ALIGNMENT)


def generate(parent, anchors, texts=("手工芸品",), **kwargs):
    return generate_span_drafts(
        parent,
        anchors,
        lambda _: (SpanDecode(text, EVIDENCE) for text in texts),
        decoder_fingerprint=DECODER,
        enabled=True,
        **kwargs,
    )


def test_missing_correct_candidate_is_generated_without_rewriting_other_speech():
    parent = window()
    result = generate(parent, [anchor(parent)])
    assert result.texts == (
        parent.text,
        "えー、手工芸品、歯垢原品を売る、いや売らない。",
    )
    assert result.drafts[0].text[:3] == parent.text[:3]
    assert result.drafts[0].text[7:] == parent.text[7:]
    assert parent.text == "えー、歯垢原品、歯垢原品を売る、いや売らない。"
    assert result.decoder_calls == 1
    assert result.requested_audio_ms == 1000


def test_second_identical_word_uses_offsets_not_find_or_global_replace():
    parent = window()
    result = generate(parent, [anchor(parent, 8, 12, 13_000, 14_000)])
    assert result.drafts[0].text == "えー、歯垢原品、手工芸品を売る、いや売らない。"


@pytest.mark.parametrize(
    "replacement", ["しゅこうげいひん", "工芸品", " 手工芸品 ", "手工芸品、手工芸品"]
)
def test_replacement_length_and_surface_are_not_normalized(replacement):
    parent = window()
    result = generate(parent, [anchor(parent)], texts=[replacement])
    assert result.drafts[0].text == parent.text[:3] + replacement + parent.text[7:]


def test_unicode_codepoint_offsets_preserve_emoji_combining_marks_and_whitespace():
    parent = window("🙂 ば 歯垢原品\nうん、うん")
    start = parent.text.index("歯垢原品")
    result = generate(parent, [anchor(parent, start, start + 4)])
    assert result.drafts[0].text == "🙂 ば 手工芸品\nうん、うん"


def test_drafts_have_no_inherited_scores_or_observed_eligibility():
    parent = window()
    draft = generate(parent, [anchor(parent)]).drafts[0]
    assert draft.observed_eligible is False
    for name in ("acoustic", "avg_logprob", "sequence_score", "token_ids", "probability"):
        assert not hasattr(draft, name)
    with pytest.raises(FrozenInstanceError):
        draft.replacement = "different"


def test_default_off_never_calls_or_consumes_anchors():
    def fail(*_):
        raise AssertionError("must not execute")

    result = generate_span_drafts(window(), None, fail, decoder_fingerprint="not needed")
    assert not result.enabled
    assert result.texts == (window().text,)
    assert result.decoder_calls == result.requested_audio_ms == 0


@pytest.mark.parametrize("enabled", [1, 0, None, "false"])
def test_enabled_must_be_boolean(enabled):
    with pytest.raises(TypeError):
        generate_span_drafts(
            window(), [], lambda _: [], decoder_fingerprint=DECODER, enabled=enabled
        )


@pytest.mark.parametrize(
    "field",
    [
        "max_spans",
        "max_hypotheses_per_span",
        "max_candidates",
        "max_audio_ms",
        "max_replacement_chars",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_strict_positive_budgets(field, bad):
    with pytest.raises((TypeError, ValueError)):
        SpanBudget(**{field: bad})


@pytest.mark.parametrize("field", ["char_start", "char_end", "start_ms", "end_ms"])
@pytest.mark.parametrize("bad", [True, 1.5, "1"])
def test_anchor_offsets_must_be_integers(field, bad):
    original = anchor(window())
    with pytest.raises(TypeError):
        replace(original, **{field: bad})


@pytest.mark.parametrize(
    "change",
    [
        {"text": "other text"},
        {"source_audio_sha256": "e" * 64},
        {"start_ms": 9999},
        {"end_ms": 18_001},
        {"candidate_id": "another"},
    ],
)
def test_anchor_is_bound_to_parent_text_audio_times_and_id(change):
    parent = window()
    with pytest.raises(ValueError, match="different parent"):
        anchor(parent).validate(replace(parent, **change))


@pytest.mark.parametrize("change", [{"char_end": 999}, {"start_ms": 9000}, {"end_ms": 19_000}])
def test_out_of_bounds_anchor_rejected_before_any_decode(change):
    parent = window()
    calls = []
    with pytest.raises(ValueError):
        generate_span_drafts(
            parent,
            [anchor(parent), replace(anchor(parent), **change)],
            lambda item: calls.append(item) or [],
            decoder_fingerprint=DECODER,
            enabled=True,
        )
    assert calls == []


@pytest.mark.parametrize("bad", ["", " ", "a" * 63, "G" * 64, "A" * 64, None])
def test_alignment_requires_explicit_digest(bad):
    with pytest.raises((TypeError, ValueError)):
        replace(anchor(window()), alignment_sha256=bad)


def test_empty_noop_duplicate_and_overlong_outputs_do_not_create_candidates():
    parent = window()
    result = generate(
        parent,
        [anchor(parent)],
        texts=["", " ", "歯垢原品", "手工芸品", "手工芸品", "x" * 257],
        budget=SpanBudget(max_hypotheses_per_span=6),
    )
    assert len(result.drafts) == 1
    assert result.attempts[0].decoded_hypotheses == 6


def test_duplicate_anchors_do_not_spend_more_audio_or_count_as_votes():
    parent = window()
    result = generate(parent, [anchor(parent), anchor(parent)])
    assert result.decoder_calls == 1
    assert result.requested_audio_ms == 1000
    assert result.attempts[-1].status == "duplicate-anchor"


def test_overlapping_targets_are_independent_not_sequential_replacements():
    parent = window("ABCDE")
    result = generate(
        parent, [anchor(parent, 1, 3), anchor(parent, 2, 4, 12_000, 13_000)], texts=["X"]
    )
    assert result.texts == ("ABCDE", "AXDE", "ABXE")


def test_infinite_decoder_output_is_bounded_without_peeking():
    parent = window()
    counts = []

    def decode(_):
        for value in repeat("歯垢原品"):
            counts.append(value)
            yield SpanDecode(value, EVIDENCE)

    result = generate_span_drafts(
        parent, [anchor(parent)], decode, decoder_fingerprint=DECODER, enabled=True
    )
    assert len(counts) == 5
    assert result.drafts == ()


def test_candidate_cap_stops_consumption_and_remaining_calls():
    parent = window()
    consumed = []

    def decode(_):
        for text in ("new one", "new two", "new three"):
            consumed.append(text)
            yield SpanDecode(text, EVIDENCE)

    result = generate_span_drafts(
        parent,
        [anchor(parent), anchor(parent, 8, 12)],
        decode,
        decoder_fingerprint=DECODER,
        budget=SpanBudget(max_candidates=1),
        enabled=True,
    )
    assert consumed == ["new one"]
    assert result.decoder_calls == 1
    assert result.attempts[-1].status == "candidate-budget"


def test_span_budget_and_audio_budget_are_separate():
    parent = window()
    anchors = [anchor(parent), anchor(parent, 8, 12, 12_000, 14_000)]
    capped_calls = generate(parent, anchors, budget=SpanBudget(max_spans=1))
    capped_audio = generate(parent, anchors, budget=SpanBudget(max_audio_ms=2500))
    assert capped_calls.attempts[-1].status == "span-budget"
    assert capped_audio.attempts[-1].status == "audio-budget"
    assert capped_calls.decoder_calls == capped_audio.decoder_calls == 1


def test_oversized_anchor_does_not_prevent_a_later_small_anchor():
    parent = window()
    anchors = [anchor(parent, 3, 7, 10_000, 14_000), anchor(parent, 8, 12, 14_000, 15_000)]
    result = generate(parent, anchors, budget=SpanBudget(max_audio_ms=1000))
    assert result.attempts[0].status == "audio-budget"
    assert result.attempts[1].status == "decoded"
    assert result.requested_audio_ms == 1000


def test_failure_discards_partial_batch_and_counts_spent_call_and_audio():
    parent = window()

    def broken(_):
        yield SpanDecode("手工芸品", EVIDENCE)
        raise RuntimeError("private path and confidential text")

    result = generate_span_drafts(
        parent, [anchor(parent)], broken, decoder_fingerprint=DECODER, enabled=True
    )
    assert result.drafts == ()
    assert result.decoder_calls == 1
    assert result.requested_audio_ms == 1000
    assert result.attempts[0].error_type == "RuntimeError"
    assert "private" not in repr(result)


def test_invalid_decoder_record_is_a_failed_attempt_not_success():
    parent = window()
    result = generate_span_drafts(
        parent,
        [anchor(parent)],
        lambda _: ["untyped text"],
        decoder_fingerprint=DECODER,
        enabled=True,
    )
    assert not result.drafts
    assert result.attempts[0].status == "error"
    assert result.attempts[0].error_type == "TypeError"


def test_failure_does_not_stop_independent_later_target():
    parent = window()

    def decode(item):
        if item.char_start == 3:
            raise RuntimeError("failure")
        return [SpanDecode("手工芸品", EVIDENCE)]

    result = generate_span_drafts(
        parent,
        [anchor(parent), anchor(parent, 8, 12)],
        decode,
        decoder_fingerprint=DECODER,
        enabled=True,
    )
    assert [row.status for row in result.attempts] == ["error", "decoded"]
    assert len(result.drafts) == 1


def test_interrupt_is_not_swallowed_as_a_decoder_failure():
    def interrupt(_):
        raise KeyboardInterrupt

    parent = window()
    with pytest.raises(KeyboardInterrupt):
        generate_span_drafts(
            parent, [anchor(parent)], interrupt, decoder_fingerprint=DECODER, enabled=True
        )


def test_digest_is_stable_and_binds_generation_evidence_and_alignment():
    parent = window()
    first = generate(parent, [anchor(parent)])
    second = generate(parent, [anchor(parent)])
    assert first.digest == second.digest
    draft = first.drafts[0]
    assert replace(draft, decode_evidence_sha256="e" * 64).digest != draft.digest
    assert replace(draft, decoder_fingerprint="e" * 64).digest != draft.digest
    changed_alignment = replace(draft.anchor, alignment_sha256="e" * 64)
    assert replace(draft, anchor=changed_alignment).digest != draft.digest


def test_expansion_rejects_disabled_work_foreign_drafts_and_duplicate_surfaces():
    parent = window()
    draft = generate(parent, [anchor(parent)]).drafts[0]
    with pytest.raises(ValueError):
        SpanExpansion(parent, drafts=(draft,))
    with pytest.raises(ValueError):
        SpanExpansion(replace(parent, text="another"), drafts=(draft,), enabled=True)
    with pytest.raises(ValueError):
        SpanExpansion(parent, drafts=(draft, draft), enabled=True)


def test_draft_rejects_empty_or_identical_replacement():
    parent = window()
    for text in ("", " ", "歯垢原品"):
        with pytest.raises(ValueError):
            SpanDraft(parent, anchor(parent), text, DECODER, EVIDENCE)


def test_expansion_snapshots_collections_and_rejects_disabled_audio_work():
    parent = window()
    draft = generate(parent, [anchor(parent)]).drafts[0]
    supplied = [draft]
    result = SpanExpansion(parent, drafts=supplied, enabled=True)
    supplied.clear()
    assert result.drafts == (draft,)
    with pytest.raises(ValueError):
        SpanExpansion(parent, requested_audio_ms=1000)
