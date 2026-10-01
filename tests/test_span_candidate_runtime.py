"""Canonical integration tests; adapters/evidence are synthetic, not audio accuracy."""

import hashlib
from dataclasses import asdict, replace

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.contracts import CandidateEvidence, sha256_json
from semantic_asr.deliberation_evidence import BoundedUtility
from semantic_asr.semantic_deliberation import VerifiedSpanProposal
from semantic_asr.span_candidate_runtime import (
    build_verified_redecode_lattice,
    run_span_redecode,
    span_verification_request,
)
from semantic_asr.span_candidates import SpanAnchor, SpanBudget, SpanWindow


@pytest.fixture
def setup(tmp_path):
    # No audio decoder reads these bytes: this fixture tests identity and contracts.
    path = tmp_path / "synthetic-recording.bin"
    path.write_bytes(b"synthetic-source-identity-only")
    pivot = CandidateEvidence(candidate_id="first", text="えー、歯垢原品を売る、いや売らない。")
    window = SpanWindow(
        pivot.candidate_id, pivot.text, hashlib.sha256(path.read_bytes()).hexdigest(), 1000, 9000
    )
    target = SpanAnchor(window.digest, 3, 7, 2000, 3000, "a" * 64)
    request = DecodeRequest(
        str(path), start_ms=1000, end_ms=9000, initial_prompt="DO NOT COPY", hotwords=("secret",)
    )

    class Adapter:
        def __init__(self):
            self.calls = []

        def decode(self, received):
            self.calls.append(received)
            return [
                CandidateEvidence(
                    candidate_id="crop", text="手工芸品", acoustic=-0.01, token_ids=(1, 2)
                )
            ]

    return path, pivot, window, target, request, Adapter()


def run(setup, **kwargs):
    _, _, window, target, request, adapter = setup
    return run_span_redecode(
        window,
        [target],
        request=request,
        adapter=adapter,
        decoder_fingerprint="b" * 64,
        enabled=True,
        **kwargs,
    )


def receipts(expansion):
    payload = span_verification_request(expansion)
    return tuple(
        VerifiedSpanProposal(
            proposal_id=row["candidateId"],
            text=row["text"],
            source_audio_sha256=expansion.window.source_audio_sha256,
            utilities=(
                BoundedUtility(
                    channel="phone",
                    value=0.1,
                    source="synthetic-test-only",
                    profile_digest="c" * 64,
                    input_digest=sha256_json(row),
                ),
            ),
            metadata={
                "spanVerificationRequestSha256": sha256_json(payload),
                "verificationScope": "whole-window",
            },
        )
        for row in payload["candidates"]
    )


def build(setup, expansion, verified):
    return build_verified_redecode_lattice(
        expansion, pivot=setup[1], verified=verified, document_id="synthetic-document"
    )


def test_real_request_contract_crops_exact_span_clears_hints_and_caps_hypotheses(setup):
    before = asdict(setup[1])
    result = run(setup, budget=SpanBudget(max_hypotheses_per_span=2))
    received = setup[-1].calls[0]
    assert isinstance(received, DecodeRequest)
    assert (received.start_ms, received.end_ms) == (2000, 3000)
    assert received.hypotheses == 2
    assert received.initial_prompt is None
    assert received.hotwords == ()
    assert setup[4].initial_prompt == "DO NOT COPY"
    assert asdict(setup[1]) == before
    assert result.drafts[0].text == "えー、手工芸品を売る、いや売らない。"
    assert not hasattr(result.drafts[0], "acoustic")
    assert not hasattr(result.drafts[0], "token_ids")


def test_disabled_runtime_does_not_read_files_or_touch_adapter(setup):
    path, _, window, target, request, adapter = setup
    path.unlink()
    result = run_span_redecode(
        window, [target], request=request, adapter=adapter, decoder_fingerprint="b" * 64
    )
    assert not result.enabled
    assert adapter.calls == []


def test_runtime_rejects_source_mismatch_before_decoder_call(setup):
    setup[0].write_bytes(b"wrong recording")
    with pytest.raises(ValueError, match="source recording"):
        run(setup)
    assert setup[-1].calls == []


def test_runtime_rejects_source_changed_by_adapter(setup):
    original = setup[-1].decode

    def mutate(request):
        setup[0].write_bytes(b"changed during decode")
        return original(request)

    setup[-1].decode = mutate
    with pytest.raises(ValueError, match="changed during"):
        run(setup)


@pytest.mark.parametrize("change", [{"start_ms": 0}, {"end_ms": 8000}, {"start_ms": None}])
def test_runtime_rejects_wrong_or_implicit_parent_window(setup, change):
    altered = (*setup[:4], replace(setup[4], **change), setup[-1])
    with pytest.raises(ValueError, match="explicitly match"):
        run(altered)
    assert setup[-1].calls == []


def test_runtime_records_noncanonical_adapter_output_as_failure(setup):
    setup[-1].decode = lambda _: ["not a CandidateEvidence"]
    result = run(setup)
    assert not result.drafts
    assert result.attempts[0].status == "error"
    assert result.attempts[0].error_type == "TypeError"


def test_verification_input_has_no_prompt_reference_path_or_crop_score(setup):
    result = run(setup)
    payload = span_verification_request(result)
    assert (payload["startMs"], payload["endMs"]) == (1000, 9000)
    assert payload["sourceAudioSha256"] == setup[2].source_audio_sha256
    serialized = str(payload)
    for forbidden in ("DO NOT COPY", "secret", str(setup[0]), "acoustic", "reference"):
        assert forbidden not in serialized


def test_independently_verified_new_text_reaches_existing_lattice(setup):
    expansion = run(setup)
    result = build(setup, expansion, receipts(expansion))
    result.verify()
    assert len(result.lattice.spans) == 1
    span = result.lattice.spans[0]
    assert span.retained_arc.text == setup[1].text
    assert {arc.text for arc in span.arcs} == set(expansion.texts)
    generated = next(arc for arc in span.arcs if arc.text != setup[1].text)
    assert generated.origin == "phonetic-proposal"
    assert generated.source_candidate_ids == ()
    assert result.projections[0].text == setup[1].text
    assert len(result.lattice.source_paths) == 1


def test_abstain_keeps_original_and_does_not_insert_unverified_text(setup):
    expansion = run(setup)
    result = build(setup, expansion, ())
    assert len(result.lattice.spans[0].arcs) == 1
    assert result.lattice.spans[0].retained_arc.text == setup[1].text


def test_retained_text_must_also_have_comparable_independent_evidence(setup):
    expansion = run(setup)
    with pytest.raises(ValueError, match="retained text"):
        build(setup, expansion, receipts(expansion)[1:])


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposal_id", "unknown"),
        ("text", "a fluent hallucination"),
        ("source_audio_sha256", "e" * 64),
        ("source_candidate_ids", ("first",)),
        ("observed_eligible", False),
        ("origin", "context-proposal"),
    ],
)
def test_lattice_rejects_foreign_or_unverified_proposal(setup, field, value):
    expansion = run(setup)
    original, changed = receipts(expansion)
    with pytest.raises(ValueError):
        build(setup, expansion, (original, replace(changed, **{field: value})))


@pytest.mark.parametrize(
    "change",
    [
        {"spanVerificationRequestSha256": "f" * 64},
        {"verificationScope": "crop-only"},
    ],
)
def test_wrong_pool_or_crop_only_verification_cannot_be_relabelled(setup, change):
    expansion = run(setup)
    original, changed = receipts(expansion)
    bad = replace(changed, metadata={**changed.metadata, **change})
    with pytest.raises(ValueError):
        build(setup, expansion, (original, bad))


def test_phone_and_derived_mora_cannot_be_double_votes(setup):
    expansion = run(setup)
    original, changed = receipts(expansion)
    bad = replace(
        changed, utilities=(*changed.utilities, replace(changed.utilities[0], channel="mora"))
    )
    with pytest.raises(ValueError, match="one independent channel"):
        build(setup, expansion, (original, bad))


@pytest.mark.parametrize(
    "change",
    [
        {"source": "another-model"},
        {"profile_digest": "e" * 64},
        {"channel": "mora"},
    ],
)
def test_incomparable_verification_domains_are_rejected(setup, change):
    expansion = run(setup)
    original, changed = receipts(expansion)
    bad = replace(changed, utilities=(replace(changed.utilities[0], **change),))
    with pytest.raises(ValueError, match="must match"):
        build(setup, expansion, (original, bad))


def test_duplicate_receipt_rejected(setup):
    expansion = run(setup)
    original, changed = receipts(expansion)
    with pytest.raises(ValueError, match="duplicate"):
        build(setup, expansion, (original, changed, changed))


def test_changed_parent_cannot_reuse_a_previous_expansion(setup):
    expansion = run(setup)
    with pytest.raises(ValueError, match="immutable expansion parent"):
        build_verified_redecode_lattice(
            expansion,
            pivot=replace(setup[1], text="a different parent"),
            verified=receipts(expansion),
            document_id="document",
        )
