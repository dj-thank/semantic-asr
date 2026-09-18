"""Contract tests use synthetic arrays, not real-audio accuracy measurements."""

import copy
import importlib.util
import itertools
import json
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "jev_shadow", Path(__file__).parents[1] / "scripts" / "run_jev_phonetic_shadow.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def record():
    return {
        "id": "validation-000",
        "baseline_id": "orig-a",
        "baseline_text": "食べれる",
        "phone_greedy": ["t", "a", "b", "e", "r", "e", "r", "u"],
        "reference": "NEVER_SEND_REFERENCE",
        "reference_phones": ["GOLD_SENTINEL"],
        "reference_kana": "NEVER_SEND_KANA",
        "context": "NEVER_SEND_CONTEXT",
        "candidates": [
            {
                "id": "orig-a",
                "text": "食べれる",
                "phone_symbols": ["t", "a", "b", "e", "r", "e", "r", "u"],
                "sequence_score": 987654321,
                "rank": 1,
            },
            {
                "id": "orig-b",
                "text": "食べられる",
                "phone_symbols": ["t", "a", "b", "e", "r", "a", "r", "e", "r", "u"],
                "sequence_score": 123456789,
                "rank": 2,
            },
        ],
    }


def answer(choice="c00"):
    return {
        "model": m.MODEL,
        "answers": {
            "supported_utterance": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.9,
                "probabilities": {"c00": 0.9, "c01": 0.05, "abstain": 0.05},
            }
        },
        "usage": {"input_tokens": 500, "output_tokens": 40},
    }


@pytest.mark.parametrize("arm", ["greedy", "paths", "reordered", "no_observation", "repeat"])
def test_strict_payload_projection(record, arm):
    before = copy.deepcopy(record)
    request, aliases = m.build_request(record, [record["phone_greedy"]], arm)
    encoded = json.dumps(request, ensure_ascii=False)
    for forbidden in [
        "NEVER_SEND",
        "GOLD_SENTINEL",
        "食べれる",
        "食べられる",
        "987654321",
        "123456789",
        "sequence_score",
        "orig-a",
        "reference",
        "context",
    ]:
        assert forbidden not in encoded
    assert record == before
    assert set(aliases.values()) == {"orig-a", "orig-b"}
    assert request["model"] == "jev-1.13.0"


def test_request_digest_changes_with_evidence(record):
    first, _ = m.build_request(record, [], "greedy")
    record["phone_greedy"] = ["b"]
    second, _ = m.build_request(record, [], "greedy")
    assert m.digest(first) != m.digest(second)


def test_reference_cannot_change_request(record):
    first, _ = m.build_request(record, [], "greedy")
    record["reference"] = "COMPLETELY_DIFFERENT"
    second, _ = m.build_request(record, [], "greedy")
    assert m.digest(first) == m.digest(second)


def test_reordered_aliases_bind_original_identity(record):
    _, first = m.build_request(record, [], "greedy")
    _, second = m.build_request(record, [], "reordered")
    assert first["c00"] == second["c01"] == "orig-a"
    assert first["c01"] == second["c00"] == "orig-b"


def test_repeat_is_same_payload_not_new_evidence(record):
    a, _ = m.build_request(record, [], "greedy")
    b, _ = m.build_request(record, [], "repeat")
    assert m.digest(a) == m.digest(b)


def test_paths_are_separate_from_greedy(record):
    a, _ = m.build_request(record, [["a"], ["b"]], "paths")
    assert a["state"]["audio_observation"]["valid_ctc_prefix_alternatives"] == [["a"], ["b"]]
    assert not a["state"]["audio_observation"]["search"]["exhaustive"]


def test_no_observation_really_removes_evidence(record):
    payload, _ = m.build_request(record, [["secret"]], "no_observation")
    assert payload["state"]["audio_observation"] == {"greedy_phones": [], "available": False}
    assert "secret" not in json.dumps(payload)


@pytest.mark.parametrize(
    "choice,reason", [("abstain", "model-abstained"), ("outside", "invalid-id")]
)
def test_fallback_preserves_baseline(record, choice, reason):
    selected, why = m.selected_or_baseline(record, choice, {"c00": "orig-a", "c01": "orig-b"})
    assert selected == "orig-a" and why == reason


def test_missing_audio_gate(record):
    assert m.selected_or_baseline(record, "c01", {"c01": "orig-b"}, False) == (
        "orig-a",
        "missing-observation",
    )


def test_homophone_gate(record):
    record["candidates"][1]["phone_symbols"] = list(record["candidates"][0]["phone_symbols"])
    assert m.selected_or_baseline(record, "c01", {"c01": "orig-b"}) == (
        "orig-a",
        "indistinguishable-pronunciations",
    )


def test_unique_supported_choice(record):
    assert m.selected_or_baseline(record, "c01", {"c01": "orig-b"}) == ("orig-b", None)


def test_model_mismatch():
    value = answer()
    value["model"] = "jev-latest"
    with pytest.raises(ValueError):
        m.validate_answer(value, {"c00": "a", "c01": "b"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9"])
def test_invalid_response_numbers(bad):
    value = answer()
    value["answers"]["supported_utterance"]["confidence"] = bad
    with pytest.raises(ValueError):
        m.validate_answer(value, {"c00": "a", "c01": "b"})


def test_unknown_response_choice():
    with pytest.raises(ValueError):
        m.validate_answer(answer("outside"), {"c00": "a", "c01": "b"})


def test_missing_response_probability():
    value = answer()
    del value["answers"]["supported_utterance"]["probabilities"]["c01"]
    with pytest.raises(ValueError):
        m.validate_answer(value, {"c00": "a", "c01": "b"})


def test_unnormalized_probability():
    value = answer()
    value["answers"]["supported_utterance"]["probabilities"]["abstain"] = 0.7
    with pytest.raises(ValueError):
        m.validate_answer(value, {"c00": "a", "c01": "b"})


def test_valid_response():
    assert m.validate_answer(answer(), {"c00": "a", "c01": "b"})["choice"] == "c00"


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_boolean_usage_is_not_integer(field):
    value = answer()
    value["usage"][field] = True
    with pytest.raises(ValueError):
        m.validate_answer(value, {"c00": "a", "c01": "b"})


def collapse(path, blank=0):
    return tuple(
        symbol
        for i, symbol in enumerate(path)
        if symbol != blank and (i == 0 or path[i - 1] != symbol)
    )


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
def test_ctc_prefix_matches_exhaustive_path_sum(frames):
    probabilities = [[0.2, 0.5, 0.3], [0.3, 0.2, 0.5], [0.6, 0.3, 0.1], [0.1, 0.4, 0.5]][:frames]
    logs = [[math.log(x) for x in row] for row in probabilities]
    expected = {}
    for path in itertools.product(range(3), repeat=frames):
        key = collapse(path)
        score = math.prod(probabilities[t][v] for t, v in enumerate(path))
        expected[key] = expected.get(key, 0) + score
    actual = dict(m.prefix_beam(logs, 0, width=1000, frame_top_k=None))
    actual = {key: value for key, value in actual.items() if math.isfinite(value)}
    assert set(actual) == set(expected)
    for key, probability in expected.items():
        assert math.exp(actual[key]) == pytest.approx(probability, abs=1e-12)


def test_repeated_phone_requires_blank_for_two_outputs():
    logs = [[math.log(x) for x in row] for row in [[0.01, 0.99], [0.99, 0.01], [0.01, 0.99]]]
    assert m.prefix_beam(logs, 0, 20, None)[0][0] == (1, 1)


def test_pruned_ctc_only_has_valid_sequences():
    logs = [[math.log(x) for x in row] for row in [[0.2, 0.5, 0.3]] * 3]
    possible = {collapse(path) for path in itertools.product(range(3), repeat=3)}
    assert all(prefix in possible for prefix, _ in m.prefix_beam(logs, 0, 2, 1))


def test_empty_key_is_rejected():
    with pytest.raises(ValueError):
        m.Client("", 1, 64000, 1)


def test_call_budget_never_enters_network():
    client = m.Client("NOT_A_REAL_KEY", 1, 64000, 10)
    client.calls = 1
    assert client.call({"state": "x"})["reason"] == "execution-budget"


def test_token_reservation_never_enters_network():
    client = m.Client("NOT_A_REAL_KEY", 1, 64000, 10)
    client.input_tokens = 1
    assert client.call({"state": "x"})["reason"] == "execution-budget"


def test_request_byte_limit_never_enters_network():
    client = m.Client("NOT_A_REAL_KEY", 1, 64000, 10)
    assert client.call({"state": "x" * 64001})["reason"] == "payload-byte-limit"


def test_redirect_is_not_followed():
    assert m.NoRedirect().redirect_request(None, None, 307, "", {}, "https://other.invalid") is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("--max-records", "97"),
        ("--max-calls", "601"),
        ("--max-input-tokens", "4000001"),
        ("--max-wall-seconds", "3601"),
    ],
)
def test_cli_bounds(field, value):
    with pytest.raises(SystemExit):
        m.parse_args(["--probe-dir", "a", "--output-dir", "b", field, value])


def test_normalization_does_not_erase_fillers_or_repairs():
    assert m.normalized("えっと、１５…じゃなくて５０。") == "えっと15じゃなくて50"
    assert m.edits(m.normalized("食べれる"), m.normalized("食べられる")) == 1


def test_duplicate_ids_rejected(record):
    record["candidates"][1]["id"] = "orig-a"
    with pytest.raises(ValueError):
        m.build_request(record, [], "greedy")


def test_baseline_mismatch_rejected(record):
    record["baseline_text"] = "not its candidate"
    with pytest.raises(ValueError):
        m.build_request(record, [], "greedy")


@pytest.mark.parametrize("arm", ["joined", "joined_reordered"])
def test_joined_is_only_a_serialization_change(record, arm):
    original_arm = "reordered" if arm.endswith("reordered") else "greedy"
    original, aliases = m.build_request(record, [], original_arm)
    joined, joined_aliases = m.build_request(record, [], arm)
    assert aliases == joined_aliases
    assert original["questions"] == joined["questions"]
    a = original["state"]
    b = joined["state"]
    assert (
        b["audio_observation"]["greedy_phones"].split() == a["audio_observation"]["greedy_phones"]
    )
    for left, right in zip(
        a["candidate_pronunciation_hypotheses"],
        b["candidate_pronunciation_hypotheses"],
        strict=True,
    ):
        assert left["id"] == right["id"]
        assert left["phones"] == right["phones"].split()
    encoded = json.dumps(joined, ensure_ascii=False)
    assert "NEVER_SEND" not in encoded and "食べれる" not in encoded


def test_duplicate_main_arms_are_rejected():
    with pytest.raises(SystemExit):
        m.parse_args(["--probe-dir", "a", "--output-dir", "b", "--main-arms", "joined", "joined"])


@pytest.mark.parametrize("bad", [None, [], "not-an-object"])
def test_response_must_be_an_object(bad):
    with pytest.raises(ValueError):
        m.validate_answer(bad, {"c00": "a", "c01": "b"})


def test_empty_observation_is_explicitly_unavailable(record):
    record["phone_greedy"] = []
    request, _ = m.build_request(record, [], "greedy")
    assert not request["state"]["audio_observation"]["available"]


def test_unexecuted_model_has_no_measured_cer():
    row = {
        "candidate_count": 1,
        "oracle_exact_in_candidates": True,
        "duration_seconds": 1,
        "reference_characters": 10,
        "baseline_errors": 2,
        "arms": {
            "baseline": {"errors": 2},
            "greedy": {
                "errors": 2,
                "raw_choice": None,
                "gate_reason": "not-executed",
                "response": {"status": "not-executed"},
            },
        },
    }
    report = m.summarize([row], {"status": "offline-only"})
    assert report["arms"]["baseline"]["cer"] == 0.2
    assert report["arms"]["greedy"]["cer"] is None
    assert report["arms"]["greedy"]["fallback_cer"] == 0.2
    assert report["arms"]["greedy"]["validated_model_decisions"] == 0
    assert report["arms"]["greedy"]["evaluation_status"] == "no-valid-model-decisions"
