"""Synthetic contract tests, not measured Japanese recognition accuracy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from semantic_asr.adapters import DecodeRequest
from semantic_asr.advanced_adapters import LoopGuardConfig, PathPreservingFasterWhisperAdapter
from semantic_asr.api import _confidence_eligible, runtime_profile
from semantic_asr.cache import EvidenceCache
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.fusion import fuse_candidates
from semantic_asr.longform import SemanticASRTranscriber
from semantic_asr.planner import EvidenceBudget, plan_evidence
from semantic_asr.semantic_lattice import build_semantic_lattice


def _candidate(candidate_id: str = "only", text: str = "明日は行きます") -> CandidateEvidence:
    return CandidateEvidence(
        candidate_id,
        text,
        acoustic=0.95,
        mora=0.95,
        preservation=0.95,
        avg_logprob=-0.01,
        source="fixture-primary",
    )


def _inputs(candidates=None):
    candidates = candidates or [_candidate()]
    ranked = fuse_candidates(candidates)
    lattice = build_semantic_lattice(
        candidates,
        posterior=ranked[0].gate.posterior,
        segment_start_ms=1_000,
        segment_end_ms=3_000,
    )
    return ranked, lattice


def test_characterize_singleton_posterior_is_not_search_coverage() -> None:
    ranked, lattice = _inputs()
    assert ranked[0].gate.posterior == {"only": 1.0}
    assert ranked[0].gate.needs_relisten is False
    assert not lattice.contradiction_islands
    plan = plan_evidence(
        ranked, lattice, whole_window_start_ms=1_000, whole_window_end_ms=3_000
    )
    assert plan.selected == ()
    assert plan.stopping_reason == "observation-already-confident"


def test_opt_in_explores_confident_singleton_without_a_contradiction() -> None:
    ranked, lattice = _inputs()
    plan = plan_evidence(
        ranked,
        lattice,
        expand_collapsed_candidates=True,
        enabled=("whisper-relisten", "qwen-second-ear"),
        whole_window_start_ms=1_000,
        whole_window_end_ms=3_000,
    )
    assert len(plan.selected) == 1
    assert plan.selected[0].kind == "qwen-second-ear"
    assert (plan.selected[0].start_ms, plan.selected[0].end_ms) == (1_000, 3_000)
    assert "collapsed-candidate-pool" in plan.selected[0].reasons


@pytest.mark.parametrize("text", ["明日は行きます", "明日 は行きます。", "明日は行きます！"])
def test_duplicate_or_format_only_paths_are_not_search_diversity(text: str) -> None:
    ranked, lattice = _inputs([_candidate(), _candidate("duplicate", text)])
    plan = plan_evidence(
        ranked, lattice, expand_collapsed_candidates=True,
        whole_window_start_ms=1_000, whole_window_end_ms=3_000,
    )
    assert len(plan.selected) == 1
    assert plan.stopping_reason == "candidate-pool-expansion"


def test_noncollapsed_pool_keeps_existing_localized_plan() -> None:
    ranked, lattice = _inputs([_candidate(), _candidate("other", "明日は行きません")])
    default = plan_evidence(ranked, lattice)
    expanded = plan_evidence(ranked, lattice, expand_collapsed_candidates=True)
    assert expanded == default
    assert all("candidate-expansion" not in action.action_id for action in expanded.selected)


@pytest.mark.parametrize(
    "budget",
    [EvidenceBudget(0, 1), EvidenceBudget(12_000, 0), EvidenceBudget(479, 1),
     EvidenceBudget(12_000, 1, minimum_utility=1.0)],
)
def test_expansion_respects_budget_actions_and_utility(budget: EvidenceBudget) -> None:
    ranked, lattice = _inputs()
    plan = plan_evidence(
        ranked, lattice, expand_collapsed_candidates=True, budget=budget,
        whole_window_start_ms=1_000, whole_window_end_ms=3_000,
    )
    assert not plan.selected
    assert plan.rejected
    assert plan.used_ms == 0


def test_expensive_second_ear_falls_back_to_admissible_wider_primary() -> None:
    ranked, lattice = _inputs()
    plan = plan_evidence(
        ranked, lattice, expand_collapsed_candidates=True, budget=EvidenceBudget(480, 1),
        whole_window_start_ms=1_000, whole_window_end_ms=3_000,
    )
    assert len(plan.selected) == 1
    assert plan.selected[0].kind == "whisper-relisten"
    assert plan.used_ms == 480
    assert plan.rejected[0].kind == "qwen-second-ear"


@pytest.mark.parametrize("start,end", [(None, None), (0, None), (None, 2_000)])
def test_missing_window_does_not_fabricate_span(start, end) -> None:
    ranked, lattice = _inputs()
    plan = plan_evidence(
        ranked, lattice, expand_collapsed_candidates=True,
        whole_window_start_ms=start, whole_window_end_ms=end,
    )
    assert not plan.selected
    assert plan.stopping_reason == "expansion-missing-window"


@pytest.mark.parametrize("start,end", [(-1, 2_000), (2_000, 2_000), (3_000, 2_000),
                                         (False, 2_000), (0, 2.5)])
def test_invalid_window_is_rejected(start, end) -> None:
    ranked, lattice = _inputs()
    with pytest.raises((TypeError, ValueError)):
        plan_evidence(
            ranked, lattice, expand_collapsed_candidates=True,
            whole_window_start_ms=start, whole_window_end_ms=end,
        )


@pytest.mark.parametrize("value", [1, "yes", None])
def test_expansion_flag_is_not_implicitly_coerced(value) -> None:
    ranked, lattice = _inputs()
    with pytest.raises(TypeError):
        plan_evidence(ranked, lattice, expand_collapsed_candidates=value)
    with pytest.raises(TypeError):
        SemanticASRTranscriber(_Decoder(), expand_collapsed_candidates=value)
    with pytest.raises(TypeError):
        plan_evidence(ranked, lattice, allow_primary_expansion=value)


class _Decoder:
    """In-memory fixture; never loads a model or reads audio."""

    model_revision = "fixture-revision-1"
    runtime_revision = "fixture-runtime-1"

    def __init__(self, *, name="primary", fail=False, empty=False, same=False):
        self.name = name
        self.model_name = f"fixture-{name}"
        self.fail = fail
        self.empty = empty
        self.same = same
        self.requests: list[DecodeRequest] = []

    def decode(self, request: DecodeRequest) -> list[CandidateEvidence]:
        self.requests.append(request)
        additional = self.name == "secondary" or request.beam_size > 5
        if additional and self.fail:
            raise RuntimeError("sensitive-path-or-token-must-not-be-published")
        if additional and self.empty:
            return []
        text = "明日は行きません" if additional and not self.same else "明日は行きます"
        return [replace(_candidate(text=text), source=self.name)]


def _run(tmp_path: Path, primary=None, **kwargs):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"synthetic request fixture, not recorded audio")
    primary = primary or _Decoder()
    transcriber = SemanticASRTranscriber(primary, **kwargs)
    result = transcriber.transcribe(path, duration_ms=2_000)
    result.verify()
    return primary, result


def test_default_does_not_add_model_calls_or_change_singleton_decision(tmp_path: Path) -> None:
    second = _Decoder(name="secondary")
    primary, result = _run(tmp_path, second_ear=second)
    assert len(primary.requests) == 1
    assert not second.requests
    assert result.segments[0].observed.decision == "accepted"
    assert "candidateExpansion" not in result.segments[0].diagnostics


@pytest.mark.parametrize("balanced", [False, True])
def test_whole_window_second_ear_adds_alternative_without_erasing_primary(
    tmp_path: Path, balanced: bool,
) -> None:
    second = _Decoder(name="secondary")
    primary, result = _run(
        tmp_path, second_ear=second, expand_collapsed_candidates=True,
        balanced_router=balanced,
    )
    assert len(primary.requests) == len(second.requests) == 1
    request = second.requests[0]
    assert (request.start_ms, request.end_ms) == (0, 2_000)
    assert (request.beam_size, request.hypotheses) == (1, 1)
    segment = result.segments[0]
    assert {row.text for row in segment.observed.candidates} == {"明日は行きます", "明日は行きません"}
    assert segment.observed.decision == "provisional"
    assert segment.observed.force_provisional is True
    receipt = segment.diagnostics["candidateExpansion"]
    assert receipt["initialDistinctSurfaceCount"] == 1
    assert receipt["finalDistinctSurfaceCount"] == 2
    assert receipt["addedDistinctSurfaceCount"] == 1
    assert receipt["attempted"] is True and receipt["completed"] is True
    assert len(segment.actions) == 1
    assert segment.normalized.observed_evidence_sha256 == segment.observed.evidence_sha256
    if balanced:
        assert segment.diagnostics["evidenceRouting"]["bypassedReason"] == "single-expansion-limit"


def test_primary_only_expands_search_once(tmp_path: Path) -> None:
    primary, result = _run(tmp_path, expand_collapsed_candidates=True)
    assert [r.beam_size for r in primary.requests] == [5, 12]
    assert [r.hypotheses for r in primary.requests] == [5, 8]
    assert {(r.start_ms, r.end_ms) for r in primary.requests} == {(0, 2_000)}
    assert result.segments[0].diagnostics["candidateExpansion"]["addedDistinctSurfaceCount"] == 1


@pytest.mark.parametrize("beam,hypotheses", [(5, 5), (4, 4), (12, 4)])
def test_identical_or_narrower_search_is_not_counted_as_expansion(
    tmp_path: Path, beam: int, hypotheses: int,
) -> None:
    primary, result = _run(
        tmp_path, expand_collapsed_candidates=True,
        relisten_beam_size=beam, relisten_hypotheses=hypotheses,
    )
    assert len(primary.requests) == 1
    segment = result.segments[0]
    assert segment.diagnostics["candidateExpansion"]["reason"] == "expansion-no-eligible-backend"
    assert segment.observed.decision == "provisional"


@pytest.mark.parametrize("kwargs", [{"fail": True}, {"empty": True}, {"same": True}])
def test_failed_empty_or_redundant_probe_preserves_original_and_is_not_pool_growth(
    tmp_path: Path, kwargs,
) -> None:
    second = _Decoder(name="secondary", **kwargs)
    primary, result = _run(tmp_path, second_ear=second, expand_collapsed_candidates=True)
    assert len(primary.requests) == len(second.requests) == 1
    segment = result.segments[0]
    assert result.observed_text == "明日は行きます"
    assert segment.observed.decision == "provisional"
    receipt = segment.diagnostics["candidateExpansion"]
    assert receipt["addedDistinctSurfaceCount"] == 0
    assert receipt["completed"] is bool(kwargs.get("same"))
    assert "sensitive-path-or-token" not in str(segment.diagnostics)


def test_budget_blocked_expansion_is_not_completion_or_acceptance(tmp_path: Path) -> None:
    primary, result = _run(
        tmp_path, expand_collapsed_candidates=True, evidence_budget=EvidenceBudget(0, 0),
    )
    assert len(primary.requests) == 1
    segment = result.segments[0]
    assert segment.observed.decision == "provisional"
    assert segment.diagnostics["candidateExpansion"]["attempted"] is False
    assert segment.diagnostics["candidateExpansion"]["completed"] is False


def test_expansion_reuses_separate_decode_cache_and_preserves_evidence(tmp_path: Path) -> None:
    primary, second = _Decoder(), _Decoder(name="secondary")
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"fixture")
    with EvidenceCache(tmp_path / "cache.sqlite3") as cache:
        transcriber = SemanticASRTranscriber(
            primary, second_ear=second, cache=cache, expand_collapsed_candidates=True,
        )
        first = transcriber.transcribe(path, duration_ms=2_000)
        repeat = transcriber.transcribe(path, duration_ms=2_000)
    assert len(primary.requests) == len(second.requests) == 1
    first.verify()
    repeat.verify()
    assert first.evidence_sha256 == repeat.evidence_sha256
    assert "base-window" in repeat.segments[0].cache_hits
    assert "qwen-second-ear" in repeat.segments[0].cache_hits
    assert repeat.segments[0].diagnostics["candidateExpansion"]["completed"] is True


def test_expansion_cannot_inherit_old_confidence_even_with_zero_budget() -> None:
    profile = runtime_profile("cpu-ja-v1")
    # Configuration-only fixture: constructing the real adapter would load a model.
    adapter = object.__new__(PathPreservingFasterWhisperAdapter)
    for name, value in {
        "model_name": profile.model,
        "model_revision": profile.model_revision,
        "device": profile.device,
        "compute_type": profile.compute_type,
        "patience": profile.patience,
        "loop_guard": LoopGuardConfig(enabled=profile.loop_guard),
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "without_timestamps": False,
        "cpu_threads": 0,
    }.items():
        setattr(adapter, name, value)
    transcriber = SemanticASRTranscriber(adapter, evidence_budget=EvidenceBudget(0, 0))
    arguments = dict(language="ja", prompted=False, transcriber=transcriber)
    assert _confidence_eligible(profile, adapter, **arguments) is True
    transcriber.expand_collapsed_candidates = True
    assert _confidence_eligible(profile, adapter, **arguments) is False
