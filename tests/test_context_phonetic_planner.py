from __future__ import annotations

from dataclasses import replace

import pytest

from semantic_asr.context_phonetic_experiment.context_scorer import (
    ContextCandidateScore,
)
from semantic_asr.context_phonetic_experiment.planner import (
    prepare_context_phonetic_experiment,
)
from semantic_asr.phonetic_experiment.planner import FrozenPhoneticCandidatePlanner

from _context_phonetic_factorial_fixture import (
    factorial_manifest,
    factorial_protocol,
    utility_artifact,
)


def test_acoustic_pool_is_generated_once_and_context_is_scored_twice(tmp_path) -> None:
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    protocol = factorial_protocol()
    planner = FrozenPhoneticCandidatePlanner(runtime, utility_artifact())

    prepared = prepare_context_phonetic_experiment(
        manifest,
        protocol,
        planner,
        scorer,
    )

    assert len(runtime.calls) == len(manifest.cases)
    expected_context_calls = len(manifest.cases) * len(manifest.cases[0].phonetic_case.lexicon.entries) * 2
    assert len(scorer.calls) == expected_context_calls
    assert len(prepared.cases) == len(manifest.cases)
    for case in prepared.cases:
        candidate_ids = {row.candidate_id for row in case.pool.candidates}
        assert {row.candidate_id for row in case.ordered.scores} == candidate_ids
        assert {row.candidate_id for row in case.shuffled.scores} == candidate_ids
        assert case.ordered.scorer_profile_digest == case.shuffled.scorer_profile_digest
        assert case.ordered.donor_case_id == case.case_id
        assert case.shuffled.donor_case_id != case.case_id


def test_prepared_identity_excludes_runtime_measurements(tmp_path) -> None:
    manifest, runtime, scorer = factorial_manifest(tmp_path)
    prepared = prepare_context_phonetic_experiment(
        manifest,
        factorial_protocol(),
        FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
        scorer,
    )
    case = prepared.cases[0]
    changed_pool = replace(
        case.pool,
        generation_latency_ms=case.pool.generation_latency_ms + 1_000.0,
    )
    changed_ordered = replace(
        case.ordered,
        scoring_latency_ms=case.ordered.scoring_latency_ms + 1_000.0,
    )
    changed = replace(case, pool=changed_pool, ordered=changed_ordered)

    assert changed.digest == case.digest


def test_context_score_bound_to_wrong_candidate_text_is_rejected(tmp_path) -> None:
    manifest, runtime, scorer = factorial_manifest(tmp_path)

    class BadScorer:
        source = scorer.source
        profile_digest = scorer.profile_digest

        def score_many(self, candidates, *, context):
            rows = scorer.score_many(candidates, context=context)
            first = rows[0]
            return (
                ContextCandidateScore(
                    candidate_id=first.candidate_id,
                    candidate_text_sha256="0" * 64,
                    context_digest=first.context_digest,
                    value=first.value,
                    source=first.source,
                    scorer_profile_digest=first.scorer_profile_digest,
                ),
                *rows[1:],
            )

    with pytest.raises(ValueError, match="different candidate text"):
        prepare_context_phonetic_experiment(
            manifest,
            factorial_protocol(),
            FrozenPhoneticCandidatePlanner(runtime, utility_artifact()),
            BadScorer(),  # type: ignore[arg-type]
        )
