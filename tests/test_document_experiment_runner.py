from __future__ import annotations

import json
from pathlib import Path

import pytest
from _document_experiment_fixture import AUDIO, RIGHTS, SPLIT, fake_plan, first_pass

from semantic_asr.contracts import sha256_json
from semantic_asr.document_experiment.ngram_scorer import DocumentLanguageScore
from semantic_asr.document_experiment.protocol import (
    CriticalReferenceToken,
    DocumentExperimentArm,
    DocumentExperimentCase,
    DocumentExperimentManifest,
    DocumentExperimentProtocol,
    FrozenReference,
)
from semantic_asr.document_experiment.runner import (
    prepare_document_experiment,
    run_document_context_experiment,
)


class ToyScorer:
    profile_digest = "d" * 64

    def score_path(
        self,
        path,
        arm,
        *,
        case_id,
        left_context="",
        right_context="",
        maximum_characters,
    ):
        del case_id, left_context, right_context
        value = 0.8 if "まだ" in path.text else -0.8
        scored = min(len(path.text), maximum_characters)
        return DocumentLanguageScore(
            value=value,
            raw_average_log_likelihood=value,
            forward_average_log_likelihood=value,
            backward_average_log_likelihood=(value if arm.direction == "bidirectional" else None),
            source="toy-scorer",
            profile_digest=self.profile_digest,
            path_digest=path.digest,
            arm_digest=arm.digest,
            scored_characters=scored * (2 if arm.direction == "bidirectional" else 1),
            scorer_calls=2 if arm.direction == "bidirectional" else 1,
        )


def experiment_case() -> DocumentExperimentCase:
    windows = (
        "レビュー完了まではまだマージしません。",
        "承認後に統合します。",
    )
    reference = FrozenReference(
        reference_id="reference",
        source_audio_sha256=AUDIO,
        text="".join(windows),
        window_texts=windows,
        critical_tokens=(
            CriticalReferenceToken(kind="negation", text="ません"),
            CriticalReferenceToken(kind="entity", text="マージ"),
        ),
    )
    return DocumentExperimentCase(
        case_id="case-1",
        first_pass=first_pass(),
        reference=reference,
        rights_decision="allow",
        license_id="fixture-license",
        source_id="source-1",
        speaker_id="test-speaker-1",
        session_id="test-session-1",
        dataset_revision="fixture-r1",
        split_manifest_sha256=SPLIT,
    )


def manifest() -> DocumentExperimentManifest:
    return DocumentExperimentManifest(
        name="fixture-manifest",
        revision="r1",
        cases=(experiment_case(),),
        rights_registry_sha256=RIGHTS,
        split_manifest_sha256=SPLIT,
        training_speaker_ids=("training-speaker",),
        calibration_speaker_ids=("calibration-speaker",),
    )


def protocol() -> DocumentExperimentProtocol:
    return DocumentExperimentProtocol(
        name="fixture-protocol",
        revision="r1",
        arms=(
            DocumentExperimentArm(
                name="acoustic-only",
                candidate_view="acoustic-only",
                direction="none",
                is_control=True,
            ),
            DocumentExperimentArm(
                name="ordered-bidirectional",
                candidate_view="ordered-document",
                direction="bidirectional",
                scorer_key="toy",
                linguistic_weight=1.0,
            ),
            DocumentExperimentArm(
                name="shuffled-control",
                candidate_view="shuffled-document",
                direction="bidirectional",
                scorer_key="toy",
                linguistic_weight=1.0,
                is_control=True,
            ),
        ),
        baseline_arm="acoustic-only",
        maximum_candidate_documents=3,
        maximum_scored_characters=10_000,
        bootstrap_resamples=200,
        bootstrap_seed=11,
    )


def test_planner_receives_no_reference_and_candidates_are_frozen_once() -> None:
    seen = []

    def planner(view):
        seen.append(view)
        assert not hasattr(view, "reference")
        return fake_plan()

    prepared = prepare_document_experiment(manifest(), protocol(), planner)

    assert len(prepared) == 1
    assert len(seen) == 1
    assert prepared[0].candidates.retained_path_digest == fake_plan().decision.retained.digest
    assert len(prepared[0].candidates.paths) == 3


def test_all_arms_share_candidates_and_context_arm_improves_fixture() -> None:
    prepared = prepare_document_experiment(manifest(), protocol(), lambda view: fake_plan())

    report = run_document_context_experiment(
        prepared,
        manifest(),
        protocol(),
        scorers={"toy": ToyScorer()},
    )

    rows = {(row.case_id, row.arm_name): row for row in report.case_results}
    baseline = rows[("case-1", "acoustic-only")]
    ordered = rows[("case-1", "ordered-bidirectional")]
    assert baseline.selected_text != ordered.selected_text
    assert "またマージ" in baseline.selected_text
    assert "まだマージ" in ordered.selected_text
    assert ordered.metrics.text.strict_edits < baseline.metrics.text.strict_edits
    assert {row.candidate_set_digest for row in report.case_results if row.case_id == "case-1"} == {
        baseline.candidate_set_digest
    }
    assert report.paired_intervals[0].point_delta < 0.0


def test_report_writes_canonical_evidence_with_negative_results(tmp_path: Path) -> None:
    frozen_manifest = manifest()
    frozen_protocol = protocol()
    prepared = prepare_document_experiment(
        frozen_manifest,
        frozen_protocol,
        lambda view: fake_plan(),
    )
    report = run_document_context_experiment(
        prepared,
        frozen_manifest,
        frozen_protocol,
        scorers={"toy": ToyScorer()},
    )

    destination = report.write(tmp_path / "report.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["reportDigest"] == report.digest
    assert payload["protocolDigest"] == frozen_protocol.digest
    assert payload["manifestDigest"] == frozen_manifest.digest
    assert any(row["arm_name"] == "shuffled-control" for row in payload["aggregates"])


def test_scored_character_budget_fails_closed() -> None:
    class BudgetBreaker(ToyScorer):
        def score_path(self, path, arm, **kwargs):
            row = super().score_path(path, arm, **kwargs)
            return DocumentLanguageScore(
                value=row.value,
                raw_average_log_likelihood=row.raw_average_log_likelihood,
                forward_average_log_likelihood=row.forward_average_log_likelihood,
                backward_average_log_likelihood=row.backward_average_log_likelihood,
                source=row.source,
                profile_digest=row.profile_digest,
                path_digest=row.path_digest,
                arm_digest=row.arm_digest,
                scored_characters=1_000_000,
                scorer_calls=row.scorer_calls,
            )

    frozen_manifest = manifest()
    frozen_protocol = protocol()
    prepared = prepare_document_experiment(
        frozen_manifest,
        frozen_protocol,
        lambda view: fake_plan(),
    )

    with pytest.raises(ValueError, match="scored-character budget"):
        run_document_context_experiment(
            prepared,
            frozen_manifest,
            frozen_protocol,
            scorers={"toy": BudgetBreaker()},
        )


def test_candidate_set_digest_binds_planner_output() -> None:
    frozen_manifest = manifest()
    frozen_protocol = protocol()
    prepared = prepare_document_experiment(
        frozen_manifest,
        frozen_protocol,
        lambda view: fake_plan(),
    )

    assert prepared[0].candidates.candidate_set_digest == sha256_json(
        {
            "caseId": prepared[0].candidates.case_id,
            "firstPassEvidenceSha256": prepared[0].candidates.first_pass_evidence_sha256,
            "planningDigest": prepared[0].candidates.planning_digest,
            "plannerOutputDigest": prepared[0].candidates.planner_output_digest,
            "retainedPathDigest": prepared[0].candidates.retained_path_digest,
            "paths": [
                {
                    "pathDigest": path.digest,
                    "baseScore": path.base_score,
                    "meanAudioSupport": path.mean_audio_support,
                    "optionDigests": [option.digest for option in path.options],
                }
                for path in prepared[0].candidates.paths
            ],
        }
    )
