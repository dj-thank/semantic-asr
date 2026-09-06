from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from semantic_asr.contracts import CandidateEvidence
from semantic_asr.distillation import (
    MultiTeacherConfig,
    TeacherJudgment,
    aggregate_teacher_judgments,
    candidate_set_digest,
)
from semantic_asr.learned_fusion import (
    FusionTrainingExample,
    LearnedFusionConfig,
    train_constrained_fusion,
)
from semantic_asr.listwise_training import (
    ListwiseTrainingConfig,
    train_listwise_semantic_mwer,
)
from semantic_asr.ngram import NGramLanguageModel
from semantic_asr.progressive_reranking import ProgressiveStage, progressive_rerank
from semantic_asr.ranker_calibration import (
    RankerCalibrationSample,
    fit_ranker_calibration,
)
from semantic_asr.ranker_training import (
    RankerExample,
    RankerTrainingConfig,
    train_pairwise_ranker,
)
from semantic_asr.rerankers import LinearCandidateRanker, StaticCandidateRanker
from semantic_asr.synthetic import synthetic_ranker_example


def _rotated(reference: str, example_id: str, offset: int) -> RankerExample:
    base = synthetic_ranker_example(
        reference,
        example_id=example_id,
        maximum_negatives=7,
        seed=41 + offset,
    )
    candidates = list(base.candidates)
    offset %= len(candidates)
    return RankerExample(
        example_id=base.example_id,
        candidates=tuple(candidates[offset:] + candidates[:offset]),
        losses=base.losses,
        context=base.context,
    )


def _training_examples() -> list[RankerExample]:
    references = [
        "えっと明日は行きません。料金は3000円です。",
        "学校へ行って切符を買います。",
        "スーパーでしんぶんを買った。",
        "きょうは東京へ行きます。",
        "会議は8月31日の19時です。",
        "予算は一億円ではありません。",
        "Qwen3-ASRをローカルで試します。",
        "この修正はまだ確定していません。",
    ]
    return [
        _rotated(reference, f"train-{index:03d}", index + 1)
        for index, reference in enumerate(references)
    ]


def _calibration_samples(
    ranker: LinearCandidateRanker,
) -> list[RankerCalibrationSample]:
    references = [
        "明日の料金は4200円です。",
        "私は行かないと思います。",
        "面談は9月2日の10時30分です。",
        "OpenAIの音声モデルを比較します。",
    ]
    output: list[RankerCalibrationSample] = []
    for index, reference in enumerate(references):
        example = _rotated(reference, f"calibration-{index:03d}", index + 2)
        scores = ranker.score(example.candidates, context=example.context)
        oracle = min(example.losses.values())
        for candidate in example.candidates:
            output.append(
                RankerCalibrationSample(
                    sample_id=f"{example.example_id}:{candidate.candidate_id}",
                    group_id=f"calibration-group-{index}",
                    score=float(scores[candidate.candidate_id]),
                    correct=example.losses[candidate.candidate_id] <= oracle + 1e-12,
                )
            )
    return output


def _fusion_examples() -> list[FusionTrainingExample]:
    output: list[FusionTrainingExample] = []
    for index in range(24):
        correct = f"correct-{index}"
        fluent = f"fluent-{index}"
        near = f"near-{index}"
        output.append(
            FusionTrainingExample(
                example_id=f"fusion-{index}",
                group_id=f"fusion-speaker-{index % 6}",
                candidates=(
                    CandidateEvidence(
                        correct,
                        "実際の発話です",
                        acoustic=0.91,
                        mora=0.92,
                        lexical=0.25,
                        preservation=0.86,
                        cross_model=0.83,
                    ),
                    CandidateEvidence(
                        fluent,
                        "文法的に自然な捏造です",
                        acoustic=0.16,
                        mora=0.20,
                        lexical=0.98,
                        preservation=0.28,
                        cross_model=0.18,
                    ),
                    CandidateEvidence(
                        near,
                        "実際に近い発話です",
                        acoustic=0.58,
                        mora=0.56,
                        lexical=0.68,
                        preservation=0.62,
                        cross_model=0.51,
                    ),
                ),
                target_distribution={correct: 1.0, fluent: 0.0, near: 0.0},
            )
        )
    return output


def run_validation() -> dict[str, object]:
    examples = _training_examples()
    pairwise = train_pairwise_ranker(
        examples,
        name="validation-pairwise",
        config=RankerTrainingConfig(epochs=120, learning_rate=0.06, seed=43),
    )
    listwise = train_listwise_semantic_mwer(
        examples,
        name="validation-listwise",
        config=ListwiseTrainingConfig(
            epochs=180,
            learning_rate=0.05,
            l2=0.001,
            seed=47,
        ),
    )
    listwise_ranker = LinearCandidateRanker(listwise.profile)
    calibration = fit_ranker_calibration(
        _calibration_samples(listwise_ranker),
        name="validation-calibration",
        source_ranker=listwise_ranker.name,
        minimum_samples=8,
        minimum_groups=2,
    )
    fusion = train_constrained_fusion(
        _fusion_examples(),
        name="validation-constrained-fusion",
        config=LearnedFusionConfig(
            epochs=160,
            learning_rate=0.06,
            acoustic_family_floor=0.72,
            seed=53,
        ),
    )
    ngram = NGramLanguageModel(order=4, mode="character", alpha=0.05).fit(
        [
            "料金は3000円です",
            "料金は3000円です",
            "会議は8月31日です",
            "明日は行きません",
        ]
    )

    teacher_candidates = (
        CandidateEvidence("teacher-a", "料金は3000円です"),
        CandidateEvidence("teacher-b", "料金は30000円です"),
    )
    digest = candidate_set_digest(teacher_candidates)
    # Entropy is derived from these score distributions by the canonical aggregator.
    # A fixture must not supply a self-reported entropy as acoustic/confidence evidence.
    teacher_consensus = aggregate_teacher_judgments(
        teacher_candidates,
        [
            TeacherJudgment(
                teacher="teacher-8b",
                candidate_set_sha256=digest,
                scores={"teacher-a": 2.4, "teacher-b": -0.7},
                score_kind="logit",
                reliability=0.90,
            ),
            TeacherJudgment(
                teacher="teacher-12b",
                candidate_set_sha256=digest,
                scores={"teacher-a": 0.82, "teacher-b": 0.18},
                score_kind="preference",
                reliability=0.86,
            ),
        ],
        config=MultiTeacherConfig(
            minimum_active_teachers=2,
            maximum_teacher_share=0.60,
            maximum_disagreement=0.42,
        ),
    )

    progressive = progressive_rerank(
        teacher_candidates,
        [
            ProgressiveStage(
                name="cheap-student",
                ranker=StaticCandidateRanker({"teacher-a": 4.0, "teacher-b": -2.0}),
                estimated_cost_ms=2,
                minimum_margin=0.50,
                maximum_entropy=0.40,
            ),
            ProgressiveStage(
                name="offline-teacher",
                ranker=StaticCandidateRanker({"teacher-a": 5.0, "teacher-b": -3.0}),
                estimated_cost_ms=1_000,
            ),
        ],
        budget_ms=2_000,
    )

    result = {
        "schemaVersion": "semantic-asr-model-free-validation-v1",
        "claimBoundary": (
            "Deterministic synthetic/model-free training validates implementation paths only; "
            "it is not evidence of real-audio recognition improvement."
        ),
        "pairwise": {
            "before": asdict(pairwise.before),
            "after": asdict(pairwise.after),
            "profileDigest": pairwise.profile.digest,
            "trainingManifestSha256": pairwise.training_manifest_sha256,
        },
        "listwise": {
            "before": asdict(listwise.before),
            "after": asdict(listwise.after),
            "profileDigest": listwise.profile.digest,
            "trainingManifestSha256": listwise.training_manifest_sha256,
        },
        "calibration": {
            "before": asdict(calibration.before),
            "after": asdict(calibration.after),
            "profileDigest": calibration.profile.digest,
            "calibrationManifestSha256": (calibration.profile.calibration_manifest_sha256),
            "converged": calibration.converged,
        },
        "fusion": {
            "before": asdict(fusion.before),
            "after": asdict(fusion.after),
            "weights": fusion.profile.weights,
            "profileDigest": fusion.profile.digest,
            "acousticFamilyFloor": fusion.profile.acoustic_family_floor,
        },
        "ngram": {
            "digest": ngram.digest,
            "correctScore": ngram.score("料金は3000円です").average_log_probability,
            "wrongScore": ngram.score("料金は30000円です").average_log_probability,
        },
        "teacherConsensus": asdict(teacher_consensus),
        "progressiveReranking": asdict(progressive),
    }

    assert pairwise.after.mean_logistic_loss < pairwise.before.mean_logistic_loss
    assert listwise.after.mean_expected_loss < listwise.before.mean_expected_loss
    assert calibration.after.negative_log_likelihood < calibration.before.negative_log_likelihood
    assert fusion.after.cross_entropy < fusion.before.cross_entropy
    assert result["ngram"]["correctScore"] > result["ngram"]["wrongScore"]
    assert teacher_consensus.usable_for_distillation
    assert progressive.early_exit and progressive.used_budget_ms == 2
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run_validation()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
