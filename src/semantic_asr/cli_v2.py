from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import Qwen3ASRAdapter, Qwen3ForcedAlignerAdapter
from .adaptive import AdaptiveKConfig
from .advanced_adapters import (
    AdaptiveRerankingAdapter,
    PathPreservingFasterWhisperAdapter,
)
from .benchmark import load_benchmark_jsonl, run_benchmark, write_benchmark_report
from .cache import EvidenceCache
from .cached_lm import HashedLMProbabilityCache, import_teacher_rows
from .calibration import CalibrationProfile
from .cascade import CascadeConfig, run_candidate_cascade
from .cli import main as legacy_main
from .contracts import CandidateEvidence
from .distillation import (
    MultiTeacherConfig,
    aggregate_teacher_judgments,
    candidate_set_digest,
    consensus_to_ranker_example,
    judgment_from_row,
)
from .evidence_router import RouterState
from .longform import SemanticASRTranscriber
from .outputs import write_outputs
from .pipeline import effort_profile
from .planner import EvidenceBudget
from .ranker_calibration import (
    RankerCalibrationProfile,
    fit_ranker_calibration,
    load_calibration_samples,
    write_calibration_result,
)
from .ranker_training import (
    RankerTrainingConfig,
    load_jsonl_examples,
    train_pairwise_ranker,
    write_training_result,
)
from .rerankers import (
    CrossEncoderCandidateRanker,
    LinearCandidateRanker,
    LinearRankerProfile,
    Qwen3CandidateRanker,
)
from .synthetic import synthetic_ranker_example
from .teachers import OllamaRanker, OpenAICompatibleRanker

_ADVANCED_COMMANDS = {
    "benchmark",
    "calibrate-ranker",
    "cascade",
    "distill-teachers",
    "train-ranker",
    "synthetic-data",
    "lm-cache-build",
    "research-smoke",
    "transcribe-v2",
}


def _candidate(row: dict[str, Any]) -> CandidateEvidence:
    aliases = {
        "candidateId": "candidate_id",
        "tokenIds": "token_ids",
        "crossModel": "cross_model",
        "moraUnits": "mora_units",
        "hypothesisCount": "hypothesis_count",
        "sequenceScore": "sequence_score",
        "avgLogprob": "avg_logprob",
        "beamConfidence": "beam_confidence",
    }
    return CandidateEvidence.from_dict({aliases.get(key, key): value for key, value in row.items()})


def _load_candidates(path: str | Path) -> list[CandidateEvidence]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("candidate manifest must be an array or contain candidates")
    return [_candidate(dict(row)) for row in rows]


def _write_or_print(payload: object, output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _load_linear_profile(path: str | Path) -> LinearRankerProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    row = payload.get("profile", payload)
    if not isinstance(row, dict):
        raise ValueError("linear ranker profile must be an object")
    return LinearRankerProfile.from_dict(row)


def _load_calibration(
    path: str | Path | None,
) -> CalibrationProfile | RankerCalibrationProfile | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    row = dict(payload.get("profile", payload))
    if payload.get("schemaVersion") == "ranker-calibration-v1" or {
        "slope",
        "intercept",
        "calibration_manifest_sha256",
    }.issubset(row):
        return RankerCalibrationProfile.from_dict(row)
    row.pop("digest", None)
    return CalibrationProfile(**row)


def _load_router_state(path: str | Path | None) -> RouterState:
    if path is None:
        return RouterState()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RouterState(
        selection_count={
            str(key): int(value)
            for key, value in dict(
                payload.get("selectionCount", payload.get("selection_count", {}))
            ).items()
        },
        reward_sum={
            str(key): float(value)
            for key, value in dict(payload.get("rewardSum", payload.get("reward_sum", {}))).items()
        },
        total_selections=int(payload.get("totalSelections", payload.get("total_selections", 0))),
        version=str(payload.get("version", "1")),
    )


def _hotwords(args: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    if args.hotwords:
        values.extend(
            value.strip() for value in args.hotwords.replace("、", ",").split(",") if value.strip()
        )
    if args.hotwords_file:
        text = Path(args.hotwords_file).read_text(encoding="utf-8")
        values.extend(
            value.strip()
            for line in text.splitlines()
            for value in line.replace("、", ",").split(",")
            if value.strip()
        )
    return tuple(dict.fromkeys(values))


def command_cascade(args: argparse.Namespace) -> int:
    candidates = _load_candidates(args.input)
    decision = run_candidate_cascade(
        candidates,
        adaptive_config=AdaptiveKConfig(
            minimum_k=args.minimum_k,
            maximum_k=args.maximum_k,
            posterior_mass_target=args.posterior_mass,
        ),
        cascade_config=CascadeConfig(selection_policy=args.selection_policy),
        semantic_criticality=args.semantic_criticality,
    )
    _write_or_print(asdict(decision), args.output)
    return 0


def command_train_ranker(args: argparse.Namespace) -> int:
    examples = load_jsonl_examples(args.input)
    result = train_pairwise_ranker(
        examples,
        name=args.name,
        config=RankerTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            seed=args.seed,
        ),
    )
    write_training_result(result, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "examples": result.example_count,
                "beforePairwiseAccuracy": result.before.pairwise_accuracy,
                "afterPairwiseAccuracy": result.after.pairwise_accuracy,
                "beforeLoss": result.before.mean_logistic_loss,
                "afterLoss": result.after.mean_logistic_loss,
                "profileDigest": result.profile.digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_calibrate_ranker(args: argparse.Namespace) -> int:
    samples = load_calibration_samples(args.input)
    result = fit_ranker_calibration(
        samples,
        name=args.name,
        source_ranker=args.source_ranker,
        l2=args.l2,
        maximum_iterations=args.maximum_iterations,
        minimum_samples=args.minimum_samples,
        minimum_groups=args.minimum_groups,
    )
    write_calibration_result(result, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "samples": result.profile.sample_count,
                "groups": result.profile.group_count,
                "before": asdict(result.before),
                "after": asdict(result.after),
                "profileDigest": result.profile.digest,
                "converged": result.converged,
                "iterations": result.iterations,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    records = load_benchmark_jsonl(args.input)
    ks = tuple(int(value.strip()) for value in args.ks.split(",") if value.strip())
    report = run_benchmark(
        records,
        ks=ks,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        require_test_split=not args.allow_non_test,
    )
    write_benchmark_report(report, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "samples": report.sample_count,
                "groups": report.group_count,
                "baselineCER": report.baseline_cer,
                "cascadeCER": report.cascade_cer,
                "mbrCER": report.mbr_cer,
                "oracleCERAtK": report.oracle_cer_at_k,
                "cascadeImprovement": (
                    None
                    if report.cascade_improvement is None
                    else asdict(report.cascade_improvement)
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_distill_teachers(args: argparse.Namespace) -> int:
    accepted: list[str] = []
    rejected: list[str] = []
    config = MultiTeacherConfig(
        temperature=args.temperature,
        minimum_active_teachers=args.minimum_active_teachers,
        maximum_teacher_share=args.maximum_teacher_share,
        maximum_disagreement=args.maximum_disagreement,
    )
    for line_number, line in enumerate(
        Path(args.input).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"distillation row {line_number} must be an object")
        raw_candidates = payload.get("candidates")
        raw_judgments = payload.get("judgments")
        if not isinstance(raw_candidates, list) or not isinstance(raw_judgments, list):
            raise ValueError(
                f"distillation row {line_number} requires candidates and judgments arrays"
            )
        candidates = [_candidate(dict(row)) for row in raw_candidates]
        digest = candidate_set_digest(candidates)
        judgments = [
            judgment_from_row(dict(row), candidate_set_sha256=digest) for row in raw_judgments
        ]
        consensus = aggregate_teacher_judgments(
            candidates,
            judgments,
            config=config,
        )
        common = {
            "exampleId": str(payload.get("exampleId") or payload.get("example_id") or line_number),
            "context": str(payload.get("context") or ""),
            "candidateSetSha256": digest,
            "teacherConsensus": asdict(consensus),
        }
        if not consensus.usable_for_distillation:
            rejected.append(json.dumps(common, ensure_ascii=False, separators=(",", ":")))
            continue
        example = consensus_to_ranker_example(
            example_id=common["exampleId"],
            candidates=candidates,
            consensus=consensus,
            context=common["context"],
        )
        accepted.append(
            json.dumps(
                {
                    **common,
                    "candidates": [candidate.as_dict() for candidate in candidates],
                    "losses": example.losses,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        ("\n".join(accepted) + "\n") if accepted else "",
        encoding="utf-8",
    )
    if args.rejected_output:
        rejected_output = Path(args.rejected_output)
        rejected_output.parent.mkdir(parents=True, exist_ok=True)
        rejected_output.write_text(
            ("\n".join(rejected) + "\n") if rejected else "",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "accepted": len(accepted),
                "rejected": len(rejected),
                "output": str(output),
                "rejectedOutput": args.rejected_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_synthetic_data(args: argparse.Namespace) -> int:
    references = [
        line.strip()
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not references:
        raise ValueError("input contains no reference text")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for index, reference in enumerate(references, 1):
        example = synthetic_ranker_example(
            reference,
            example_id=f"synthetic-{index:06d}",
            maximum_negatives=args.maximum_negatives,
            seed=args.seed + index,
            context=args.context,
        )
        rows.append(
            json.dumps(
                {
                    "exampleId": example.example_id,
                    "context": example.context,
                    "candidates": [candidate.as_dict() for candidate in example.candidates],
                    "losses": example.losses,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "examples": len(rows), "output": str(target)}))
    return 0


def command_lm_cache_build(args: argparse.Namespace) -> int:
    key = bytes.fromhex(args.key_hex)
    cache = HashedLMProbabilityCache(
        key=key,
        maximum_context=args.maximum_context,
        backoff_penalty=args.backoff_penalty,
    )
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    count = import_teacher_rows(
        cache,
        rows,
        teacher=args.teacher,
        teacher_revision=args.teacher_revision,
    )
    cache.export(args.output)
    print(json.dumps({"status": "ok", "entries": count, "output": args.output}))
    return 0


def command_research_smoke(args: argparse.Namespace) -> int:
    examples = [
        synthetic_ranker_example(
            "えっと明日は行きません。料金は3000円です。",
            example_id="smoke-1",
        ),
        synthetic_ranker_example(
            "学校へ行って切符を買います。",
            example_id="smoke-2",
        ),
        synthetic_ranker_example(
            "スーパーでしんぶんを買った。",
            example_id="smoke-3",
        ),
    ]
    training = train_pairwise_ranker(
        examples,
        name="research-smoke",
        config=RankerTrainingConfig(epochs=100, seed=23),
    )
    decision = run_candidate_cascade(examples[0].candidates)
    payload = {
        "status": "ok",
        "claimBoundary": (
            "Synthetic deterministic training validates code paths only; "
            "it is not evidence of real-audio CER improvement."
        ),
        "training": {
            "before": asdict(training.before),
            "after": asdict(training.after),
            "profileDigest": training.profile.digest,
            "manifestDigest": training.training_manifest_sha256,
        },
        "cascade": {
            "selectedCandidateId": decision.selected_candidate_id,
            "mbrCandidateId": decision.mbr.selected_candidate_id,
            "adaptiveK": asdict(decision.adaptive_k),
            "requiresAdditionalEvidence": decision.requires_additional_evidence,
        },
    }
    _write_or_print(payload, args.output)
    return 0


def _build_ranker(args: argparse.Namespace):
    if args.ranker_backend == "none":
        return None
    if args.ranker_backend == "linear":
        if not args.ranker_profile:
            raise ValueError("--ranker-profile is required for the linear backend")
        return LinearCandidateRanker(_load_linear_profile(args.ranker_profile))
    if args.ranker_backend == "cross-encoder":
        if not args.ranker_model:
            raise ValueError("--ranker-model is required for the cross-encoder backend")
        return CrossEncoderCandidateRanker(
            args.ranker_model,
            model_revision=args.ranker_model_revision,
            model_artifact_sha256=args.ranker_model_artifact_sha256,
            runtime_revision=args.ranker_runtime_revision or args.runtime_revision,
            device=args.ranker_device,
            batch_size=args.ranker_batch_size,
        )
    if args.ranker_backend == "qwen3":
        return Qwen3CandidateRanker(
            model=args.ranker_model or "Qwen/Qwen3-Reranker-0.6B",
            model_revision=args.ranker_model_revision,
            model_artifact_sha256=args.ranker_model_artifact_sha256,
            runtime_revision=args.ranker_runtime_revision or args.runtime_revision,
            device_map=args.ranker_device,
            dtype=args.ranker_dtype,
            batch_size=args.ranker_batch_size,
        )
    raise AssertionError(args.ranker_backend)


def command_transcribe_v2(args: argparse.Namespace) -> int:
    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    policy = effort_profile(args.effort)
    maximum_hypotheses = args.maximum_hypotheses or policy.maximum_candidates
    evidence_budget_ms = (
        args.evidence_budget_ms
        if args.evidence_budget_ms is not None
        else policy.evidence_budget_ms
    )
    max_evidence_actions = (
        args.max_evidence_actions
        if args.max_evidence_actions is not None
        else policy.maximum_evidence_actions
    )
    qwen_second_ear = (
        args.qwen_second_ear if args.qwen_second_ear is not None else policy.enable_second_ear
    )

    base = PathPreservingFasterWhisperAdapter(
        model=args.model,
        model_revision=args.model_revision,
        model_artifact_sha256=args.model_artifact_sha256,
        runtime_revision=args.runtime_revision,
        device=args.device,
        compute_type=args.compute_type,
        patience=args.patience,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    ranker = _build_ranker(args)
    if ranker is not None:
        base = AdaptiveRerankingAdapter(
            base,
            ranker,
            maximum_hypotheses=maximum_hypotheses,
            calibration_profile=_load_calibration(args.ranker_calibration),
            lexical_blend=args.ranker_lexical_blend,
        )
    second_ear = (
        Qwen3ASRAdapter(
            model=args.qwen_model,
            model_revision=args.qwen_model_revision,
            model_artifact_sha256=args.qwen_model_artifact_sha256,
            runtime_revision=args.runtime_revision,
            dtype=args.qwen_dtype,
            device_map=args.qwen_device_map,
            return_timestamps=args.qwen_timestamps,
        )
        if qwen_second_ear
        else None
    )
    aligner = (
        Qwen3ForcedAlignerAdapter(
            model=args.qwen_aligner_model,
            model_revision=args.qwen_aligner_revision,
            model_artifact_sha256=args.qwen_aligner_artifact_sha256,
            runtime_revision=args.runtime_revision,
            dtype=args.qwen_dtype,
            device_map=args.qwen_device_map,
        )
        if args.qwen_aligner
        else None
    )
    teacher = None
    if args.teacher_model:
        if args.teacher_protocol == "ollama":
            teacher = OllamaRanker(
                model=args.teacher_model,
                endpoint=args.teacher_endpoint or "http://127.0.0.1:11434/api/chat",
            )
        else:
            teacher = OpenAICompatibleRanker(
                model=args.teacher_model,
                endpoint=args.teacher_endpoint or "http://127.0.0.1:8000/v1/chat/completions",
            )
    cache = None if args.no_cache else EvidenceCache(args.cache)
    try:
        transcriber = SemanticASRTranscriber(
            base,
            second_ear=second_ear,
            forced_aligner=aligner,
            teacher=teacher,
            cache=cache,
            evidence_budget=EvidenceBudget(
                total_cost_ms=evidence_budget_ms,
                max_actions=max_evidence_actions,
            ),
            balanced_router=args.evidence_router == "balanced",
            router_state=_load_router_state(args.router_state),
            window_ms=args.window_ms,
            overlap_ms=args.overlap_ms,
        )
        result = transcriber.transcribe(
            source,
            duration_ms=args.duration_ms,
            language=None if args.language == "auto" else args.language,
            initial_prompt=args.initial_prompt,
            hotwords=_hotwords(args),
            context=args.context,
        )
        formats = (
            {value.strip() for value in args.formats.split(",") if value.strip()}
            if args.formats != "all"
            else None
        )
        outputs = write_outputs(
            result,
            args.output_dir,
            overwrite=args.overwrite,
            formats=formats,
        )
    finally:
        if cache is not None:
            cache.close()
    print(
        json.dumps(
            {
                "status": "ok",
                "runtime": "v0.2-path-pool-adaptive-reranking",
                "effort": asdict(policy),
                "resolvedMaximumHypotheses": maximum_hypotheses,
                "resolvedEvidenceBudgetMs": evidence_budget_ms,
                "resolvedMaximumEvidenceActions": max_evidence_actions,
                "resolvedQwenSecondEar": qwen_second_ear,
                "outputs": outputs,
                "diagnostics": result.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_advanced_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-asr",
        description="Semantic ASR v0.2 research and adaptive runtime commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cascade = commands.add_parser("cascade")
    cascade.add_argument("input")
    cascade.add_argument("--output")
    cascade.add_argument(
        "--selection-policy",
        choices=["fusion", "mbr-tiebreak"],
        default="fusion",
    )
    cascade.add_argument("--minimum-k", type=int, default=2)
    cascade.add_argument("--maximum-k", type=int, default=12)
    cascade.add_argument("--posterior-mass", type=float, default=0.94)
    cascade.add_argument("--semantic-criticality", type=float, default=0.0)
    cascade.set_defaults(func=command_cascade)

    train = commands.add_parser("train-ranker")
    train.add_argument("input")
    train.add_argument("--output", required=True)
    train.add_argument("--name", default="semantic-asr-linear-v0.2")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--l2", type=float, default=0.002)
    train.add_argument("--seed", type=int, default=17)
    train.set_defaults(func=command_train_ranker)

    calibrate = commands.add_parser("calibrate-ranker")
    calibrate.add_argument("input")
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--name", default="semantic-asr-ranker-calibration-v0.2")
    calibrate.add_argument("--source-ranker", required=True)
    calibrate.add_argument("--l2", type=float, default=0.001)
    calibrate.add_argument("--maximum-iterations", type=int, default=100)
    calibrate.add_argument("--minimum-samples", type=int, default=8)
    calibrate.add_argument("--minimum-groups", type=int, default=2)
    calibrate.set_defaults(func=command_calibrate_ranker)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("input")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--ks", default="1,3,5,8,12,16,25,50")
    benchmark.add_argument("--bootstrap-iterations", type=int, default=2000)
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument(
        "--allow-non-test",
        action="store_true",
        help="Development only; final reports must use a locked test split.",
    )
    benchmark.set_defaults(func=command_benchmark)

    distill = commands.add_parser("distill-teachers")
    distill.add_argument("input")
    distill.add_argument("--output", required=True)
    distill.add_argument("--rejected-output")
    distill.add_argument("--temperature", type=float, default=1.0)
    distill.add_argument("--minimum-active-teachers", type=int, default=1)
    distill.add_argument("--maximum-teacher-share", type=float, default=0.60)
    distill.add_argument("--maximum-disagreement", type=float, default=0.42)
    distill.set_defaults(func=command_distill_teachers)

    synthetic = commands.add_parser("synthetic-data")
    synthetic.add_argument("input")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--maximum-negatives", type=int, default=8)
    synthetic.add_argument("--seed", type=int, default=17)
    synthetic.add_argument("--context", default="")
    synthetic.set_defaults(func=command_synthetic_data)

    cache = commands.add_parser("lm-cache-build")
    cache.add_argument("input")
    cache.add_argument("--output", required=True)
    cache.add_argument("--key-hex", required=True)
    cache.add_argument("--teacher", required=True)
    cache.add_argument("--teacher-revision")
    cache.add_argument("--maximum-context", type=int, default=8)
    cache.add_argument("--backoff-penalty", type=float, default=0.35)
    cache.set_defaults(func=command_lm_cache_build)

    smoke = commands.add_parser("research-smoke")
    smoke.add_argument("--output")
    smoke.set_defaults(func=command_research_smoke)

    transcribe = commands.add_parser("transcribe-v2")
    transcribe.add_argument("audio")
    transcribe.add_argument("-o", "--output-dir", default="transcripts")
    transcribe.add_argument("--overwrite", action="store_true")
    transcribe.add_argument("--duration-ms", type=int)
    transcribe.add_argument("--language", default="ja")
    transcribe.add_argument("--model", default="large-v3-turbo")
    transcribe.add_argument("--model-revision")
    transcribe.add_argument("--model-artifact-sha256")
    transcribe.add_argument("--runtime-revision")
    transcribe.add_argument("--device", default="auto")
    transcribe.add_argument("--compute-type", default="default")
    transcribe.add_argument("--window-ms", type=int, default=28_000)
    transcribe.add_argument("--overlap-ms", type=int, default=1_200)
    transcribe.add_argument(
        "--effort",
        choices=["ultra-light", "cpu-quality", "edge-gpu", "research"],
        default="cpu-quality",
    )
    transcribe.add_argument("--maximum-hypotheses", type=int)
    transcribe.add_argument("--patience", type=float, default=1.4)
    transcribe.add_argument("--repetition-penalty", type=float, default=1.0)
    transcribe.add_argument("--no-repeat-ngram-size", type=int, default=0)
    transcribe.add_argument(
        "--ranker-backend",
        choices=["none", "linear", "cross-encoder", "qwen3"],
        default="none",
    )
    transcribe.add_argument("--ranker-profile")
    transcribe.add_argument("--ranker-calibration")
    transcribe.add_argument("--ranker-model")
    transcribe.add_argument(
        "--ranker-model-revision",
        "--ranker-revision",
        dest="ranker_model_revision",
    )
    transcribe.add_argument(
        "--ranker-model-artifact-sha256",
        "--ranker-artifact-sha256",
        dest="ranker_model_artifact_sha256",
    )
    transcribe.add_argument("--ranker-runtime-revision")
    transcribe.add_argument("--ranker-device", default="cpu")
    transcribe.add_argument("--ranker-dtype", default="auto")
    transcribe.add_argument("--ranker-batch-size", type=int, default=8)
    transcribe.add_argument("--ranker-lexical-blend", type=float, default=0.65)
    transcribe.add_argument("--initial-prompt")
    transcribe.add_argument("--hotwords")
    transcribe.add_argument("--hotwords-file")
    transcribe.add_argument("--context", default="")
    transcribe.add_argument("--cache", default=".semantic-asr/evidence-cache-v2.sqlite3")
    transcribe.add_argument("--no-cache", action="store_true")
    transcribe.add_argument("--evidence-budget-ms", type=int)
    transcribe.add_argument("--max-evidence-actions", type=int)
    transcribe.add_argument(
        "--evidence-router",
        choices=["legacy", "balanced"],
        default="balanced",
    )
    transcribe.add_argument("--router-state")
    transcribe.add_argument(
        "--qwen-second-ear",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    transcribe.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-0.6B")
    transcribe.add_argument("--qwen-model-revision")
    transcribe.add_argument("--qwen-model-artifact-sha256")
    transcribe.add_argument("--qwen-device-map", default="cuda:0")
    transcribe.add_argument("--qwen-dtype", default="float16")
    transcribe.add_argument("--qwen-timestamps", action="store_true")
    transcribe.add_argument("--qwen-aligner", action="store_true")
    transcribe.add_argument(
        "--qwen-aligner-model",
        default="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    transcribe.add_argument("--qwen-aligner-revision")
    transcribe.add_argument("--qwen-aligner-artifact-sha256")
    transcribe.add_argument("--teacher-model")
    transcribe.add_argument(
        "--teacher-protocol",
        choices=["ollama", "openai"],
        default="ollama",
    )
    transcribe.add_argument("--teacher-endpoint")
    transcribe.add_argument(
        "--formats",
        default="all",
        help="json,observed,normalized,md,srt,vtt or all",
    )
    transcribe.set_defaults(func=command_transcribe_v2)
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values or values[0] not in _ADVANCED_COMMANDS:
        return legacy_main(values)
    args = build_advanced_parser().parse_args(values)
    return int(args.func(args))
