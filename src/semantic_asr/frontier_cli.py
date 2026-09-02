from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .advanced_adapters import LoopGuardConfig, PathPreservingFasterWhisperAdapter
from .deployment_gate import (
    DeploymentGatePolicy,
    deployment_evaluation_from_dict,
    evaluate_deployment_candidate,
)
from .enrichment import EnrichmentConfig, enrich_manifest_rows, load_second_ear
from .experiment_runner import (
    CandidateGenerationConfig,
    finalize_generated_checkpoint,
    generate_manifest_to_checkpoint,
    load_audio_manifest,
)
from .fusion_io import load_fusion_examples, write_learned_fusion_result
from .learned_fusion import LearnedFusionConfig, train_constrained_fusion
from .listwise_training import (
    ListwiseTrainingConfig,
    train_listwise_semantic_mwer,
    write_listwise_training_result,
)
from .ngram import NGramLanguageModel
from .pipeline import effort_profile
from .ranker_training import load_jsonl_examples
from .revisions import FASTER_WHISPER_MODEL_REVISIONS, resolve_hugging_face_revision
from .throttling import RuntimePressure, ThrottleState, throttle_effort

FRONTIER_COMMANDS = {
    "deployment-gate",
    "enrich-candidates",
    "generate-candidates",
    "throttle-policy",
    "train-fusion",
    "train-listwise-ranker",
    "train-ngram",
}


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def command_train_ngram(args: argparse.Namespace) -> int:
    source = Path(args.input)
    texts = [
        line.strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not texts:
        raise ValueError("n-gram corpus is empty")
    model = NGramLanguageModel(
        order=args.order,
        mode=args.mode,
        alpha=args.alpha,
        lowercase_ascii=not args.preserve_ascii_case,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_revision=args.source_revision,
    ).fit(texts)
    model.save(args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "documents": model.document_count,
                "tokens": model.token_count,
                "vocabulary": len(model.vocabulary),
                "order": model.order,
                "mode": model.mode,
                "digest": model.digest,
                "sourceSha256": model.source_sha256,
                "sourceRevision": model.source_revision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_train_listwise(args: argparse.Namespace) -> int:
    examples = load_jsonl_examples(args.input)
    result = train_listwise_semantic_mwer(
        examples,
        name=args.name,
        config=ListwiseTrainingConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            temperature=args.temperature,
            gradient_clip=args.gradient_clip,
            seed=args.seed,
        ),
    )
    write_listwise_training_result(result, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "examples": result.example_count,
                "before": asdict(result.before),
                "after": asdict(result.after),
                "profileDigest": result.profile.digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_train_fusion(args: argparse.Namespace) -> int:
    examples = load_fusion_examples(args.input)
    initial_weights = None
    if args.initial_profile:
        profile = _json(args.initial_profile).get("profile", _json(args.initial_profile))
        initial_weights = dict(profile["weights"])
    result = train_constrained_fusion(
        examples,
        name=args.name,
        initial_weights=initial_weights,
        config=LearnedFusionConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2_to_initial=args.l2_to_initial,
            temperature=args.temperature,
            acoustic_family_floor=args.acoustic_family_floor,
            seed=args.seed,
        ),
    )
    write_learned_fusion_result(result, args.output)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(Path(args.output)),
                "before": asdict(result.before),
                "after": asdict(result.after),
                "weights": result.profile.weights,
                "acousticFamilyFloor": result.profile.acoustic_family_floor,
                "profileDigest": result.profile.digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_enrich_candidates(args: argparse.Namespace) -> int:
    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    second_ear = (
        load_second_ear(
            args.second_ear,
            source=args.second_ear_source,
            model_revision=args.second_ear_revision,
        )
        if args.second_ear
        else {}
    )
    ngram_model = NGramLanguageModel.load(args.ngram) if args.ngram else None
    config = EnrichmentConfig(
        add_second_ear_candidate=args.add_second_ear_candidate,
        second_ear_source=args.second_ear_source,
        ngram_model=ngram_model,
        ngram_name=(
            f"{ngram_model.mode}-{ngram_model.order}gram:{ngram_model.digest[:12]}"
            if ngram_model is not None
            else "ngram"
        ),
    )
    enriched = enrich_manifest_rows(rows, second_ear=second_ear, config=config)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in enriched)
        + ("\n" if enriched else ""),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(enriched),
                "secondEarRows": sum(1 for row in rows if str(row.get("sampleId")) in second_ear),
                "ngram": config.ngram_name if ngram_model is not None else None,
                "output": str(target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_generate_candidates(args: argparse.Namespace) -> int:
    records = load_audio_manifest(args.input)
    model_revision = resolve_hugging_face_revision(
        args.model,
        args.model_revision,
        FASTER_WHISPER_MODEL_REVISIONS,
    )
    fallback_temperatures = tuple(
        float(value) for value in str(args.fallback_temperatures or "").split(",") if value.strip()
    )
    adapter = PathPreservingFasterWhisperAdapter(
        model=args.model,
        model_revision=model_revision,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        length_penalty=args.length_penalty,
        patience=args.patience,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        loop_guard=LoopGuardConfig(
            enabled=not args.no_loop_guard,
            max_tokens_per_second=args.max_tokens_per_second,
            max_tokens_floor=args.max_tokens_floor,
            compression_ratio_threshold=args.compression_ratio_threshold,
            log_prob_threshold=args.log_prob_threshold,
            fallback_temperatures=fallback_temperatures,
            fallback_samples=args.fallback_samples,
            drop_degenerate=not args.keep_degenerate,
            max_characters_per_second=args.max_characters_per_second,
            extra_samples=args.extra_samples,
            extra_sample_temperature=args.extra_sample_temperature,
            extra_sample_topk=args.extra_sample_topk,
        ),
        without_timestamps=args.without_timestamps,
    )
    hotwords = tuple(
        value.strip()
        for value in (args.hotwords or "").replace("、", ",").split(",")
        if value.strip()
    )
    config = CandidateGenerationConfig(
        language=None if args.language == "auto" else args.language,
        beam_size=args.beam_size,
        hypotheses=args.hypotheses,
        initial_prompt=args.initial_prompt,
        hotwords=hotwords,
        return_timestamps=args.return_timestamps,
        fail_on_non_allow_rights=not args.allow_review_rights,
        model_revision=model_revision,
        runtime_revision=args.runtime_revision,
    )
    output = Path(args.output)
    checkpoint = Path(str(output) + ".partial")
    generated = generate_manifest_to_checkpoint(
        records,
        adapter,
        config=config,
        checkpoint_path=checkpoint,
        progress=lambda completed, total: print(
            json.dumps(
                {"generatedCandidates": completed, "totalCandidates": total},
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        ),
    )
    if len(generated) != len(records):
        raise RuntimeError("generated checkpoint is incomplete")
    finalize_generated_checkpoint(
        checkpoint,
        output_path=output,
        ranker_path=args.ranker_output,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(records),
                "groups": len({record.group_id for record in records}),
                "output": str(Path(args.output)),
                "rankerOutput": args.ranker_output,
                "adapter": adapter.name,
                "model": adapter.model_name,
                "generationConfigSha256": config.digest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_deployment_gate(args: argparse.Namespace) -> int:
    baseline = deployment_evaluation_from_dict(_json(args.baseline))
    candidate = deployment_evaluation_from_dict(_json(args.candidate))
    policy = DeploymentGatePolicy(**_json(args.policy)) if args.policy else DeploymentGatePolicy()
    decision = evaluate_deployment_candidate(baseline, candidate, policy=policy)
    payload = asdict(decision)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if decision.accepted else 2


def command_throttle_policy(args: argparse.Namespace) -> int:
    profile = effort_profile(args.effort)
    previous = ThrottleState(
        level=args.previous_level,
        stable_steps=args.previous_stable_steps,
    )
    pressure = RuntimePressure(
        latency_ratio=args.latency_ratio,
        memory_pressure=args.memory_pressure,
        queue_pressure=args.queue_pressure,
        thermal_pressure=args.thermal_pressure,
        battery_saver=args.battery_saver,
    )
    decision = throttle_effort(profile, pressure, previous=previous)
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-asr",
        description=(
            "Semantic ASR frontier training, deployment, and real-audio experiment commands."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ngram = commands.add_parser("train-ngram")
    ngram.add_argument("input", help="UTF-8 text corpus, one document per line")
    ngram.add_argument("--output", required=True)
    ngram.add_argument(
        "--mode",
        choices=["character", "mora", "subword", "whitespace"],
        default="character",
    )
    ngram.add_argument("--order", type=int, default=5)
    ngram.add_argument("--alpha", type=float, default=0.1)
    ngram.add_argument("--preserve-ascii-case", action="store_true")
    ngram.add_argument("--source-revision")
    ngram.set_defaults(func=command_train_ngram)

    listwise = commands.add_parser("train-listwise-ranker")
    listwise.add_argument("input")
    listwise.add_argument("--output", required=True)
    listwise.add_argument("--name", default="semantic-asr-listwise-mwer-v0.2")
    listwise.add_argument("--epochs", type=int, default=160)
    listwise.add_argument("--learning-rate", type=float, default=0.05)
    listwise.add_argument("--l2", type=float, default=0.002)
    listwise.add_argument("--temperature", type=float, default=1.0)
    listwise.add_argument("--gradient-clip", type=float, default=5.0)
    listwise.add_argument("--seed", type=int, default=29)
    listwise.set_defaults(func=command_train_listwise)

    fusion = commands.add_parser("train-fusion")
    fusion.add_argument("input")
    fusion.add_argument("--output", required=True)
    fusion.add_argument("--initial-profile")
    fusion.add_argument("--name", default="semantic-asr-constrained-fusion-v0.2")
    fusion.add_argument("--epochs", type=int, default=200)
    fusion.add_argument("--learning-rate", type=float, default=0.08)
    fusion.add_argument("--l2-to-initial", type=float, default=0.02)
    fusion.add_argument("--temperature", type=float, default=0.18)
    fusion.add_argument("--acoustic-family-floor", type=float, default=0.72)
    fusion.add_argument("--seed", type=int, default=37)
    fusion.set_defaults(func=command_train_fusion)

    enrich = commands.add_parser("enrich-candidates")
    enrich.add_argument("input", help="candidate JSONL from generate-candidates")
    enrich.add_argument("--output", required=True)
    enrich.add_argument("--second-ear", help="probe_second_ear.py JSONL output")
    enrich.add_argument("--second-ear-source", default="qwen3-asr")
    enrich.add_argument("--second-ear-revision")
    enrich.add_argument(
        "--add-second-ear-candidate",
        action="store_true",
        help="Append the second-ear hypothesis as an additional acoustically grounded candidate.",
    )
    enrich.add_argument("--ngram", help="NGramLanguageModel JSON from train-ngram")
    enrich.set_defaults(func=command_enrich_candidates)

    generate = commands.add_parser("generate-candidates")
    generate.add_argument("input", help="rights-gated audio manifest JSONL")
    generate.add_argument("--output", required=True, help="benchmark JSONL")
    generate.add_argument("--ranker-output")
    generate.add_argument("--model", default="large-v3-turbo")
    generate.add_argument("--model-revision")
    generate.add_argument("--runtime-revision")
    generate.add_argument("--device", default="auto")
    generate.add_argument("--compute-type", default="default")
    generate.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CTranslate2 CPU thread count; use a positive fixed value for reproducible runs.",
    )
    generate.add_argument("--language", default="ja")
    generate.add_argument("--beam-size", type=int, default=12)
    generate.add_argument("--hypotheses", type=int, default=12)
    generate.add_argument("--length-penalty", type=float, default=1.0)
    generate.add_argument("--patience", type=float, default=1.4)
    generate.add_argument("--repetition-penalty", type=float, default=1.0)
    generate.add_argument("--no-repeat-ngram-size", type=int, default=0)
    generate.add_argument("--initial-prompt")
    generate.add_argument("--hotwords")
    generate.add_argument("--return-timestamps", action="store_true")
    generate.add_argument(
        "--no-loop-guard",
        action="store_true",
        help="Disable the duration-aware token budget, degeneracy check, and sampled fallback.",
    )
    generate.add_argument("--max-tokens-per-second", type=float, default=14.0)
    generate.add_argument("--max-tokens-floor", type=int, default=32)
    generate.add_argument("--compression-ratio-threshold", type=float, default=2.4)
    generate.add_argument("--log-prob-threshold", type=float, default=-1.0)
    generate.add_argument(
        "--fallback-temperatures",
        default="0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated sampling temperatures tried when the beam stage is degenerate.",
    )
    generate.add_argument("--fallback-samples", type=int, default=5)
    generate.add_argument("--max-characters-per-second", type=float, default=12.0)
    generate.add_argument(
        "--extra-samples",
        type=int,
        default=0,
        help="Always add this many sampled hypotheses (own score domain) for sample-based MBR.",
    )
    generate.add_argument("--extra-sample-temperature", type=float, default=1.0)
    generate.add_argument("--extra-sample-topk", type=int, default=0)
    generate.add_argument(
        "--without-timestamps",
        action="store_true",
        help="Decode with <|notimestamps|> (the v0.2 behaviour); loops far more on short clips.",
    )
    generate.add_argument(
        "--keep-degenerate",
        action="store_true",
        help="Keep degenerate paths in the pool (demoted) instead of dropping them.",
    )
    generate.add_argument(
        "--allow-review-rights",
        action="store_true",
        help="Development only; production experiments should fail closed.",
    )
    generate.set_defaults(func=command_generate_candidates)

    gate = commands.add_parser("deployment-gate")
    gate.add_argument("baseline")
    gate.add_argument("candidate")
    gate.add_argument("--policy")
    gate.add_argument("--output")
    gate.set_defaults(func=command_deployment_gate)

    throttle = commands.add_parser("throttle-policy")
    throttle.add_argument(
        "--effort",
        choices=["ultra-light", "cpu-quality", "edge-gpu", "research"],
        default="cpu-quality",
    )
    throttle.add_argument("--latency-ratio", type=float, default=0.0)
    throttle.add_argument("--memory-pressure", type=float, default=0.0)
    throttle.add_argument("--queue-pressure", type=float, default=0.0)
    throttle.add_argument("--thermal-pressure", type=float, default=0.0)
    throttle.add_argument("--battery-saver", action="store_true")
    throttle.add_argument("--previous-level", type=int, default=0)
    throttle.add_argument("--previous-stable-steps", type=int, default=0)
    throttle.set_defaults(func=command_throttle_policy)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
