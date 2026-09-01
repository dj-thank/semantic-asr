from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .listwise_training import (
    ListwiseTrainingConfig,
    train_listwise_semantic_mwer,
    write_listwise_training_result,
)
from .ranker_dataset import load_ranker_examples
from .ranker_training import (
    RankerTrainingConfig,
    train_pairwise_ranker,
    write_training_result,
)

TRAINING_COMMANDS = {"train-ranker", "train-listwise-ranker"}


def command_train_pairwise(args: argparse.Namespace) -> int:
    examples = load_ranker_examples(args.input, require_train_split=True)
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
                "objective": "pairwise-preference",
                "output": str(Path(args.output)),
                "examples": result.example_count,
                "before": asdict(result.before),
                "after": asdict(result.after),
                "profileDigest": result.profile.digest,
                "trainingManifestSha256": result.training_manifest_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_train_listwise(args: argparse.Namespace) -> int:
    examples = load_ranker_examples(args.input, require_train_split=True)
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
                "objective": "listwise-semantic-mwer",
                "output": str(Path(args.output)),
                "examples": result.example_count,
                "before": asdict(result.before),
                "after": asdict(result.after),
                "profileDigest": result.profile.digest,
                "trainingManifestSha256": result.training_manifest_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-asr",
        description="Leakage-safe Semantic ASR candidate-ranker training commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pairwise = commands.add_parser("train-ranker")
    pairwise.add_argument("input")
    pairwise.add_argument("--output", required=True)
    pairwise.add_argument("--name", default="semantic-asr-linear-v0.2")
    pairwise.add_argument("--epochs", type=int, default=80)
    pairwise.add_argument("--learning-rate", type=float, default=0.08)
    pairwise.add_argument("--l2", type=float, default=0.002)
    pairwise.add_argument("--seed", type=int, default=17)
    pairwise.set_defaults(func=command_train_pairwise)

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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
