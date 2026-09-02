"""Run the post-candidate research pipeline on already generated N-best manifests.

The steps mirror ``docs/REAL_AUDIO_EXPERIMENTS.md`` sections 3-7 and the self-hosted
workflow: merge chunked candidate files, partition, train a ranker, fit held-out calibration,
apply it to the locked test split, and benchmark. Every step is executed through the public
CLI so the run is reproducible from the documented commands.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="glob of candidate JSONL files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ranker", choices=["listwise", "pairwise"], default="listwise")
    parser.add_argument("--ks", default="1,3,5,8,12,16,25,50")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--cli", default="semantic-asr")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged = out / "all-candidates.jsonl"
    files = sorted(glob.glob(args.candidates))
    if not files:
        raise SystemExit(f"no candidate files match {args.candidates}")
    with merged.open("w", encoding="utf-8") as handle:
        for path in files:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    handle.write(line + "\n")
    print(json.dumps({"mergedFiles": len(files), "merged": str(merged)}))

    cli = [args.cli]
    run([*cli, "partition-manifest", str(merged), "--output-dir", str(out / "splits")])
    train_command = "train-listwise-ranker" if args.ranker == "listwise" else "train-ranker"
    ranker = out / "ranker.json"
    run([*cli, train_command, str(out / "splits" / "train.jsonl"), "--output", str(ranker)])
    scores = out / "calibration-scores.jsonl"
    run(
        [
            *cli,
            "score-ranker-calibration",
            str(out / "splits" / "calibration.jsonl"),
            "--ranker-profile",
            str(ranker),
            "--output",
            str(scores),
        ]
    )
    profile_name = json.loads(ranker.read_text(encoding="utf-8"))["profile"]["name"]
    calibration = out / "calibration.json"
    run(
        [
            *cli,
            "calibrate-ranker",
            str(scores),
            "--source-ranker",
            profile_name,
            "--output",
            str(calibration),
        ]
    )
    reranked = out / "test-reranked.jsonl"
    run(
        [
            *cli,
            "apply-ranker",
            str(out / "splits" / "test.jsonl"),
            "--ranker-profile",
            str(ranker),
            "--calibration",
            str(calibration),
            "--output",
            str(reranked),
        ]
    )
    run(
        [
            *cli,
            "benchmark",
            str(reranked),
            "--output",
            str(out / "report.json"),
            "--ks",
            args.ks,
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ]
    )
    run(
        [
            *cli,
            "benchmark",
            str(out / "splits" / "test.jsonl"),
            "--output",
            str(out / "report-raw.json"),
            "--ks",
            args.ks,
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
