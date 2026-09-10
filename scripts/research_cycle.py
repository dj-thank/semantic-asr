"""Finite fixed-trial post-candidate cycle using existing training and score contracts.

This is stage resumption, not optimizer resumption. A trial interrupted during
training is not restarted beyond max_trials. Development model selection and
previously unseen publication evaluation await the separate dataset contracts.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codex_pipeline import (
    CLI,
    BudgetExceeded,
    enforce_budget,
    environment,
    files_under,
    positive,
    run_stage,
    sha256,
    source_identity,
    write_json,
)
from run_real_audio_pipeline import (
    _read_candidate_files,
    ensure_safe_output_dir,
)

STAGES = (
    "provision",
    "fit-train",
    "score-calibration",
    "calibrate",
    "freeze",
    "select",
    "evaluate",
    "classify-errors",
    "report",
)
OUTPUTS = {
    "provision": (
        "train.jsonl",
        "calibration.jsonl",
        "test-inference.jsonl",
        "test-reference.jsonl",
    ),
    "fit-train": ("ranker.json",),
    "score-calibration": ("calibration-scores.jsonl",),
    "calibrate": ("calibration.json",),
    "freeze": ("freeze.json",),
    "select": ("decisions.jsonl",),
    "evaluate": ("paired-report.json", "error-counts.jsonl"),
    "classify-errors": ("error-taxonomy.json",),
    "report": ("report.json", "report-raw.json"),
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values),
        encoding="utf-8",
    )
    temporary.replace(path)


def select_candidates(candidates, ranker, calibration) -> dict[str, Any]:
    """Inference accepts candidates only; evaluation references cannot enter here."""
    from semantic_asr.benchmark import BenchmarkUtterance, _ordered_candidates
    from semantic_asr.cascade import CascadeConfig, run_candidate_cascade
    from semantic_asr.offline_rerank import rerank_record

    # Preserve score-domain provenance, but reject known supervised metadata.
    def reject_supervision(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).casefold().replace("_", "") in {
                    "reference",
                    "annotatedreference",
                    "referencereading",
                    "losses",
                    "correct",
                    "gold",
                    "target",
                    "context",
                }:
                    raise ValueError("supervised metadata is forbidden in inference candidates")
                reject_supervision(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_supervision(item)

    for candidate in candidates:
        reject_supervision(candidate.metadata)
    # The legacy adapter requires a benchmark record. Its constant placeholder
    # contains no evaluation text and never contributes to scoring or selection.
    record = BenchmarkUtterance(
        "inference",
        "inference",
        "inference",
        "test",
        "[not-an-evaluation-reference]",
        tuple(candidates),
    )
    enhanced = rerank_record(record, ranker, calibration=calibration)
    ordered = _ordered_candidates(enhanced)
    decision = run_candidate_cascade(
        ordered, cascade_config=CascadeConfig(selection_policy="fusion")
    )
    baseline = _ordered_candidates(record)[0]
    selected = next(
        r.candidate
        for r in decision.ranked
        if r.candidate.candidate_id == decision.selected_candidate_id
    )
    return {
        "baseline_id": baseline.candidate_id,
        "selected_id": selected.candidate_id,
        "baseline_text": baseline.text,
        "selected_text": selected.text,
        "candidate_texts": [c.text for c in candidates],
        "requires_additional_evidence": decision.requires_additional_evidence,
    }


def verify_freeze(output: Path) -> dict[str, Any]:
    config = read_json(output / "config.json")
    frozen = read_json(output / "freeze.json")
    if frozen.get("config_digest") != digest(config):
        raise ValueError("freeze configuration mismatch")
    required = ("ranker.json", "calibration.json", *OUTPUTS["provision"])
    if set(frozen.get("artifacts", {})) != set(required):
        raise ValueError("freeze artifact set mismatch")
    for name in required:
        if sha256(output / name) != frozen["artifacts"][name]:
            raise ValueError("frozen artifact changed: " + name)
    return frozen


def provision(output: Path, config: dict[str, Any]) -> None:
    from semantic_asr.benchmark import benchmark_utterance_from_row, verify_split_isolation

    all_rows = []
    for item in config["inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("candidate input changed")
        all_rows.extend(rows(Path(item["path"])))
    records = [benchmark_utterance_from_row(row) for row in all_rows]
    if len({r.sample_id for r in records}) != len(records):
        raise ValueError("duplicate sample ID")
    verify_split_isolation(records)
    if {r.split for r in records} != {"train", "calibration", "test"}:
        raise ValueError("nonempty train/calibration/test are required; dev is not supported yet")
    for split in ("train", "calibration"):
        # Retain original rights, generation and lineage fields in the derived file.
        write_rows(output / f"{split}.jsonl", [r for r in all_rows if r["split"] == split])
    inference, references = [], []
    for record in records:
        if record.split != "test":
            continue
        inference.append(
            {"sampleId": record.sample_id, "candidates": [c.as_dict() for c in record.candidates]}
        )
        references.append(
            {
                "sampleId": record.sample_id,
                "groupId": record.group_id,
                "reference": record.reference,
                "annotatedReference": record.annotated_reference,
            }
        )
    write_rows(output / "test-inference.jsonl", inference)
    write_rows(output / "test-reference.jsonl", references)


def select(output: Path) -> None:
    from semantic_asr.contracts import CandidateEvidence
    from semantic_asr.experiment_cli import _calibration, _linear_ranker

    verify_freeze(output)
    ranker, calibration = (
        _linear_ranker(output / "ranker.json"),
        _calibration(output / "calibration.json"),
    )
    decisions = []
    for row in rows(output / "test-inference.jsonl"):
        if set(row) != {"sampleId", "candidates"}:
            raise ValueError("inference rows must not carry reference or context fields")
        result = select_candidates(
            tuple(CandidateEvidence.from_dict(c) for c in row["candidates"]), ranker, calibration
        )
        decisions.append({"sampleId": row["sampleId"], **result})
    write_rows(output / "decisions.jsonl", decisions)


def evaluate(output: Path, config: dict[str, Any]) -> None:
    from semantic_asr.evaluation import (
        edit_distance,
        exact_cer,
        normalize_characters,
        normalize_characters_lenient,
        spoken_reference_surface,
    )
    from semantic_asr.experiment import PairedErrorCounts, paired_error_rate_comparison

    frozen = verify_freeze(output)
    decisions = rows(output / "decisions.jsonl")
    references = rows(output / "test-reference.jsonl")
    expected = frozen["expected_sample_ids"]
    if [r["sampleId"] for r in decisions] != expected or [
        r["sampleId"] for r in references
    ] != expected:
        raise ValueError("evaluation cohort or order changed")
    counts = []
    reports = {}
    for name, normalize in (
        ("strict_cer", normalize_characters),
        ("lenient_cer", normalize_characters_lenient),
    ):
        paired = []
        for reference_row, decision in zip(references, decisions, strict=True):
            raw = reference_row["reference"]
            annotation = reference_row["annotatedReference"]
            if exact_cer(raw, decision["baseline_text"], annotated_reference=annotation) is None:
                raise ValueError("unsafe or missing evaluation reference; no silent exclusion")
            reference = normalize(
                spoken_reference_surface(annotation) if annotation is not None else raw
            )
            if not reference:
                raise ValueError("empty evaluation denominator")
            base = edit_distance(reference, normalize(decision["baseline_text"]))
            candidate = edit_distance(reference, normalize(decision["selected_text"]))
            paired.append(
                PairedErrorCounts(
                    reference_row["sampleId"],
                    reference_row["groupId"],
                    len(reference),
                    base,
                    candidate,
                )
            )
            if name == "strict_cer":
                counts.append(
                    {
                        **asdict(paired[-1]),
                        "changed": decision["baseline_text"] != decision["selected_text"],
                        "oracle_errors": min(
                            edit_distance(reference, normalize(text))
                            for text in decision["candidate_texts"]
                        ),
                        "provisional": decision["requires_additional_evidence"],
                    }
                )
        comparison = paired_error_rate_comparison(
            paired,
            baseline_system="unchanged-asr-top1",
            candidate_system="calibrated-fusion",
            metric=name,
            iterations=config["bootstrap_iterations"],
            seed=config["seed"],
            expected_sample_ids=expected,
        )
        reports[name] = asdict(comparison)
        reports[name]["utterance_mean_baseline"] = sum(
            r.baseline_errors / r.reference_units for r in paired
        ) / len(paired)
        reports[name]["utterance_mean_candidate"] = sum(
            r.candidate_errors / r.reference_units for r in paired
        ) / len(paired)
    reports.update(
        sample_count=len(expected),
        evaluation_role=config["evaluation_role"],
        promotion="not-evaluated",
        new_acoustic_or_lora_weights=False,
        common_candidate_budget=True,
        identical_total_compute_cost=False,
        group_independence="unverified",
        config_digest=digest(config),
        false_corrections=sum(
            r["baseline_errors"] == 0 and r["candidate_errors"] > 0 for r in counts
        ),
        improved=sum(r["candidate_errors"] < r["baseline_errors"] for r in counts),
        harmed=sum(r["candidate_errors"] > r["baseline_errors"] for r in counts),
        ties=sum(r["candidate_errors"] == r["baseline_errors"] for r in counts),
        provisional=sum(r["provisional"] for r in counts),
    )
    write_rows(output / "error-counts.jsonl", counts)
    write_json(output / "paired-report.json", reports)


def worker(stage: str, output: Path) -> None:
    config = read_json(output / "config.json")
    if stage == "provision":
        provision(output, config)
    elif stage == "freeze":
        names = ("ranker.json", "calibration.json", *OUTPUTS["provision"])
        write_json(
            output / "freeze.json",
            {
                "schema": "research-freeze-v1",
                "config_digest": digest(config),
                "artifacts": {n: sha256(output / n) for n in names},
                "expected_sample_ids": [
                    r["sampleId"] for r in rows(output / "test-inference.jsonl")
                ],
            },
        )
    elif stage == "select":
        select(output)
    elif stage == "evaluate":
        evaluate(output, config)
    elif stage == "classify-errors":
        values = rows(output / "error-counts.jsonl")
        write_json(
            output / "error-taxonomy.json",
            {
                "schema": "research-error-taxonomy-v1",
                "audio_listened": False,
                "items": [
                    {
                        "sample_id": r["sample_id"],
                        "classification": (
                            "candidate-coverage" if r["oracle_errors"] > 0 else "selection"
                        ),
                        "cause": "unknown",
                        "errors": r["candidate_errors"],
                    }
                    for r in values
                    if r["candidate_errors"] > 0
                ],
                "unmeasured": [
                    "segmentation",
                    "G2P",
                    "acoustic",
                    "orthographic",
                    "gold-phone-mora",
                    "semantic-critical",
                ],
            },
        )
    elif stage == "report":
        report = read_json(output / "paired-report.json")
        # Compatibility filenames explicitly report corpus CER, not legacy utterance means.
        for name, key in (("report.json", "candidate_mean"), ("report-raw.json", "baseline_mean")):
            write_json(
                output / name,
                {
                    "schema": "bounded-research-summary-v1",
                    "sample_count": report["sample_count"],
                    "aggregation": "corpus-error-rate",
                    "baseline_cer": report["strict_cer"]["baseline_mean"],
                    "cascade_cer": report["strict_cer"][key],
                    "mbr_cer": None,
                    "promotion": "not-evaluated",
                    "paired_report": "paired-report.json",
                },
            )
    else:
        raise ValueError("unknown worker stage")


def commands(stage: str, output: Path, config: dict[str, Any]) -> list[str]:
    if stage == "fit-train":
        command = "train-ranker" if config["ranker"] == "pairwise" else "train-listwise-ranker"
        return [
            *CLI,
            command,
            str(output / "train.jsonl"),
            "--output",
            str(output / "ranker.json"),
            "--epochs",
            str(config["epochs"]),
            "--seed",
            str(config["seed"]),
        ]
    if stage == "score-calibration":
        return [
            *CLI,
            "score-ranker-calibration",
            str(output / "calibration.jsonl"),
            "--ranker-profile",
            str(output / "ranker.json"),
            "--output",
            str(output / "calibration-scores.jsonl"),
        ]
    if stage == "calibrate":
        name = read_json(output / "ranker.json")["profile"]["name"]
        return [
            *CLI,
            "calibrate-ranker",
            str(output / "calibration-scores.jsonl"),
            "--source-ranker",
            name,
            "--output",
            str(output / "calibration.json"),
        ]
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        stage,
        "--output-dir",
        str(output),
    ]


def execute(args: argparse.Namespace) -> int:
    output = ensure_safe_output_dir(args.output_dir)
    files = sorted(glob.glob(args.candidates))
    if not args.allow_raw_export:
        raise PermissionError("explicit local-research authorization required")
    if not files or sum(Path(p).stat().st_size for p in files) > min(
        args.max_storage_bytes, 64 * 1024 * 1024
    ):
        raise ValueError("missing candidates or input size budget exceeded")
    _read_candidate_files(files)  # Existing rights contract, before creating outputs.
    config = {
        "schema": "fixed-research-cycle-v1",
        "inputs": [{"path": str(Path(p).resolve()), "sha256": sha256(Path(p))} for p in files],
        "source": source_identity(),
        "environment": environment(),
        **{
            k: getattr(args, k)
            for k in (
                "ranker",
                "epochs",
                "seed",
                "bootstrap_iterations",
                "max_trials",
                "audio_seconds",
                "max_audio_seconds",
                "max_wall_seconds",
                "max_storage_bytes",
                "evaluation_role",
            )
        },
        "development_selection": "not-performed-fixed-before-test",
        "automatic_promotion": False,
    }
    if config["audio_seconds"] > config["max_audio_seconds"]:
        raise BudgetExceeded("audio budget exceeded")
    receipt = {
        "schema": "research-cycle-v1",
        "config_digest": digest(config),
        "completed": [],
        "artifacts": {},
        "stages": [],
        "attempts": [],
        "trials_started": 0,
        "spent_seconds": 0.0,
        "status": "running",
        "promotion": "not-evaluated",
    }
    if args.resume:
        existing = read_json(output / "config.json")
        if existing != config:
            raise ValueError("resume input, configuration, code or environment mismatch")
        receipt = read_json(output / "cycle.json")
        if receipt["config_digest"] != digest(config):
            raise ValueError("resume receipt configuration mismatch")
        if receipt["completed"] != list(STAGES[: len(receipt["completed"])]):
            raise ValueError("invalid completed stage prefix")
        for name, expected in receipt["artifacts"].items():
            if name not in {n for s in receipt["completed"] for n in OUTPUTS[s]}:
                raise ValueError("unexpected receipt artifact")
            if sha256(output / name) != expected:
                raise ValueError("resume artifact digest mismatch: " + name)
        expected_names = {n for s in receipt["completed"] for n in OUTPUTS[s]}
        if set(receipt["artifacts"]) != expected_names:
            raise ValueError("incomplete artifact receipt")
        if receipt["status"] == "running":
            # The OS writer lock has been acquired. Conservatively charge elapsed
            # time since the interrupted attempt, including any offline interval.
            active_started = receipt.get("active_started_at")
            if not isinstance(active_started, (int, float)) or active_started > time.time():
                raise ValueError("unfinalized run has no valid timing evidence")
            receipt["spent_seconds"] += time.time() - active_started
            if receipt["attempts"] and receipt["attempts"][-1]["status"] == "running":
                receipt["attempts"][-1]["status"] = "interrupted"
        if receipt["completed"] == list(STAGES) and receipt["status"] == "completed":
            enforce_budget(
                output,
                time.monotonic() + args.max_wall_seconds - receipt["spent_seconds"],
                args.max_storage_bytes,
            )
            return 0
    else:
        if output.exists():
            raise FileExistsError("run directory exists; use --resume for a verified prefix")
        output.mkdir(parents=True)
        write_json(output / "config.json", config)
    started = time.monotonic()
    deadline = started + args.max_wall_seconds - receipt["spent_seconds"]
    receipt["status"] = "running"
    receipt["active_started_at"] = time.time()
    write_json(output / "cycle.json", receipt)
    code = 1
    try:
        for stage in STAGES[len(receipt["completed"]) :]:
            enforce_budget(output, deadline, args.max_storage_bytes)
            if read_json(output / "config.json") != config:
                raise ValueError("on-disk configuration changed during cycle")
            if source_identity() != config["source"]:
                raise ValueError("source changed during cycle")
            for item in config["inputs"]:
                if sha256(Path(item["path"])) != item["sha256"]:
                    raise ValueError("candidate input changed during cycle")
            for name, expected in receipt["artifacts"].items():
                if sha256(output / name) != expected:
                    raise ValueError("completed artifact changed during cycle")
            if stage == "fit-train":
                if receipt["trials_started"] >= args.max_trials:
                    raise BudgetExceeded(
                        "trial budget exhausted; optimizer resumption is unavailable"
                    )
                receipt["trials_started"] += 1
                write_json(output / "cycle.json", receipt)
            attempt = {"stage": stage, "status": "running"}
            receipt["attempts"].append(attempt)
            write_json(output / "cycle.json", receipt)
            # Each attempt owns a separate log; interrupted/failed evidence stays intact.
            run_stage(
                f"{len(receipt['attempts']):03d}-{stage}",
                commands(stage, output, config),
                output,
                receipt,
                deadline,
                args.max_storage_bytes,
            )
            if read_json(output / "config.json") != config:
                raise ValueError("on-disk configuration changed during stage")
            for name in OUTPUTS[stage]:
                path = output / name
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError("stage missing required output: " + name)
                receipt["artifacts"][name] = sha256(path)
            attempt["status"] = "completed"
            receipt["completed"].append(stage)
            receipt["status"] = "running"
            write_json(output / "cycle.json", receipt)
        if source_identity() != config["source"]:
            raise ValueError("source changed during cycle")
        if read_json(output / "config.json") != config:
            raise ValueError("on-disk configuration changed before completion")
        for item in config["inputs"]:
            if sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError("candidate input changed before completion")
        enforce_budget(output, deadline, args.max_storage_bytes)
        receipt["status"] = "completed"
        code = 0
    except (KeyboardInterrupt, BudgetExceeded) as error:
        receipt.update(status="partial", reason=str(error) or "interrupted")
        code = 3
    except Exception as error:
        receipt.update(status="failed", reason=type(error).__name__ + ": " + str(error))
    finally:
        receipt["spent_seconds"] += time.monotonic() - started
        if receipt["attempts"] and receipt["attempts"][-1]["status"] == "running":
            receipt["attempts"][-1]["status"] = receipt["status"]
        receipt["local_artifacts"] = {
            str(p.relative_to(output)): sha256(p)
            for p in files_under(output)
            if p.name not in {"cycle.json", "receipt.json"}
        }
        write_json(output / "cycle.json", receipt)
    print(json.dumps({"status": receipt["status"], "completed": receipt["completed"]}))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", choices=STAGES)
    if "--worker" in (argv if argv is not None else sys.argv[1:]):
        args = parser.parse_args(argv)
        worker(args.worker, Path(args.output_dir))
        return 0
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--allow-raw-export", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ranker", choices=("pairwise", "listwise"), default="pairwise")
    parser.add_argument("--epochs", type=positive, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-iterations", type=positive, default=2000)
    parser.add_argument(
        "--evaluation-role",
        choices=("synthetic", "development", "regression-exposed"),
        required=True,
    )
    for name in ("max-trials", "max-audio-seconds", "max-wall-seconds", "max-storage-bytes"):
        parser.add_argument("--" + name, type=positive, required=True)
    parser.add_argument(
        "--audio-seconds",
        type=float,
        required=True,
        help="Measured duration bound by the generation receipt; no re-inference here",
    )
    args = parser.parse_args(argv)
    if not 100 <= args.bootstrap_iterations <= 10000 or args.epochs > 1000:
        parser.error("bootstrap iterations must be 100..10000; epochs must be 1..1000")
    if not math.isfinite(args.audio_seconds) or args.audio_seconds <= 0:
        parser.error("audio-seconds must be finite and positive")
    try:
        from semantic_asr.experiment_runner import _checkpoint_writer_lock

        output = ensure_safe_output_dir(args.output_dir)
        with _checkpoint_writer_lock(output.with_name(output.name + "-writer")):
            return execute(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
