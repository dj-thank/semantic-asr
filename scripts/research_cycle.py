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
import os
import stat
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from codex_pipeline import (
    CLI,
    BudgetExceeded,
    enforce_budget,
    environment,
    positive,
    run_stage,
    sha256,
    source_identity,
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


def strict_json(text: str) -> Any:
    """Reject ambiguous objects and non-finite numbers, including float overflow."""

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = item
        return value

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    def reject_constant(token):
        raise ValueError("nonstandard JSON constant: " + token)

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_float=finite_float,
        parse_constant=reject_constant,
    )


def read_json(path: Path) -> Any:
    return strict_json(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        strict_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_text_atomic(path: Path, text: str) -> None:
    """Use an exclusive temporary file; never follow a predictable .tmp alias."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix="." + path.name + "-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Any) -> None:
    write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    write_text_atomic(
        path, "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values)
    )


def output_files(output: Path) -> list[Path]:
    """Inspect without following aliases or opening special files.

    This guards a private run directory, not an OS sandbox against concurrent
    hostile mutation. Stage workers must still be trusted and bounded.
    """

    def onerror(error):
        raise error

    result = []
    for root, directories, filenames in os.walk(output, followlinks=False, onerror=onerror):
        for name in sorted(directories + filenames):
            path = Path(root) / name
            info = path.lstat()
            reparse = getattr(info, "st_file_attributes", 0) & getattr(
                stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
            )
            if stat.S_ISLNK(info.st_mode) or reparse:
                raise ValueError("output alias is forbidden: " + str(path.relative_to(output)))
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(
                    "output must be a regular file with a single link: "
                    + str(path.relative_to(output))
                )
            result.append(path)
    return sorted(result)


def preserve_abandoned_outputs(output: Path, stage: str, attempt: int) -> None:
    """Quarantine previous partial outputs; a successful retry must create its own."""
    names = [name for name in OUTPUTS[stage] if (output / name).exists()]
    if not names:
        return
    archive = output / "attempt-artifacts" / f"before-{attempt:03d}-{stage}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in names:
        (output / name).rename(archive / name)


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
    if not isinstance(frozen, dict) or frozen.get("schema") != "research-freeze-v1":
        raise ValueError("unsupported freeze receipt schema")
    if frozen.get("config_digest") != digest(config):
        raise ValueError("freeze configuration mismatch")
    required = ("ranker.json", "calibration.json", *OUTPUTS["provision"])
    if not isinstance(frozen.get("artifacts"), dict) or set(frozen["artifacts"]) != set(required):
        raise ValueError("freeze artifact set mismatch")
    for name in required:
        if sha256(output / name) != frozen["artifacts"][name]:
            raise ValueError("frozen artifact changed: " + name)
    inference = rows(output / "test-inference.jsonl")
    if not inference or any(
        not isinstance(row, dict)
        or set(row) != {"sampleId", "candidates"}
        or not isinstance(row["sampleId"], str)
        or not row["sampleId"].strip()
        for row in inference
    ):
        raise ValueError("invalid frozen inference cohort")
    expected = [row["sampleId"] for row in inference]
    if len(set(expected)) != len(expected) or frozen.get("expected_sample_ids") != expected:
        raise ValueError("freeze cohort or order mismatch")
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


def selection_rows(output: Path) -> list[dict[str, Any]]:
    """Replay the frozen candidate-only policy without parsing evaluation text."""
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
    return decisions


def select(output: Path) -> None:
    write_rows(output / "decisions.jsonl", selection_rows(output))


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
    if frozen["config_digest"] != digest(config):
        raise ValueError("evaluation configuration differs from frozen policy")
    decisions = rows(output / "decisions.jsonl")
    expected = frozen["expected_sample_ids"]
    if any(not isinstance(row, dict) for row in decisions) or [
        row.get("sampleId") for row in decisions
    ] != expected:
        raise ValueError("evaluation cohort or order changed")
    # Membership alone is insufficient: a different valid candidate could replace
    # the policy's selection. Replay the existing selector before parsing
    # evaluation references.
    replayed = selection_rows(output)
    if any(
        not isinstance(row, dict)
        or type(row.get("requires_additional_evidence")) is not bool
        for row in decisions
    ) or decisions != replayed:
        raise ValueError("saved decisions do not match the frozen candidate-only policy")
    references = rows(output / "test-reference.jsonl")
    if [r["sampleId"] for r in references] != expected:
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


def validate_resume_receipt(receipt: Any, config: dict[str, Any]) -> None:
    """Reject malformed accounting before any resume arithmetic or receipt write.

    This validates the existing local journal, not an authenticated external ledger.
    Trial charges may exceed recorded fit attempts after a crash between those writes;
    accepting that conservative overcount must never refund an already charged trial.
    """
    if not isinstance(receipt, dict) or receipt.get("schema") != "research-cycle-v1":
        raise ValueError("unsupported resume receipt schema")
    if receipt.get("status") not in ("running", "completed", "failed", "partial"):
        raise ValueError("invalid resume receipt status")
    if receipt.get("config_digest") != digest(config):
        raise ValueError("resume receipt configuration mismatch")
    spent = receipt.get("spent_seconds")
    if type(spent) not in (int, float) or not 0 <= spent <= sys.float_info.max:
        raise ValueError("resume spent_seconds must be finite and nonnegative")
    trials = receipt.get("trials_started")
    if type(trials) is not int or not 0 <= trials <= config["max_trials"]:
        raise ValueError("resume trials_started must be an integer within the trial budget")
    completed = receipt.get("completed")
    if not isinstance(completed, list) or completed != list(STAGES[: len(completed)]):
        raise ValueError("invalid completed stage prefix")
    if receipt["status"] == "completed" and completed != list(STAGES):
        raise ValueError("completed receipt has an unfinished stage prefix")
    if not isinstance(receipt.get("artifacts"), dict) or not isinstance(
        receipt.get("stages"), list
    ):
        raise ValueError("invalid resume artifact or stage ledger")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("invalid resume attempt ledger")
    finished = []
    fit_attempts = 0
    for index, attempt in enumerate(attempts):
        if (
            not isinstance(attempt, dict)
            or len(finished) == len(STAGES)
            or attempt.get("stage") != STAGES[len(finished)]
            or attempt.get("status")
            not in ("running", "completed", "failed", "partial", "interrupted")
        ):
            raise ValueError("invalid resume attempt sequence")
        if attempt["status"] == "running" and (
            index != len(attempts) - 1 or receipt["status"] != "running"
        ):
            raise ValueError("running attempt is not the active terminal attempt")
        fit_attempts += attempt["stage"] == "fit-train"
        if attempt["status"] == "completed":
            finished.append(attempt["stage"])
    if finished != completed or fit_attempts > trials:
        raise ValueError("resume attempts disagree with completed stages or charged trials")
    steps = {}
    previous_index = -1
    names = {f"{i + 1:03d}-{a['stage']}": i for i, a in enumerate(attempts)}
    for step in receipt["stages"]:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str):
            raise ValueError("invalid execution stage record")
        index = names.get(step["name"], -1)
        if index <= previous_index:
            raise ValueError("execution stages are duplicate, unknown or out of order")
        previous_index = index
        status = step.get("status")
        if status not in ("running", "passed", "not-completed"):
            raise ValueError("invalid execution stage status")
        if status == "running" and (
            index != len(attempts) - 1 or attempts[index]["status"] != "running"
        ):
            raise ValueError("execution stage is not the active attempt")
        seconds = step.get("seconds")
        if (status != "running" or seconds is not None) and (
            type(seconds) not in (int, float) or not 0 <= seconds <= sys.float_info.max
        ):
            raise ValueError("invalid execution stage timing")
        code = step.get("returncode")
        if (code is not None and type(code) is not int) or (status == "passed" and code != 0):
            raise ValueError("invalid execution return code")
        steps[index] = step
    for index, attempt in enumerate(attempts):
        if attempt["status"] == "completed" and steps.get(index, {}).get("status") != "passed":
            raise ValueError("completed attempt has no successful execution record")
    if receipt["status"] == "running":
        active_started = receipt.get("active_started_at")
        if (
            type(active_started) not in (int, float)
            or not 0 < active_started < math.inf
            or active_started > time.time()
        ):
            raise ValueError("unfinalized run has no valid timing evidence")


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
        output_files(output)  # Reject aliases before reading or writing resume evidence.
        existing = read_json(output / "config.json")
        if existing != config:
            raise ValueError("resume input, configuration, code or environment mismatch")
        receipt = read_json(output / "cycle.json")
        validate_resume_receipt(receipt, config)
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
            active_started = receipt["active_started_at"]
            recovered_seconds = time.time() - active_started
            receipt["spent_seconds"] += recovered_seconds
            if receipt["attempts"] and receipt["attempts"][-1]["status"] == "running":
                receipt["attempts"][-1]["status"] = "interrupted"
                if receipt["stages"] and receipt["stages"][-1]["status"] == "running":
                    # Keep the interrupted record, not a permanently active step.
                    # This bound includes the offline interval, not measured CPU time.
                    receipt["stages"][-1].update(
                        status="not-completed",
                        seconds=recovered_seconds,
                        timing_basis="conservative-recovery-bound",
                    )
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
    previously_spent = receipt["spent_seconds"]
    deadline = started + args.max_wall_seconds - previously_spent
    receipt["status"] = "running"
    receipt["active_started_at"] = time.time()
    write_json(output / "cycle.json", receipt)
    code = 1
    try:
        for stage in STAGES[len(receipt["completed"]) :]:
            output_files(output)
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
            preserve_abandoned_outputs(output, stage, len(receipt["attempts"]))
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
            output_files(output)
            # A worker must not mutate evidence committed by an earlier stage.
            # Check before accepting this stage, including the final report worker.
            for name, expected in receipt["artifacts"].items():
                if sha256(output / name) != expected:
                    raise ValueError("completed artifact changed during stage: " + name)
            pending_artifacts = {}
            for name in OUTPUTS[stage]:
                path = output / name
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError("stage missing required output: " + name)
                values = rows(path) if path.suffix == ".jsonl" else [read_json(path)]
                if not values or not all(isinstance(value, dict) for value in values):
                    raise ValueError("stage output must contain JSON objects: " + name)
                pending_artifacts[name] = sha256(path)
            # Commit the entire output set only after every output has passed.
            receipt["artifacts"].update(pending_artifacts)
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
        if receipt["attempts"] and receipt["attempts"][-1]["status"] == "running":
            receipt["attempts"][-1]["status"] = receipt["status"]
        try:
            receipt["local_artifacts"] = {
                str(p.relative_to(output)): sha256(p)
                for p in output_files(output)
                if p.name not in {"cycle.json", "receipt.json"}
            }
        except (OSError, ValueError) as error:
            # An inventory failure must not discard the original terminal failure.
            receipt["artifact_inventory_error"] = type(error).__name__ + ": " + str(error)
            receipt.setdefault("reason", "artifact inventory could not be verified")
            receipt["status"] = "failed"
            code = 1
        # Hashing and writing terminal evidence are work too. Never report a
        # completed cycle whose final inventory has exceeded the declared budget.
        if code == 0:
            try:
                enforce_budget(output, deadline, args.max_storage_bytes)
            except BudgetExceeded as error:
                receipt.update(status="partial", reason=str(error))
                code = 3
        receipt["spent_seconds"] = previously_spent + time.monotonic() - started
        write_json(output / "cycle.json", receipt)
        if code == 0:
            try:
                enforce_budget(output, deadline, args.max_storage_bytes)
            except BudgetExceeded as error:
                receipt.update(status="partial", reason=str(error))
                receipt["spent_seconds"] = previously_spent + time.monotonic() - started
                write_json(output / "cycle.json", receipt)
                code = 3
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
