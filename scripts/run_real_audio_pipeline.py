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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_FIELDS = ("reference", "annotatedReference", "referenceReading")
_RIGHTS_DECISION_FIELDS = ("rightsDecision", "rights_decision")
_LICENSE_FIELDS = ("license", "licenseId", "license_id", "licenseName", "license_name")


def ensure_safe_output_path(output: str | Path) -> Path:
    """Resolve a pipeline output and reject checkout/root destinations.

    ``Path.resolve`` is intentionally applied before the containment check.  This
    makes an existing symlink (including a symlink in an intermediate directory)
    subject to the same checkout guard as a normal path.
    """

    resolved = Path(output).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("pipeline output must not be a filesystem root")
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError(
        "pipeline output must be outside the repository checkout; "
        "use an external local-research directory"
    )


def ensure_safe_output_dir(output_dir: str | Path) -> Path:
    """Resolve and validate the pipeline output directory."""

    resolved = ensure_safe_output_path(output_dir)
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def _nested_values(row: Mapping[str, Any], key: str) -> list[Any]:
    """Read all copies of a rights field from a row and metadata containers."""

    values: list[Any] = []
    if key in row:
        values.append(row[key])
    for container_name in ("generation", "rights"):
        container = row.get(container_name)
        if isinstance(container, Mapping) and key in container:
            values.append(container[key])
    return values


def _non_empty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def row_is_reference_bearing(row: Mapping[str, Any]) -> bool:
    """Return whether a candidate row carries a non-empty reference surface."""

    return any(_non_empty_text(row.get(field)) is not None for field in _REFERENCE_FIELDS)


def row_exposes_rights_evidence(row: Mapping[str, Any]) -> bool:
    """Return whether a row includes any rights decision or licence field."""

    return any(_nested_values(row, field) for field in (*_RIGHTS_DECISION_FIELDS, *_LICENSE_FIELDS))


def validate_candidate_rights(row: Mapping[str, Any], *, line_number: int) -> tuple[str, str]:
    """Require an ``allow`` decision and a non-empty licence for one row.

    Generated candidate manifests put the licence under ``generation.licenseId``;
    local manifests commonly use top-level ``licenseId``.  Both forms are accepted,
    but a missing, ``review`` or ``deny`` value always stops the pipeline.
    """

    decision_values = [
        value for field in _RIGHTS_DECISION_FIELDS for value in _nested_values(row, field)
    ]
    decisions = [_non_empty_text(value) for value in decision_values]
    decision = next((value for value in decisions if value is not None), None)
    if decision is None:
        raise PermissionError(f"candidate row {line_number} is missing rightsDecision")
    if any(value is None or value.casefold() != "allow" for value in decisions):
        raise PermissionError(
            f"candidate row {line_number} rights decision is {decision!r}; expected 'allow'"
        )

    license_values = [value for field in _LICENSE_FIELDS for value in _nested_values(row, field)]
    licences = [_non_empty_text(value) for value in license_values]
    license_id = next((value for value in licences if value is not None), None)
    if license_id is None:
        raise PermissionError(f"candidate row {line_number} is missing license evidence")
    if any(value is None for value in licences):
        raise PermissionError(f"candidate row {line_number} is missing license evidence")
    return decision, license_id


def _pipeline_output_paths(output_dir: str | Path) -> dict[str, Path]:
    """Validate every path this runner passes to a child CLI before any writes."""

    out = ensure_safe_output_dir(output_dir)
    split_dir = ensure_safe_output_dir(out / "splits")
    paths = {
        "out": out,
        "merged": ensure_safe_output_path(out / "all-candidates.jsonl"),
        "splits": split_dir,
        "train_split": ensure_safe_output_path(split_dir / "train.jsonl"),
        "calibration_split": ensure_safe_output_path(split_dir / "calibration.jsonl"),
        "test_split": ensure_safe_output_path(split_dir / "test.jsonl"),
        "partition": ensure_safe_output_path(split_dir / "partition.json"),
        "ranker": ensure_safe_output_path(out / "ranker.json"),
        "scores": ensure_safe_output_path(out / "calibration-scores.jsonl"),
        "calibration": ensure_safe_output_path(out / "calibration.json"),
        "reranked": ensure_safe_output_path(out / "test-reranked.jsonl"),
        "report": ensure_safe_output_path(out / "report.json"),
        "report_raw": ensure_safe_output_path(out / "report-raw.json"),
    }
    return paths


def _read_candidate_files(files: list[str]) -> tuple[list[tuple[str, str]], bool, set[str]]:
    """Parse candidate JSONL before creating any output or derived artifact."""

    lines: list[tuple[str, str]] = []
    reference_bearing = False
    licenses: set[str] = set()
    for path in files:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"candidate row {path}:{line_number} must be a JSON object")
            bearing = row_is_reference_bearing(row)
            reference_bearing = reference_bearing or bearing
            # A row carrying rights evidence is checked even when it is a safe,
            # metadata-only row; this prevents review/deny values being ignored.
            if bearing or row_exposes_rights_evidence(row):
                _, license_id = validate_candidate_rights(row, line_number=line_number)
                licenses.add(license_id)
            lines.append((path, line))
    if not lines:
        raise ValueError("candidate files contain no JSON rows")
    return lines, reference_bearing, licenses


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="glob of candidate JSONL files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="external output directory; reference-bearing outputs never go in the checkout",
    )
    parser.add_argument(
        "--allow-raw-export",
        "--allow-raw",
        "--export-raw",
        "--local-research-output",
        dest="allow_raw_export",
        action="store_true",
        help=("explicitly authorize local-research processing of reference-bearing input/output"),
    )
    parser.add_argument("--ranker", choices=["listwise", "pairwise"], default="listwise")
    parser.add_argument("--ks", default="1,3,5,8,12,16,25,50")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--cli", default="semantic-asr")
    args = parser.parse_args()

    files = sorted(glob.glob(args.candidates))
    if not files:
        raise SystemExit(f"no candidate files match {args.candidates}")

    # Read and authorize the complete input before creating an output directory.
    # This prevents a failed rights check from leaving a misleading partial run.
    candidate_lines, reference_bearing, licenses = _read_candidate_files(files)
    if reference_bearing and not args.allow_raw_export:
        parser.error(
            "reference-bearing candidate input/output requires explicit local-research/raw-export "
            "authorization; pass --allow-raw-export and use an external output directory"
        )
    paths = _pipeline_output_paths(args.output_dir)
    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["splits"].mkdir(parents=True, exist_ok=True)

    with paths["merged"].open("w", encoding="utf-8", newline="\n") as handle:
        for _, line in candidate_lines:
            handle.write(line + "\n")
    print(
        json.dumps(
            {
                "mergedFiles": len(files),
                "merged": str(paths["merged"]),
                "referenceBearing": reference_bearing,
                "rawExportAuthorized": bool(args.allow_raw_export),
                "licenses": sorted(licenses),
            },
            ensure_ascii=False,
        )
    )

    cli = [args.cli]
    run([*cli, "partition-manifest", str(paths["merged"]), "--output-dir", str(paths["splits"])])
    train_command = "train-listwise-ranker" if args.ranker == "listwise" else "train-ranker"
    ranker = paths["ranker"]
    run(
        [
            *cli,
            train_command,
            str(paths["train_split"]),
            "--output",
            str(ranker),
        ]
    )
    scores = paths["scores"]
    run(
        [
            *cli,
            "score-ranker-calibration",
            str(paths["calibration_split"]),
            "--ranker-profile",
            str(ranker),
            "--output",
            str(scores),
        ]
    )
    profile_name = json.loads(ranker.read_text(encoding="utf-8"))["profile"]["name"]
    calibration = paths["calibration"]
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
    reranked = paths["reranked"]
    run(
        [
            *cli,
            "apply-ranker",
            str(paths["test_split"]),
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
            str(paths["report"]),
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
            str(paths["test_split"]),
            "--output",
            str(paths["report_raw"]),
            "--ks",
            args.ks,
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
