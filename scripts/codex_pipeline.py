"""Bounded, repository-specific Codex validation and local research entry point.

This driver calls existing CLIs; it does not implement a second experiment,
checkpoint, score, rights or model-promotion contract. Research outputs stay local.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import sysconfig
import time
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "semantic-asr",
    "pytest",
    "ruff",
    "build",
    "setuptools",
    "wheel",
    "numpy",
    "torch",
    "safetensors",
    "soundfile",
    "faster-whisper",
    "ctranslate2",
)
CLI = [sys.executable, "-c", "from semantic_asr.cli_root import main; raise SystemExit(main())"]


class Blocked(RuntimeError):
    """A required resource or explicit authorization is absent."""


class BudgetExceeded(RuntimeError):
    """The bounded run stopped before completing all stages."""


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT).decode("utf-8").strip()


def source_identity() -> dict[str, Any]:
    """Bind committed identity AND staged/unstaged/untracked source, without staging it."""
    names = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode("utf-8")
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(set(names) - {""}):
        path = ROOT / name
        digest.update(name.encode("utf-8") + b"\0")
        if path.is_symlink():
            content = b"symlink\0" + os.readlink(path).encode("utf-8")
        elif path.is_file():
            content = b"file\0" + bytes.fromhex(sha256(path))
        else:
            content = b"missing\0"
        digest.update(content + b"\0")
    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "workspace_sha256": digest.hexdigest(),
    }


def environment() -> dict[str, Any]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
        "is_dependency_lock": False,
    }


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("budget must be a positive integer")
    return number


def output_path(value: str) -> Path:
    result = Path(value).expanduser().resolve()
    if result == Path(result.anchor) or result.is_relative_to(ROOT):
        raise ValueError("output must be outside the checkout and not a filesystem root")
    if result.exists():
        raise FileExistsError("use a new output directory; existing runs are never overwritten")
    return result


def files_under(directory: Path) -> list[Path]:
    result = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            if not path.resolve().is_relative_to(directory):
                raise ValueError("output symlinks may not escape the run directory")
            continue
        if path.is_file():
            result.append(path)
    return sorted(result)


def enforce_budget(output: Path, deadline: float, storage: int) -> None:
    if time.monotonic() >= deadline:
        raise BudgetExceeded("wall-clock budget exceeded")
    if sum(path.stat().st_size for path in files_under(output)) > storage:
        raise BudgetExceeded("output storage budget exceeded")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        # Also terminate descendants after their parent has exited.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    elif process.poll() is None:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.poll() is None:
            process.kill()
    process.wait()


def run_stage(
    name: str,
    command: list[str],
    output: Path,
    receipt: dict[str, Any],
    deadline: float,
    storage: int,
    *,
    cwd: Path = ROOT,
) -> None:
    enforce_budget(output, deadline, storage)
    step: dict[str, Any] = {"name": name, "command": command, "status": "running"}
    receipt["stages"].append(step)
    write_json(output / "receipt.json", receipt)
    started = time.monotonic()
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    env.pop("PYTHONPATH", None)
    process = None
    try:
        with (output / f"{name}.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            while process.poll() is None:
                enforce_budget(output, deadline, storage)
                time.sleep(0.1)
            step["returncode"] = process.returncode
            enforce_budget(output, deadline, storage)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        step["status"] = "passed"
    except BaseException:
        step["status"] = "not-completed"
        raise
    finally:
        if process is not None:
            stop_process(process)
        step["seconds"] = round(time.monotonic() - started, 3)
        write_json(output / "receipt.json", receipt)


def require_modules(names: tuple[str, ...]) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise Blocked("install the documented environment; missing: " + ", ".join(missing))


def junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("JUnit contains no executed test cases")
    return {
        "tests": len(cases),
        "failures": sum(x.find("failure") is not None for x in cases),
        "errors": sum(x.find("error") is not None for x in cases),
        "skipped": sum(x.find("skipped") is not None for x in cases),
    }


def check(args: argparse.Namespace, output: Path, receipt: dict[str, Any], stage: Any) -> None:
    require_modules(("pytest", "ruff", "build"))
    if args.lane == "core":
        if importlib.util.find_spec("torch") is not None:
            raise Blocked("core lane requires a separate environment without torch")
    else:
        require_modules(("torch", "numpy", "soundfile", "safetensors"))
    stage("format", [sys.executable, "-m", "ruff", "format", "--diff", "src", "tests", "scripts"])
    stage("lint", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"])
    stage("compile", [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"])
    try:
        stage(
            "pytest", [sys.executable, "-m", "pytest", "-q", f"--junitxml={output / 'tests.xml'}"]
        )
    finally:
        if (output / "tests.xml").exists():
            receipt["tests"] = junit_counts(output / "tests.xml")
    if args.lane == "training-cpu" and receipt["tests"]["skipped"]:
        raise ValueError(
            "training-cpu lane has skipped tests; inspect JUnit instead of claiming coverage"
        )
    stage("replay-fixtures", [sys.executable, "scripts/replay_phonetic_decisions.py"])
    receipt["replay_scope"] = "bundled regression fixtures, not all 96 decisions or new inference"
    stage("demo", [*CLI, "demo", "--output", str(output / "demo.json")])
    stage("research-smoke", [*CLI, "research-smoke", "--output", str(output / "smoke.json")])
    stage(
        "model-free-optimization",
        [
            sys.executable,
            "scripts/run_v02_model_free_validation.py",
            "--output",
            str(output / "optimization.json"),
        ],
    )
    for command in ("cascade", "generate-candidates", "transcribe-v2", "train-ranker"):
        stage(f"help-{command}", [*CLI, command, "--help"])
    build = [sys.executable, "-m", "build", "--wheel", "--outdir", str(output / "dist")]
    if not args.isolated_build:
        build.append("--no-isolation")
    receipt["isolated_build"] = bool(args.isolated_build)
    stage("build", build)
    wheels = list((output / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one freshly built wheel")
    venv = output / "wheel-venv"
    stage("wheel-venv", [sys.executable, "-m", "venv", "--copies", str(venv)])
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    stage(
        "wheel-install",
        [str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheels[0])],
        cwd=output,
    )
    code = (
        "import pathlib, semantic_asr; "
        "assert pathlib.Path(semantic_asr.__file__).resolve()"
        f".is_relative_to(pathlib.Path({str(venv)!r})); "
        "[getattr(semantic_asr, n) for n in semantic_asr.__all__]; "
        "from semantic_asr.cli_root import main; raise SystemExit(main())"
    )
    for command, extra in (
        ("demo", ["--output", str(output / "wheel-demo.json")]),
        ("research-smoke", ["--output", str(output / "wheel-smoke.json")]),
        ("transcribe-v2", ["--help"]),
    ):
        stage(f"wheel-{command}", [str(python), "-I", "-c", code, command, *extra], cwd=output)


def research_input(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_local_research:
        raise Blocked(
            "explicit --allow-local-research is required; it does not authorize publication"
        )
    from run_real_audio_pipeline import validate_candidate_rights

    from semantic_asr.experiment_runner import load_audio_manifest
    from semantic_asr.revisions import resolve_hugging_face_revision, verify_artifact_sha256

    if not 1 <= args.beam_size <= 50 or not 1 <= args.hypotheses <= 50:
        raise ValueError("beam/hypotheses must be between 1 and 50")
    if not 1 <= args.bootstrap_iterations <= 10000:
        raise ValueError("bootstrap iterations must be between 1 and 10000")
    manifest = Path(args.manifest).expanduser().resolve()
    if manifest.is_relative_to(ROOT) or manifest.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("use a bounded manifest outside the checkout")
    with manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index > args.max_records or len(line) > 1024 * 1024:
                raise ValueError("manifest exceeds record/line budget")
            row = json.loads(line)
            if not isinstance(row, dict) or "split" not in row:
                raise ValueError("every manifest row must declare its split")
            validate_candidate_rights(row, line_number=index)
    records = load_audio_manifest(manifest)
    if {row.split for row in records} != {"train", "calibration", "test"}:
        raise ValueError("nonempty train, calibration and test splits are required")
    seconds = 0.0
    identities = []
    audio_splits: dict[str, str] = {}
    for row in records:
        path = Path(row.audio_path)
        if not path.is_absolute() or path.resolve().is_relative_to(ROOT):
            raise ValueError("materialized WAV paths must be absolute and outside the checkout")
        with wave.open(str(path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
                raise ValueError("bounded research input requires mono 16 kHz PCM16 WAV")
            if audio.getnframes() <= 0:
                raise ValueError("empty WAV")
            seconds += audio.getnframes() / audio.getframerate()
            if seconds > args.max_audio_seconds:
                raise BudgetExceeded("input audio duration budget exceeded")
            size = 0
            while chunk := audio.readframes(32768):
                size += len(chunk)
            if size != audio.getnframes() * 2:
                raise ValueError("truncated PCM16 WAV")
        digest = sha256(path)
        if digest in audio_splits and audio_splits[digest] != row.split:
            raise ValueError("identical WAV bytes appear in different splits")
        audio_splits[digest] = row.split
        identities.append(digest)
    if Path(args.model).expanduser().is_dir():
        if args.model_revision or not args.model_artifact_sha256:
            raise ValueError("local models require artifact SHA-256, not a Hub revision")
        model = {"artifact_sha256": verify_artifact_sha256(args.model, args.model_artifact_sha256)}
    else:
        if args.model_artifact_sha256:
            raise ValueError("artifact SHA-256 is only for local model directories")
        model = {"revision": resolve_hugging_face_revision(args.model, args.model_revision, {})}
    return {
        "manifest_sha256": sha256(manifest),
        "audio_sha256": identities,
        "record_count": len(records),
        "audio_seconds": seconds,
        "model": model,
        "split_counts": {
            split: sum(row.split == split for row in records)
            for split in ("train", "calibration", "test")
        },
        "evaluation_role": args.evaluation_role,
        "speaker_disjointness_verified": False,
    }


def research(args: argparse.Namespace, output: Path, receipt: dict[str, Any], stage: Any) -> None:
    receipt["inputs"] = research_input(args)
    require_modules(("faster_whisper", "ctranslate2", "numpy"))
    receipt["runtime_network_policy"] = "offline model cache/local directory; provision separately"
    executable = Path(sysconfig.get_path("scripts")) / (
        "semantic-asr.exe" if os.name == "nt" else "semantic-asr"
    )
    if not executable.is_file():
        raise Blocked("install semantic-asr in the active Python environment")
    model_flags = (
        ["--model-artifact-sha256", receipt["inputs"]["model"]["artifact_sha256"]]
        if "artifact_sha256" in receipt["inputs"]["model"]
        else ["--model-revision", receipt["inputs"]["model"]["revision"]]
    )
    runtime = receipt["source"]["head"] + "+tree-" + receipt["source"]["tree"]
    for name in ("faster-whisper", "ctranslate2"):
        runtime += "+" + name + "-" + str(receipt["environment"]["packages"][name])
    candidates = output / "all-candidates.jsonl"
    stage(
        "generate",
        [
            *CLI,
            "generate-candidates",
            str(Path(args.manifest).expanduser().resolve()),
            "--output",
            str(candidates),
            "--allow-raw-export",
            "--model",
            args.model,
            *model_flags,
            "--runtime-revision",
            runtime,
            "--device",
            args.device,
            "--compute-type",
            args.compute_type,
            "--cpu-threads",
            "2",
            "--beam-size",
            str(args.beam_size),
            "--hypotheses",
            str(args.hypotheses),
        ],
    )
    stage(
        "post-candidate",
        [
            sys.executable,
            "scripts/run_real_audio_pipeline.py",
            "--candidates",
            glob.escape(str(candidates)),
            "--output-dir",
            str(output / "pipeline"),
            "--allow-raw-export",
            "--ranker",
            args.ranker,
            "--cli",
            str(executable),
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ],
    )
    for name in ("report.json", "report-raw.json"):
        report = json.loads((output / "pipeline" / name).read_text(encoding="utf-8"))
        if (
            type(report.get("sample_count")) is not int
            or report["sample_count"] != receipt["inputs"]["split_counts"]["test"]
        ):
            raise ValueError("benchmark did not account for the complete test cohort")
        for key in ("baseline_cer", "cascade_cer", "mbr_cer"):
            value = report.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("benchmark has a missing/non-finite error metric")
    for name in ("ranker.json", "calibration.json"):
        profile = json.loads((output / "pipeline" / name).read_text(encoding="utf-8"))
        if not isinstance(profile.get("profile"), dict) or not profile["profile"]:
            raise ValueError("missing trained/calibrated profile")
    if research_input(args) != receipt["inputs"]:
        raise ValueError("input/model bytes changed during research")
    receipt["new_acoustic_or_lora_weights"] = False
    receipt["research_scope"] = "one legacy v0.2 generation/ranker/calibration/evaluation cycle"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "plan", help="print implemented stages without running/installing/downloading"
    )
    validation = commands.add_parser(
        "check", help="validate code without repairing or publishing it"
    )
    validation.add_argument("--lane", choices=("core", "training-cpu"), default="core")
    validation.add_argument("--isolated-build", action="store_true")
    validation.add_argument("--require-clean", action="store_true")
    experiment = commands.add_parser("research", help="one explicitly authorized, local-only cycle")
    experiment.add_argument("--allow-local-research", action="store_true")
    experiment.add_argument("--manifest", required=True)
    experiment.add_argument("--model", required=True)
    experiment.add_argument("--model-revision")
    experiment.add_argument("--model-artifact-sha256")
    experiment.add_argument(
        "--evaluation-role", choices=("development", "regression-exposed"), required=True
    )
    experiment.add_argument("--max-audio-seconds", type=positive, required=True)
    experiment.add_argument("--max-records", type=positive, required=True)
    experiment.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    experiment.add_argument(
        "--compute-type", choices=("int8", "float16", "int8_float16", "float32"), default="int8"
    )
    experiment.add_argument("--beam-size", type=positive, default=12)
    experiment.add_argument("--hypotheses", type=positive, default=12)
    experiment.add_argument("--ranker", choices=("listwise", "pairwise"), default="listwise")
    experiment.add_argument("--bootstrap-iterations", type=positive, default=2000)
    for child in (validation, experiment):
        child.add_argument("--output-dir", required=True)
        child.add_argument("--max-wall-seconds", type=positive, required=True)
        child.add_argument("--max-storage-bytes", type=positive, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "check": [
                        "format",
                        "lint",
                        "compile",
                        "pytest",
                        "regression-fixtures",
                        "synthetic-optimization",
                        "CLI-discovery",
                        "outside-checkout-wheel",
                    ],
                    "research": [
                        "authorize-and-bound",
                        "generate",
                        "existing-post-candidate-driver",
                    ],
                    "trials": 1,
                    "automatic_promotion": False,
                    "handoff": "docs/development/CODEX_HANDOFF.md",
                },
                indent=2,
            )
        )
        return 0
    try:
        output = output_path(args.output_dir)
        output.mkdir(parents=True)
    except (ValueError, OSError) as error:
        print(type(error).__name__ + ": invalid or occupied output destination", file=sys.stderr)
        return 2
    started = time.monotonic()
    deadline = started + args.max_wall_seconds
    receipt: dict[str, Any] = {
        "schema": "codex-pipeline-receipt-v1",
        "command": args.command,
        "status": "running",
        "stages": [],
        "environment": environment(),
        "max_wall_seconds": args.max_wall_seconds,
        "max_storage_bytes": args.max_storage_bytes,
        "trials": 1,
        "promotion": "not-evaluated",
        "automatic_publish": False,
    }
    if args.command == "check":
        receipt["lane"] = args.lane
    else:
        receipt["max_audio_seconds"] = args.max_audio_seconds
        receipt["max_records"] = args.max_records
        receipt["paid_provisioning"] = False
    write_json(output / "receipt.json", receipt)
    code = 1
    try:
        receipt["source"] = source_identity()
        if receipt["source"]["dirty"] and (args.command == "research" or args.require_clean):
            raise Blocked("clean committed source required for this run")

        def stage(name: str, command: list[str], *, cwd: Path = ROOT) -> None:
            run_stage(name, command, output, receipt, deadline, args.max_storage_bytes, cwd=cwd)

        if args.command == "check":
            check(args, output, receipt, stage)
        else:
            research(args, output, receipt, stage)
        enforce_budget(output, deadline, args.max_storage_bytes)
        receipt["status"] = "completed"
        code = 0
    except Blocked as error:
        receipt.update(status="blocked", reason=str(error))
        code = 2
    except (BudgetExceeded, KeyboardInterrupt) as error:
        receipt.update(status="partial", reason=str(error) or "interrupted")
        code = 3
    except Exception as error:
        # Receipts may contain private paths. Never upload research receipts/logs automatically.
        receipt.update(status="failed", reason=type(error).__name__ + ": " + str(error))
    finally:
        try:
            receipt["source_after"] = source_identity()
            if receipt.get("source") != receipt["source_after"]:
                receipt.update(status="failed", reason="source changed during execution")
                code = 1
            receipt["artifacts"] = {
                str(path.relative_to(output)): {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files_under(output)
                if path.name != "receipt.json"
                and "wheel-venv" not in path.relative_to(output).parts
            }
        except Exception as error:
            receipt.update(status="failed", evidence_error=type(error).__name__)
            code = 1
        receipt["seconds"] = round(time.monotonic() - started, 3)
        write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "promotion": "not-evaluated"}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
