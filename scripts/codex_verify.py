"""Fixed, model-free verification entry point for Codex and CI.

This runs existing project commands; it is not an agent, research scheduler,
checkpoint codec, or model promotion service. Raw logs stay local to a fresh
external output directory. Only report.json is intended for CI publication.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("core", "installed", "cpu")
CPU_TESTS = (
    "tests/test_training_optional.py",
    "tests/test_acoustic_verifier_optional.py",
    "tests/test_mora_training_regressions.py",
    "tests/test_weight_pilot.py",
    "tests/test_training_supervision_contract.py",
)


@dataclass(frozen=True)
class Stage:
    name: str
    argv: tuple[str, ...]
    cwd: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(root: Path) -> dict:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], timeout=15)

    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(set(names) - {b""}):
        path = root / os.fsdecode(raw)
        if path.is_symlink():
            value = b"link\0" + os.fsencode(os.readlink(path))
        elif path.is_file():
            value = sha256_file(path).encode("ascii")
        elif not path.exists():
            value = b"missing"
        else:
            raise ValueError("unsupported source entry (for example an uninitialized submodule)")
        mode = str(path.lstat().st_mode).encode() if path.exists() or path.is_symlink() else b"0"
        digest.update(raw + b"\0" + mode + b"\0" + value + b"\0")
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "tree": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "effective_source_sha256": digest.hexdigest(),
        "index_sha256": hashlib.sha256(git("ls-files", "--stage", "-z")).hexdigest(),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def safe_output(root: Path, output: Path) -> Path:
    resolved = output.expanduser().resolve()
    if resolved == Path(resolved.anchor) or resolved.is_relative_to(root.resolve()):
        raise ValueError("output must be outside the checkout and not a filesystem root")
    if output.is_symlink() or resolved.exists():
        raise ValueError("output must be a new directory; existing evidence is never overwritten")
    return resolved


def commands(root: Path, output: Path, profile: str, python: str = sys.executable) -> list[Stage]:
    if profile not in PROFILES:
        raise ValueError("unknown verification profile")
    tests = CPU_TESTS if profile == "cpu" else ("tests",)
    stages = []
    if profile == "core":
        stages.append(Stage("base-environment", (python, "-c", (
            "import importlib.util; "
            "assert importlib.util.find_spec('torch') is None, "
            "'core evidence requires an environment without PyTorch'"
        )), root))
    elif profile == "cpu":
        stages.append(Stage("cpu-environment", (python, "-c", (
            "import torch, numpy, safetensors; "
            "assert torch.empty(0).device.type == 'cpu'; print(torch.__version__)"
        )), root))
    for name, args in (
        ("format", ("-m", "ruff", "format", "--check", "src", "tests", "scripts")),
        ("lint", ("-m", "ruff", "check", "src", "tests", "scripts")),
        ("compile", ("-m", "compileall", "-q", "src", "tests", "scripts")),
        ("tests", ("-m", "pytest", "-q", *tests, f"--junitxml={output / 'pytest.xml'}")),
        ("replay", ("scripts/replay_phonetic_decisions.py",)),
    ):
        stages.append(Stage(name, (python, *args), root))
    cli = (python, "-m", "semantic_asr")
    for name, args in (
        ("demo", ("demo", "--output", str(output / "demo.json"))),
        ("smoke", ("research-smoke", "--output", str(output / "research-smoke.json"))),
        ("help", ("--help",)),
        ("cascade-help", ("cascade", "--help")),
        ("ranker-help", ("train-ranker", "--help")),
        ("transcribe-help", ("transcribe-v2", "--help")),
    ):
        stages.append(Stage(name, (*cli, *args), root))
    stages.append(Stage("wheel-build", (
        python, "-m", "build", "--wheel", "--outdir", str(output / "wheelhouse")
    ), root))
    return stages


def wheel_commands(output: Path, python: str = sys.executable) -> list[Stage]:
    wheels = sorted((output / "wheelhouse").glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
        raise ValueError("the current build must produce exactly one regular wheel")
    env = output / "wheel-venv"
    executable = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    probe = (
        "import pathlib,sys,semantic_asr as p; "
        "assert pathlib.Path(p.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)); "
        "[getattr(p,n) for n in getattr(p,'__all__',())]; "
        "assert not {'torch','transformers','faster_whisper'} & set(sys.modules); "
        "print('outside-checkout exports OK')"
    )
    return [
        Stage("wheel-venv", (python, "-m", "venv", str(env)), output),
        Stage("wheel-install", (
            str(executable), "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])
        ), output),
        Stage("wheel-import", (str(executable), "-I", "-c", probe), output),
        Stage("wheel-demo", (
            str(executable), "-I", "-m", "semantic_asr", "demo",
            "--output", str(output / "wheel-demo.json")
        ), output),
        Stage("wheel-smoke", (
            str(executable), "-I", "-m", "semantic_asr", "research-smoke",
            "--output", str(output / "wheel-smoke.json")
        ), output),
    ]


def terminate(process: subprocess.Popen) -> None:
    """Stop the stage process tree on timeout, including build/test descendants."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run_stage(stage: Stage, log: Path, timeout: float) -> tuple[str, int | None]:
    if timeout <= 0:
        return "timeout", None
    env = dict(os.environ)
    # Model downloads are forbidden during verification. Isolated wheel BUILD may
    # need package-index access; provisioning that access is an environment task.
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "PYTHONUTF8": "1", "PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    if stage.name.startswith("wheel-") and stage.name != "wheel-build":
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
    with log.open("xb") as stream:
        try:
            process = subprocess.Popen(
                stage.argv, cwd=stage.cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except OSError:
            return "failed-to-start", None
        try:
            code = process.wait(timeout=timeout)
            return ("passed" if code == 0 else "failed"), code
        except subprocess.TimeoutExpired:
            terminate(process)
            return "timeout", None
        except KeyboardInterrupt:
            terminate(process)
            return "interrupted", None


def test_counts(path: Path) -> dict | None:
    if not path.is_file():
        return None
    cases = list(ET.parse(path).getroot().iter("testcase"))
    skipped = [case.find("skipped") for case in cases if case.find("skipped") is not None]
    return {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": len(skipped),
        "xfail": sum("xfail" in item.get("type", "") for item in skipped),
    }


# Prevent pytest collecting this utility when imported into test modules.
test_counts.__test__ = False


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify(root: Path, output: Path, profile: str, total: int, per_stage: int) -> int:
    if total <= 0 or per_stage <= 0:
        raise ValueError("timeouts must be positive")
    output = safe_output(root, output)
    before = source_identity(root)
    stages = commands(root, output, profile)
    output.mkdir(parents=True, exist_ok=False)
    packages = sorted({
        (dist.metadata.get("Name", "unknown"), dist.version)
        for dist in importlib.metadata.distributions()
    })
    report = {
        "schema": "semantic-asr-codex-verification-v1", "profile": profile,
        "source_before": before, "python": platform.python_version(),
        "platform": {"os": platform.system(), "machine": platform.machine()},
        "packages": [{"name": name, "version": version} for name, version in packages],
        "limits": {"total_seconds": total, "stage_seconds": per_stage},
        "stages": [], "status": "running", "tests": None,
        "new_model_inference": False, "new_acoustic_or_llm_weights": False,
        "experiment_complete": False, "promotion_approved": False,
    }
    started = time.monotonic()
    code = 1
    try:
        write_report(output / "report.json", report)
        index = 0
        while index < len(stages):
            stage = stages[index]
            remaining = total - (time.monotonic() - started)
            log = output / f"{index + 1:02d}-{stage.name}.log"
            stamp = time.monotonic()
            print(f"[{index + 1}] {stage.name}", flush=True)
            status, returncode = run_stage(stage, log, min(per_stage, remaining))
            required = {
                "tests": "pytest.xml", "demo": "demo.json", "smoke": "research-smoke.json",
                "wheel-demo": "wheel-demo.json", "wheel-smoke": "wheel-smoke.json",
            }.get(stage.name)
            if status == "passed" and required:
                artifact = output / required
                if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size == 0:
                    status = "missing-artifact"
            if status == "passed" and stage.name == "tests":
                counts = test_counts(output / "pytest.xml")
                if not counts or not counts["tests"] or counts["failures"] or counts["errors"]:
                    status = "invalid-test-evidence"
            argv = [arg.replace(str(root), "$REPO").replace(str(output), "$OUT")
                    .replace(sys.executable, "$PYTHON") for arg in stage.argv]
            report["stages"].append({
                "name": stage.name, "argv": argv,
                "cwd": "$REPO" if stage.cwd == root else "$OUT",
                "status": status, "returncode": returncode,
                "seconds": round(time.monotonic() - stamp, 3),
                "log_sha256": sha256_file(log) if log.exists() else None,
            })
            if source_identity(root) != before:
                report["status"] = "source-changed"
                break
            if status != "passed":
                report["status"] = status
                code = 124 if status == "timeout" else 130 if status == "interrupted" else 1
                break
            if stage.name == "wheel-build":
                stages.extend(wheel_commands(output))
            write_report(output / "report.json", report)
            index += 1
        else:
            report["status"] = "passed"
            code = 0
    except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as exc:
        # Avoid publishing exception text containing private paths or input text.
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
    finally:
        try:
            report["source_after"] = source_identity(root)
            report["tests"] = test_counts(output / "pytest.xml")
            if report["source_after"] != before:
                report["status"] = "source-changed"
                code = 1
            counts = report["tests"]
            if code == 0 and counts and counts["skipped"]:
                report["status"] = "passed-with-skips"
        except (OSError, ValueError, subprocess.SubprocessError, ET.ParseError) as exc:
            report["status"] = "evidence-error"
            report["error_type"] = type(exc).__name__
            code = 1
        report["seconds"] = round(time.monotonic() - started, 3)
        # Logs, transcripts, environments and the full output directory are NEVER
        # uploaded by this runner. These digests identify local evidence only.
        report["artifacts"] = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.glob("*"))
            if path.is_file() and not path.is_symlink() and path.name != "report.json"
        }
        for path in sorted((output / "wheelhouse").glob("*.whl")):
            if path.is_file() and not path.is_symlink():
                report["artifacts"][str(path.relative_to(output))] = sha256_file(path)
        write_report(output / "report.json", report)
    print(f"{report['status']}: {output / 'report.json'}", flush=True)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="installed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--total-seconds", type=int, default=1200)
    parser.add_argument("--stage-seconds", type=int, default=600)
    parser.add_argument("--plan", action="store_true", help="print fixed stages without running them")
    args = parser.parse_args(argv)
    output = args.output_dir or Path(tempfile.gettempdir()) / f"semantic-asr-{uuid.uuid4().hex}"
    if args.plan:
        print(json.dumps({
            "profile": args.profile,
            "stages": [s.name for s in commands(ROOT, output, args.profile)]
            + ["wheel-venv", "wheel-install", "wheel-import", "wheel-demo", "wheel-smoke"],
            "new_model_inference": False,
        }, indent=2))
        return 0
    try:
        return verify(ROOT, output, args.profile, args.total_seconds, args.stage_seconds)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"verification preflight failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
