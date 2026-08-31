from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import FasterWhisperAdapter, Qwen3ASRAdapter, Qwen3ForcedAlignerAdapter
from .cache import EvidenceCache
from .calibration import (
    CalibrationProfile,
    brier_score,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    risk_coverage_curve,
)
from .contracts import CandidateEvidence
from .fusion import evidence_summary, fuse_candidates
from .longform import SemanticASRTranscriber
from .outputs import write_outputs
from .planner import EvidenceBudget, plan_evidence
from .rights import RightsRegistry
from .semantic_lattice import build_semantic_lattice
from .teachers import OllamaRanker, OpenAICompatibleRanker


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
    normalized = {aliases.get(key, key): value for key, value in row.items()}
    return CandidateEvidence.from_dict(normalized)


def _load_candidates(path: str | Path) -> list[CandidateEvidence]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("candidate manifest must be an array or contain candidates")
    return [_candidate(row) for row in rows]


def _hotwords(args: argparse.Namespace) -> tuple[str, ...]:
    values: list[str] = []
    if args.hotwords:
        values.extend(
            item.strip() for item in args.hotwords.replace("、", ",").split(",") if item.strip()
        )
    if args.hotwords_file:
        text = Path(args.hotwords_file).read_text(encoding="utf-8")
        values.extend(
            item.strip()
            for line in text.splitlines()
            for item in line.replace("、", ",").split(",")
            if item.strip()
        )
    return tuple(dict.fromkeys(values))


def _write_or_print(payload: object, output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def command_demo(args: argparse.Namespace) -> int:
    candidates = [
        CandidateEvidence(
            "spoken",
            "昨日学校を行きました",
            acoustic=0.91,
            mora=0.89,
            lexical=0.38,
            preservation=0.96,
            teacher=0.08,
            source="whisper",
        ),
        CandidateEvidence(
            "clean",
            "昨日学校に行きました",
            acoustic=0.62,
            mora=0.58,
            lexical=0.95,
            preservation=0.42,
            teacher=0.92,
            source="whisper",
        ),
        CandidateEvidence(
            "wrong",
            "昨日会社に行きました",
            acoustic=0.31,
            mora=0.22,
            lexical=0.83,
            preservation=0.21,
            teacher=0.71,
            source="whisper",
        ),
    ]
    ranked = fuse_candidates(candidates)
    lattice = build_semantic_lattice(
        candidates,
        posterior=ranked[0].gate.posterior,
        pivot_candidate_id=ranked[0].candidate.candidate_id,
        segment_start_ms=0,
        segment_end_ms=4_000,
    )
    plan = plan_evidence(ranked, lattice)
    _write_or_print(
        {
            "selected": ranked[0].candidate.text,
            "decision": "provisional" if ranked[0].gate.abstain else "accepted",
            "summary": evidence_summary(ranked),
            "ranked": [asdict(item) for item in ranked],
            "lattice": asdict(lattice),
            "evidencePlan": asdict(plan),
        },
        args.output,
    )
    return 0


def command_fuse(args: argparse.Namespace) -> int:
    candidates = _load_candidates(args.input)
    ranked = fuse_candidates(candidates)
    lattice = build_semantic_lattice(
        candidates,
        posterior=ranked[0].gate.posterior,
        pivot_candidate_id=ranked[0].candidate.candidate_id,
        segment_start_ms=args.start_ms,
        segment_end_ms=args.end_ms,
    )
    plan = plan_evidence(
        ranked,
        lattice,
        budget=EvidenceBudget(
            total_cost_ms=args.budget_ms,
            max_actions=args.max_actions,
        ),
    )
    _write_or_print(
        {
            "selectedCandidateId": ranked[0].candidate.candidate_id,
            "observedTranscript": ranked[0].candidate.text,
            "summary": evidence_summary(ranked),
            "ranked": [asdict(item) for item in ranked],
            "lattice": asdict(lattice),
            "evidencePlan": asdict(plan),
        },
        args.output,
    )
    return 0


def command_calibrate(args: argparse.Namespace) -> int:
    probabilities: list[float] = []
    labels: list[int] = []
    for line_number, line in enumerate(
        Path(args.input).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            probability = float(row["confidence"])
            label = int(bool(row["correct"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid calibration row {line_number}") from exc
        if not 0 <= probability <= 1:
            raise ValueError(f"confidence outside [0, 1] on line {line_number}")
        probabilities.append(probability)
        labels.append(label)
    if len(probabilities) < 10:
        raise ValueError("at least ten held-out examples are required")
    temperature = fit_temperature(probabilities, labels)
    profile = CalibrationProfile(
        name=args.name,
        temperature=temperature,
        input_kind="probability",
    )
    calibrated = [float(profile.transform(value)) for value in probabilities]
    payload = {
        "schemaVersion": "1.0.0",
        "profile": {**asdict(profile), "digest": profile.digest},
        "sampleCount": len(labels),
        "before": {
            "ece": expected_calibration_error(probabilities, labels, bins=args.bins),
            "brier": brier_score(probabilities, labels),
            "nll": negative_log_likelihood(probabilities, labels),
        },
        "after": {
            "ece": expected_calibration_error(calibrated, labels, bins=args.bins),
            "brier": brier_score(calibrated, labels),
            "nll": negative_log_likelihood(calibrated, labels),
        },
        "riskCoverage": [
            {"coverage": coverage, "risk": risk}
            for coverage, risk in risk_coverage_curve(calibrated, labels)
        ],
    }
    _write_or_print(payload, args.output)
    return 0


def command_rights(args: argparse.Namespace) -> int:
    registry = RightsRegistry.load(args.registry)
    record = registry.require(args.asset, args.operation)
    _write_or_print({"status": "allowed", "asset": asdict(record)}, args.output)
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    source = Path(args.audio).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    base = FasterWhisperAdapter(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
    )
    second_ear = None
    if args.qwen_second_ear:
        second_ear = Qwen3ASRAdapter(
            model=args.qwen_model,
            dtype=args.qwen_dtype,
            device_map=args.qwen_device_map,
            max_inference_batch_size=1,
            return_timestamps=args.qwen_timestamps,
        )
    aligner = None
    if args.qwen_aligner:
        aligner = Qwen3ForcedAlignerAdapter(
            model=args.qwen_aligner_model,
            dtype=args.qwen_dtype,
            device_map=args.qwen_device_map,
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
                total_cost_ms=args.evidence_budget_ms,
                max_actions=args.max_evidence_actions,
            ),
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
            {item.strip() for item in args.formats.split(",") if item.strip()}
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
            {"status": "ok", "outputs": outputs, "diagnostics": result.diagnostics},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-asr",
        description="Evidence-preserving, semantics-aware Japanese ASR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a deterministic fusion demo")
    demo.add_argument("--output")
    demo.set_defaults(func=command_demo)

    fuse = subparsers.add_parser("fuse", help="fuse an existing candidate manifest")
    fuse.add_argument("input")
    fuse.add_argument("--output")
    fuse.add_argument("--start-ms", type=int, default=0)
    fuse.add_argument("--end-ms", type=int, default=30_000)
    fuse.add_argument("--budget-ms", type=int, default=12_000)
    fuse.add_argument("--max-actions", type=int, default=8)
    fuse.set_defaults(func=command_fuse)

    calibrate = subparsers.add_parser(
        "calibrate", help="fit held-out scalar confidence calibration"
    )
    calibrate.add_argument("input")
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--name", default="observed-posterior")
    calibrate.add_argument("--bins", type=int, default=15)
    calibrate.set_defaults(func=command_calibrate)

    rights = subparsers.add_parser("rights", help="fail-closed rights check")
    rights.add_argument("registry")
    rights.add_argument("asset")
    rights.add_argument(
        "operation",
        choices=[
            "train",
            "derive_features",
            "redistribute_raw",
            "export_speaker_id",
        ],
    )
    rights.add_argument("--output")
    rights.set_defaults(func=command_rights)

    transcribe = subparsers.add_parser(
        "transcribe", help="complete long-form Japanese transcription"
    )
    transcribe.add_argument("audio")
    transcribe.add_argument("-o", "--output-dir", default="transcripts")
    transcribe.add_argument("--overwrite", action="store_true")
    transcribe.add_argument("--duration-ms", type=int)
    transcribe.add_argument("--language", default="ja")
    transcribe.add_argument("--model", default="large-v3-turbo")
    transcribe.add_argument("--device", default="auto")
    transcribe.add_argument("--compute-type", default="default")
    transcribe.add_argument("--window-ms", type=int, default=28_000)
    transcribe.add_argument("--overlap-ms", type=int, default=1_200)
    transcribe.add_argument("--initial-prompt")
    transcribe.add_argument("--hotwords")
    transcribe.add_argument("--hotwords-file")
    transcribe.add_argument("--context", default="")
    transcribe.add_argument("--cache", default=".semantic-asr/evidence-cache-v1.sqlite3")
    transcribe.add_argument("--no-cache", action="store_true")
    transcribe.add_argument("--evidence-budget-ms", type=int, default=12_000)
    transcribe.add_argument("--max-evidence-actions", type=int, default=8)
    transcribe.add_argument("--qwen-second-ear", action="store_true")
    transcribe.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-0.6B")
    transcribe.add_argument("--qwen-device-map", default="cuda:0")
    transcribe.add_argument("--qwen-dtype", default="float16")
    transcribe.add_argument("--qwen-timestamps", action="store_true")
    transcribe.add_argument("--qwen-aligner", action="store_true")
    transcribe.add_argument("--qwen-aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    transcribe.add_argument("--teacher-model")
    transcribe.add_argument("--teacher-protocol", choices=["ollama", "openai"], default="ollama")
    transcribe.add_argument("--teacher-endpoint")
    transcribe.add_argument(
        "--formats", default="all", help="json,observed,normalized,md,srt,vtt or all"
    )
    transcribe.set_defaults(func=command_transcribe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
