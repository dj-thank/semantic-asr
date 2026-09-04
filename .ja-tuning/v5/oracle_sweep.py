"""Fifth real-Japanese-audio iteration: candidate-space and consensus sweep.

The experiment is reference-aware only while aggregating evaluation metrics.  It never
persists raw audio, references, or hypotheses.  All public artifacts are pinned and all
samples used by rounds 1--4 are excluded before selection.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import tempfile
import unicodedata
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from semantic_asr.adapters import DecodeRequest, FasterWhisperAdapter

SCHEMA = "semantic-asr-ja-candidate-oracle-v5"
SEED = "semantic-asr-ja-candidate-oracle-v5-20260904"
PRIMARY_MODEL = "large-v3-turbo"
PRIMARY_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
KOTOBA_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
KOTOBA_REVISION = "f44edd35eaeb2274e85ac7b31fb2c6f59ff1c4bc"
DATASETS = (
    ("reazon", "japanese-asr/ja_asr.reazonspeech_test", "dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f"),
    ("jsut", "japanese-asr/ja_asr.jsut_basic5000", "278db379fc96167ff2293d7abf9ab86976afcd78"),
    ("common_voice", "japanese-asr/ja_asr.common_voice_8_0", "bf8819e8d9a5feb51b0c718686bd20ea67a3c729"),
)


@dataclass(frozen=True, slots=True)
class V4State:
    selection_seed: str
    pilot_reazon_keys: frozenset[str]
    prior_sample_hashes: frozenset[str]


@dataclass(frozen=True, slots=True)
class Sample:
    dataset: str
    dataset_id: str
    revision: str
    row_index: int
    sample_hash: str
    reference_hash: str
    reference: str
    waveform: np.ndarray
    duration: float
    original_rate: int


@dataclass(frozen=True, slots=True)
class Hypothesis:
    source: str
    source_rank: int
    text: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(ch for ch in value if unicodedata.category(ch)[0] not in {"P", "S", "Z", "C"})


def levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def sample_hash(dataset_id: str, revision: str, row_index: int, normalized_reference: str) -> str:
    return sha256_text(canonical_json({
        "dataset": dataset_id,
        "revision": revision,
        "row": row_index,
        "referenceSha256": sha256_text(normalized_reference),
    }))


def resample_16k(waveform: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16_000:
        return np.asarray(waveform, dtype=np.float32)
    divisor = gcd(rate, 16_000)
    output = resample_poly(np.asarray(waveform, dtype=np.float64), 16_000 // divisor, rate // divisor)
    if not np.isfinite(output).all():
        raise ValueError("non-finite resampling output")
    return np.asarray(output, dtype=np.float32)


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "frozenset"
            and len(value.args) == 1
        ):
            return frozenset(ast.literal_eval(value.args[0]))
        return ast.literal_eval(value)
    raise KeyError(name)


def parse_v4_state(path: Path) -> V4State:
    tree = ast.parse(path.read_text())
    round2 = _literal_assignment(tree, "ROUND2_SAMPLE_HASHES")
    round3 = _literal_assignment(tree, "ROUND3_SAMPLE_HASHES")
    return V4State(
        selection_seed=str(_literal_assignment(tree, "SELECTION_SEED")),
        pilot_reazon_keys=frozenset(_literal_assignment(tree, "PILOT_REAZON_SAMPLE_KEYS")),
        prior_sample_hashes=frozenset(round2 | round3),
    )


def v4_role(state: V4State, dataset: str, reference: str) -> str:
    fraction = int(
        sha256_text(
            ":".join((state.selection_seed, dataset, "role", sha256_text(normalize_text(reference))))
        )[:16],
        16,
    ) / float(0xFFFFFFFFFFFFFFFF)
    if fraction < 0.58:
        return "fit"
    if fraction < 0.80:
        return "calibration"
    return {"reazon": "reazon_test", "jsut": "jsut_test", "common_voice": "external_test"}[dataset]


def pilot_reazon_sample_key(row_index: int, reference: str) -> str:
    return sha256_text(f"{DATASETS[0][2]}:{row_index}:{normalize_text(reference)}")


def reconstruct_round4_reference_hashes(
    state: V4State, max_duration: float
) -> tuple[set[str], set[str]]:
    counts = {
        "reazon": (("fit", 28), ("calibration", 14), ("reazon_test", 18)),
        "jsut": (("fit", 10), ("calibration", 8), ("jsut_test", 10)),
        "common_voice": (("fit", 10), ("calibration", 8), ("external_test", 14)),
    }
    excluded_refs: set[str] = set()
    selected_hashes: set[str] = set()
    for dataset_name, dataset_id, revision in DATASETS:
        ds = load_dataset(dataset_id, revision=revision, split="test")
        ds = ds.cast_column("audio", Audio(decode=False))
        for role, count in counts[dataset_name]:
            candidates: list[tuple[str, int, str, str, str]] = []
            seen: set[str] = set()
            for row_index, row in enumerate(ds):
                reference = str(row.get("transcription") or "").strip()
                normalized = normalize_text(reference)
                if not 4 <= len(normalized) <= 120:
                    continue
                ref_hash = sha256_text(normalized)
                if ref_hash in excluded_refs or ref_hash in seen:
                    continue
                if v4_role(state, dataset_name, reference) != role:
                    continue
                key = sample_hash(dataset_id, revision, row_index, normalized)
                if key in state.prior_sample_hashes:
                    continue
                if dataset_name == "reazon" and pilot_reazon_sample_key(row_index, reference) in state.pilot_reazon_keys:
                    continue
                order = sha256_text(canonical_json({
                    "seed": state.selection_seed,
                    "dataset": dataset_id,
                    "revision": revision,
                    "role": role,
                    "row": row_index,
                    "referenceSha256": ref_hash,
                }))
                candidates.append((order, row_index, reference, ref_hash, key))
                seen.add(ref_hash)
            candidates.sort()
            selected = 0
            for _, row_index, _, ref_hash, key in candidates:
                waveform, rate = sf.read(io.BytesIO(ds[row_index]["audio"]["bytes"]), dtype="float32")
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
                if waveform.ndim != 1 or not np.isfinite(waveform).all():
                    continue
                duration = len(resample_16k(waveform, int(rate))) / 16_000
                if not 1.0 <= duration <= max_duration:
                    continue
                excluded_refs.add(ref_hash)
                selected_hashes.add(key)
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise RuntimeError(f"reconstructed {selected}/{count} for round4 {dataset_name}:{role}")
    if len(selected_hashes) != 120 or len(excluded_refs) != 120:
        raise AssertionError(
            f"round-4 reconstruction mismatch: samples={len(selected_hashes)} refs={len(excluded_refs)}"
        )
    return excluded_refs, selected_hashes


def select_fresh(
    *,
    state: V4State,
    count_per_dataset: int,
    max_duration: float,
    excluded_reference_hashes: set[str],
    round4_hashes: set[str],
) -> list[Sample]:
    output: list[Sample] = []
    for dataset_name, dataset_id, revision in DATASETS:
        ds = load_dataset(dataset_id, revision=revision, split="test")
        ds = ds.cast_column("audio", Audio(decode=False))
        candidates: list[tuple[str, int, str, str, str]] = []
        seen: set[str] = set()
        for row_index, row in enumerate(ds):
            reference = str(row.get("transcription") or "").strip()
            normalized = normalize_text(reference)
            if not 4 <= len(normalized) <= 120:
                continue
            ref_hash = sha256_text(normalized)
            if ref_hash in excluded_reference_hashes or ref_hash in seen:
                continue
            key = sample_hash(dataset_id, revision, row_index, normalized)
            if key in state.prior_sample_hashes or key in round4_hashes:
                continue
            if dataset_name == "reazon" and pilot_reazon_sample_key(row_index, reference) in state.pilot_reazon_keys:
                continue
            order = sha256_text(canonical_json({
                "seed": SEED,
                "dataset": dataset_id,
                "revision": revision,
                "row": row_index,
                "referenceSha256": ref_hash,
            }))
            candidates.append((order, row_index, reference, ref_hash, key))
            seen.add(ref_hash)
        candidates.sort()
        selected = 0
        for _, row_index, reference, ref_hash, key in candidates:
            raw = ds[row_index]["audio"]["bytes"]
            waveform, rate = sf.read(io.BytesIO(raw), dtype="float32")
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                continue
            waveform = resample_16k(waveform, int(rate))
            duration = len(waveform) / 16_000
            if not 1.0 <= duration <= max_duration:
                continue
            output.append(Sample(
                dataset=dataset_name,
                dataset_id=dataset_id,
                revision=revision,
                row_index=row_index,
                sample_hash=key,
                reference_hash=ref_hash,
                reference=reference,
                waveform=waveform,
                duration=duration,
                original_rate=int(rate),
            ))
            excluded_reference_hashes.add(ref_hash)
            selected += 1
            if selected == count_per_dataset:
                break
        if selected != count_per_dataset:
            raise RuntimeError(f"selected {selected}/{count_per_dataset} for {dataset_name}")
        print(canonical_json({"stage": "dataset", "dataset": dataset_name, "selected": selected}), flush=True)
    return output


def normalize_candidates(candidates: Sequence[Any], source: str) -> list[Hypothesis]:
    output: list[Hypothesis] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = normalize_text(candidate.text)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(Hypothesis(source=source, source_rank=int(candidate.rank), text=text))
    if not output:
        raise RuntimeError(f"{source} returned no usable hypothesis")
    return output


def union_candidates(*groups: Sequence[Hypothesis]) -> list[Hypothesis]:
    output: list[Hypothesis] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.text in seen:
                continue
            seen.add(item.text)
            output.append(item)
    return output


def candidate_metrics(reference: str, rows: Sequence[Hypothesis]) -> dict[str, Any]:
    errors = [levenshtein(reference, row.text) for row in rows]
    baseline = errors[0]
    oracle = min(errors)
    oracle_index = errors.index(oracle)
    return {
        "baselineEdits": baseline,
        "oracleEdits": oracle,
        "referenceCharacters": max(1, len(reference)),
        "candidateCount": len(rows),
        "oracleReachable": oracle < baseline,
        "exactReachable": oracle == 0,
        "oraclePosition": oracle_index + 1,
        "oracleSource": rows[oracle_index].source,
        "oracleSourceRank": rows[oracle_index].source_rank,
    }


def choose_consensus(primary: Sequence[Hypothesis], kotoba: Sequence[Hypothesis], policy: str) -> int:
    baseline = primary[0].text
    if policy == "kotoba_top_exact":
        target = kotoba[0].text
        if target == baseline:
            return 0
        for index, row in enumerate(primary[1:3], 1):
            if row.text == target:
                return index
        return 0
    if policy == "kotoba_any_exact":
        alternatives = {row.text for row in kotoba}
        if baseline in alternatives:
            return 0
        for index, row in enumerate(primary[1:4], 1):
            if row.text in alternatives:
                return index
        return 0
    raise ValueError(policy)


def aggregate(records: Sequence[dict[str, Any]], config: str) -> dict[str, Any]:
    rows = [row["configs"][config] for row in records]
    chars = sum(row["referenceCharacters"] for row in rows)
    baseline_edits = sum(row["baselineEdits"] for row in rows)
    oracle_edits = sum(row["oracleEdits"] for row in rows)
    return {
        "examples": len(rows),
        "characters": chars,
        "baselineCer": baseline_edits / chars,
        "oracleCer": oracle_edits / chars,
        "absoluteOracleCerReduction": (baseline_edits - oracle_edits) / chars,
        "relativeOracleCerReduction": (baseline_edits - oracle_edits) / max(1, baseline_edits),
        "oracleReachable": sum(bool(row["oracleReachable"]) for row in rows),
        "exactReachable": sum(bool(row["exactReachable"]) for row in rows),
        "meanCandidateCount": sum(row["candidateCount"] for row in rows) / len(rows),
        "meanOraclePosition": sum(row["oraclePosition"] for row in rows) / len(rows),
    }


def aggregate_policy(records: Sequence[dict[str, Any]], policy: str) -> dict[str, Any]:
    chars = sum(row["referenceCharacters"] for row in records)
    rows = [row["policies"][policy] for row in records]
    baseline = sum(row["baselineEdits"] for row in rows)
    selected = sum(row["selectedEdits"] for row in rows)
    return {
        "examples": len(records),
        "characters": chars,
        "baselineCer": baseline / chars,
        "cer": selected / chars,
        "absoluteCerReduction": (baseline - selected) / chars,
        "relativeCerReduction": (baseline - selected) / max(1, baseline),
        "changed": sum(row["selectedPrimaryIndex"] != 0 for row in rows),
        "helpfulFlips": sum(row["selectedEdits"] < row["baselineEdits"] for row in rows),
        "harmfulFlips": sum(row["selectedEdits"] > row["baselineEdits"] for row in rows),
        "neutralFlips": sum(row["selectedPrimaryIndex"] != 0 and row["selectedEdits"] == row["baselineEdits"] for row in rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-driver", type=Path, required=True)
    parser.add_argument("--count-per-dataset", type=int, default=15)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = __import__("time").monotonic()
    state = parse_v4_state(args.v4_driver)
    excluded_refs, round4_hashes = reconstruct_round4_reference_hashes(state, args.max_duration)
    samples = select_fresh(
        state=state,
        count_per_dataset=args.count_per_dataset,
        max_duration=args.max_duration,
        excluded_reference_hashes=excluded_refs,
        round4_hashes=round4_hashes,
    )
    if len({row.sample_hash for row in samples}) != len(samples):
        raise AssertionError("duplicate fresh sample")

    primary = FasterWhisperAdapter(
        model=PRIMARY_MODEL,
        device="cpu",
        compute_type="int8",
        model_revision=PRIMARY_REVISION,
        runtime_revision="real-ja-oracle-v5-primary",
        cpu_threads=args.cpu_threads,
    )
    kotoba = FasterWhisperAdapter(
        model=KOTOBA_MODEL,
        device="cpu",
        compute_type="int8",
        model_revision=KOTOBA_REVISION,
        runtime_revision="real-ja-oracle-v5-kotoba",
        cpu_threads=args.cpu_threads,
    )

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="semantic-asr-ja-oracle-v5-") as directory:
        work = Path(directory)
        for position, sample in enumerate(samples, 1):
            path = work / f"sample-{position:04d}.wav"
            sf.write(path, sample.waveform, 16_000, subtype="PCM_16")
            p6 = normalize_candidates(primary.decode(DecodeRequest(audio_path=str(path), language="ja", beam_size=6, hypotheses=6)), "primary6")
            p12 = normalize_candidates(primary.decode(DecodeRequest(audio_path=str(path), language="ja", beam_size=12, hypotheses=12)), "primary12")
            k4 = normalize_candidates(kotoba.decode(DecodeRequest(audio_path=str(path), language="ja", beam_size=4, hypotheses=4)), "kotoba4")
            reference = normalize_text(sample.reference)
            configs = {
                "primary6": candidate_metrics(reference, p6),
                "primary12": candidate_metrics(reference, p12),
                "primary6_plus_kotoba4": candidate_metrics(reference, union_candidates(p6, k4)),
                "primary12_plus_kotoba4": candidate_metrics(reference, union_candidates(p12, k4)),
                "kotoba4": candidate_metrics(reference, k4),
            }
            baseline_edits = configs["primary6"]["baselineEdits"]
            policy_baseline_edits = levenshtein(reference, p12[0].text)
            policies: dict[str, Any] = {}
            for policy in ("kotoba_top_exact", "kotoba_any_exact"):
                selected = choose_consensus(p12, k4, policy)
                selected_edits = levenshtein(reference, p12[selected].text)
                policies[policy] = {
                    "selectedPrimaryIndex": selected,
                    "selectedPrimaryRank": p12[selected].source_rank,
                    "baselineEdits": policy_baseline_edits,
                    "selectedEdits": selected_edits,
                }
            records.append({
                "sampleHash": sample.sample_hash,
                "referenceHash": sample.reference_hash,
                "dataset": sample.dataset,
                "durationMillis": round(sample.duration * 1000),
                "referenceCharacters": max(1, len(reference)),
                "baselineEdits": baseline_edits,
                "configs": configs,
                "policies": policies,
                "primaryTopStable": p6[0].text == p12[0].text,
            })
            print(canonical_json({
                "stage": "decode",
                "progress": f"{position}/{len(samples)}",
                "dataset": sample.dataset,
                "sampleHash": sample.sample_hash,
                "p6": len(p6),
                "p12": len(p12),
                "k4": len(k4),
                "p6Oracle": configs["primary6"]["oracleEdits"],
                "p12Oracle": configs["primary12"]["oracleEdits"],
                "unionOracle": configs["primary12_plus_kotoba4"]["oracleEdits"],
            }), flush=True)

    configs = ("primary6", "primary12", "primary6_plus_kotoba4", "primary12_plus_kotoba4", "kotoba4")
    policies = ("kotoba_top_exact", "kotoba_any_exact")
    by_dataset = {}
    for dataset_name, _, _ in DATASETS:
        subset = [row for row in records if row["dataset"] == dataset_name]
        by_dataset[dataset_name] = {
            "configs": {name: aggregate(subset, name) for name in configs},
            "policies": {name: aggregate_policy(subset, name) for name in policies},
        }
    pooled = {
        "configs": {name: aggregate(records, name) for name in configs},
        "policies": {name: aggregate_policy(records, name) for name in policies},
    }
    report = {
        "schema": SCHEMA,
        "selectionSeed": SEED,
        "datasetFingerprint": sha256_text(canonical_json(sorted((row["sampleHash"], row["dataset"], row["durationMillis"]) for row in records))),
        "previousAudioExcluded": {
            "round2To3SampleHashes": len(state.prior_sample_hashes),
            "round1ReazonLegacyKeys": len(state.pilot_reazon_keys),
            "round4SamplesReconstructed": len(round4_hashes),
            "round4ReferenceHashesReconstructed": len(excluded_refs) - len(samples),
            "overlapDetected": False,
        },
        "datasets": [{"name": name, "id": did, "revision": rev, "samples": sum(row["dataset"] == name for row in records)} for name, did, rev in DATASETS],
        "models": {
            "primary": {"id": PRIMARY_MODEL, "revision": PRIMARY_REVISION, "decodeProfiles": [{"beam": 6, "hypotheses": 6}, {"beam": 12, "hypotheses": 12}]},
            "kotoba": {"id": KOTOBA_MODEL, "revision": KOTOBA_REVISION, "beam": 4, "hypotheses": 4},
        },
        "dataContract": {
            "rawAudioPersisted": False,
            "rawReferencePersisted": False,
            "rawHypothesisPersisted": False,
            "evaluationOnly": True,
            "notPromotionEvidence": True,
            "purpose": "measure candidate-space oracle headroom before another reranker fit",
        },
        "byDataset": by_dataset,
        "pooled": pooled,
        "diagnostics": {
            "primaryTopChangedWithBeam": sum(not row["primaryTopStable"] for row in records),
            "primary12StrictOracleWinsOverPrimary6": sum(row["configs"]["primary12"]["oracleEdits"] < row["configs"]["primary6"]["oracleEdits"] for row in records),
            "kotobaAddsOracleWinToPrimary6": sum(row["configs"]["primary6_plus_kotoba4"]["oracleEdits"] < row["configs"]["primary6"]["oracleEdits"] for row in records),
            "kotobaAddsOracleWinToPrimary12": sum(row["configs"]["primary12_plus_kotoba4"]["oracleEdits"] < row["configs"]["primary12"]["oracleEdits"] for row in records),
        },
        "hashedOutcomes": records,
        "runtimeSeconds": __import__("time").monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print("ORACLE_SWEEP_SUMMARY=" + canonical_json({"pooled": pooled, "byDataset": by_dataset, "diagnostics": report["diagnostics"], "datasetFingerprint": report["datasetFingerprint"]}), flush=True)


if __name__ == "__main__":
    main()
