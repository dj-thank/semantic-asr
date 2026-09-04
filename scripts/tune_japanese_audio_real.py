"""Tune candidate reranking on pinned real Japanese broadcast audio.

The script is intentionally reference-blind while extracting features. References are used
only after every candidate has been generated and acoustically scored, for deterministic
calibration/test evaluation. Raw audio, references, and hypotheses remain in a temporary
working directory; the written report contains aggregate metrics and hashed sample IDs only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import os
import random
import statistics
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyopenjtalk
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCTC, AutoProcessor

from semantic_asr.adapters import DecodeRequest, FasterWhisperAdapter

DATASET_ID = "japanese-asr/ja_asr.reazonspeech_test"
DATASET_REVISION = "dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f"
WHISPER_MODEL = "large-v3-turbo"
WHISPER_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
PHONEME_MODEL = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
SPLIT_SEED = "semantic-asr-ja-real-v1"
REPORT_SCHEMA = "semantic-asr-ja-real-tuning-v1"
NO_GATE = 1.0e9


@dataclass
class CandidateRow:
    text: str
    asr_score: float
    ctc_nll_phone: float
    ctc_nll_frame: float
    phone_edit: float
    phone_length_ratio: float
    char_edits: int
    char_length: int


@dataclass
class ExampleRow:
    sample_hash: str
    split: str
    duration_seconds: float
    observed_phones: tuple[int, ...]
    surprisal_mean: float
    surprisal_std: float
    surprisal_spike_rate: float
    asr_margin: float
    candidates: list[CandidateRow]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    out: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if category.startswith(("P", "Z", "C")):
            continue
        out.append(character)
    return "".join(out)


def levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def split_for(reference: str) -> str:
    reference_digest = sha256_text(normalize_text(reference))
    bucket = int(sha256_text(f"{SPLIT_SEED}:{reference_digest}")[:8], 16) / 0xFFFFFFFF
    if bucket < 0.50:
        return "train"
    if bucket < 0.75:
        return "calibration"
    return "test"


def stable_sample_key(index: int, reference: str) -> str:
    return sha256_text(f"{DATASET_REVISION}:{index}:{normalize_text(reference)}")


def load_real_samples(per_split: dict[str, int], *, max_duration: float) -> list[dict[str, Any]]:
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    ordered: dict[str, list[tuple[str, int]]] = {name: [] for name in per_split}
    seen_reference_split: dict[str, str] = {}
    for index, row in enumerate(ds):
        reference = str(row["transcription"] or "").strip()
        normalized = normalize_text(reference)
        if not (4 <= len(normalized) <= 120):
            continue
        split = split_for(reference)
        reference_digest = sha256_text(normalized)
        prior = seen_reference_split.setdefault(reference_digest, split)
        if prior != split:
            raise AssertionError("normalized reference leakage across splits")
        ordered[split].append((stable_sample_key(index, reference), index))
    for values in ordered.values():
        values.sort()

    selected: list[dict[str, Any]] = []
    for split, target in per_split.items():
        for _, index in ordered[split]:
            row = ds[index]
            audio_bytes = row["audio"]["bytes"]
            waveform, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if rate != 16_000:
                raise ValueError(f"unexpected sample rate {rate}")
            duration = len(waveform) / rate
            if not (1.0 <= duration <= max_duration):
                continue
            reference = str(row["transcription"]).strip()
            selected.append(
                {
                    "index": index,
                    "split": split,
                    "reference": reference,
                    "waveform": np.asarray(waveform, dtype=np.float32),
                    "duration": duration,
                    "sample_hash": stable_sample_key(index, reference),
                }
            )
            if sum(item["split"] == split for item in selected) >= target:
                break
        actual = sum(item["split"] == split for item in selected)
        if actual != target:
            raise RuntimeError(f"could select only {actual}/{target} samples for {split}")
    selected.sort(key=lambda row: (row["split"], row["sample_hash"]))
    return selected


def g2p_ids(text: str, vocab: dict[str, int]) -> tuple[int, ...]:
    phones = pyopenjtalk.g2p(unicodedata.normalize("NFKC", text), kana=False).split()
    filtered = [phone for phone in phones if phone not in {"sil", "pau"}]
    missing = sorted({phone for phone in filtered if phone not in vocab})
    if missing:
        raise ValueError(f"Text2Phone produced out-of-vocabulary phones: {missing}")
    return tuple(vocab[phone] for phone in filtered)


def collapse_ctc(ids: Iterable[int], *, blank: int, ignored: set[int]) -> tuple[int, ...]:
    output: list[int] = []
    previous: int | None = None
    for value in ids:
        value = int(value)
        if value != previous and value != blank and value not in ignored:
            output.append(value)
        previous = value
    return tuple(output)


class TrigramPhoneLM:
    def __init__(self, sequences: Sequence[tuple[int, ...]], vocabulary_size: int, add_k: float = 0.1):
        self.vocabulary_size = vocabulary_size
        self.add_k = add_k
        self.ngrams: Counter[tuple[int, int, int]] = Counter()
        self.contexts: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            padded = (-2, -2, *sequence, -1)
            for index in range(2, len(padded)):
                context = (padded[index - 2], padded[index - 1])
                token = padded[index]
                self.contexts[context] += 1
                self.ngrams[(context[0], context[1], token)] += 1

    def surprisals(self, sequence: tuple[int, ...]) -> tuple[float, ...]:
        padded = (-2, -2, *sequence, -1)
        values: list[float] = []
        support = self.vocabulary_size + 1
        for index in range(2, len(padded)):
            context = (padded[index - 2], padded[index - 1])
            token = padded[index]
            numerator = self.ngrams[(context[0], context[1], token)] + self.add_k
            denominator = self.contexts[context] + self.add_k * support
            values.append(-math.log2(numerator / denominator))
        return tuple(values)


def robust_z(values: Sequence[float], *, higher_is_better: bool) -> list[float]:
    clean = [float(value) for value in values]
    if len(clean) == 1:
        return [0.0]
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean)
    if std <= 1e-9:
        return [0.0 for _ in clean]
    direction = 1.0 if higher_is_better else -1.0
    return [direction * (value - mean) / std for value in clean]


def score_candidates(example: ExampleRow, config: dict[str, float]) -> list[float]:
    candidates = example.candidates
    asr = robust_z([row.asr_score for row in candidates], higher_is_better=True)
    ctc_phone = robust_z([row.ctc_nll_phone for row in candidates], higher_is_better=False)
    ctc_frame = robust_z([row.ctc_nll_frame for row in candidates], higher_is_better=False)
    phone_edit = robust_z([row.phone_edit for row in candidates], higher_is_better=False)
    length = robust_z([row.phone_length_ratio for row in candidates], higher_is_better=False)
    return [
        asr[index]
        + config["w_ctc_phone"] * ctc_phone[index]
        + config["w_ctc_frame"] * ctc_frame[index]
        + config["w_phone_edit"] * phone_edit[index]
        + config["w_length"] * length[index]
        for index in range(len(candidates))
    ]


def choose_index(example: ExampleRow, config: dict[str, float]) -> int:
    rerank = (
        example.asr_margin <= config["margin_gate"]
        or example.surprisal_mean >= config["surprisal_gate"]
        or example.surprisal_spike_rate >= config["spike_gate"]
    )
    if not rerank:
        return 0
    scores = score_candidates(example, config)
    return min(range(len(scores)), key=lambda index: (-scores[index], index))


def aggregate_metrics(examples: Sequence[ExampleRow], config: dict[str, float] | None = None) -> dict[str, Any]:
    baseline_edits = 0
    selected_edits = 0
    oracle_edits = 0
    characters = 0
    exact_baseline = 0
    exact_selected = 0
    helpful = 0
    harmful = 0
    neutral = 0
    changed = 0
    oracle_reachable = 0
    for example in examples:
        baseline = example.candidates[0]
        selected_index = 0 if config is None else choose_index(example, config)
        selected = example.candidates[selected_index]
        oracle = min(example.candidates, key=lambda row: (row.char_edits, -row.asr_score))
        baseline_edits += baseline.char_edits
        selected_edits += selected.char_edits
        oracle_edits += oracle.char_edits
        characters += baseline.char_length
        exact_baseline += baseline.char_edits == 0
        exact_selected += selected.char_edits == 0
        oracle_reachable += oracle.char_edits < baseline.char_edits
        if selected_index != 0:
            changed += 1
            if selected.char_edits < baseline.char_edits:
                helpful += 1
            elif selected.char_edits > baseline.char_edits:
                harmful += 1
            else:
                neutral += 1
    count = len(examples)
    return {
        "examples": count,
        "characters": characters,
        "cer": selected_edits / max(1, characters),
        "baselineCer": baseline_edits / max(1, characters),
        "oracleCer": oracle_edits / max(1, characters),
        "exactRate": exact_selected / max(1, count),
        "baselineExactRate": exact_baseline / max(1, count),
        "changed": changed,
        "changeRate": changed / max(1, count),
        "helpfulFlips": helpful,
        "harmfulFlips": harmful,
        "neutralFlips": neutral,
        "oracleReachable": oracle_reachable,
        "relativeCerReduction": (
            (baseline_edits - selected_edits) / baseline_edits if baseline_edits else 0.0
        ),
    }


def config_key(config: dict[str, float]) -> tuple[Any, ...]:
    return (
        config["w_ctc_phone"],
        config["w_ctc_frame"],
        config["w_phone_edit"],
        config["w_length"],
        config["margin_gate"],
        config["surprisal_gate"],
        config["spike_gate"],
    )


def search_configs(
    calibration: Sequence[ExampleRow],
    *,
    surprisal_thresholds: Sequence[float],
    spike_thresholds: Sequence[float],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    iterations: list[dict[str, Any]] = []
    coarse: list[dict[str, float]] = []
    for w_ctc_phone, w_ctc_frame, w_phone_edit, w_length in itertools.product(
        (0.0, 0.2, 0.5, 1.0, 2.0),
        (0.0, 0.2, 0.5),
        (0.0, 0.2, 0.5, 1.0),
        (0.0, 0.2),
    ):
        if w_ctc_phone == w_ctc_frame == w_phone_edit == w_length == 0.0:
            continue
        coarse.append(
            {
                "w_ctc_phone": w_ctc_phone,
                "w_ctc_frame": w_ctc_frame,
                "w_phone_edit": w_phone_edit,
                "w_length": w_length,
                "margin_gate": NO_GATE,
                "surprisal_gate": NO_GATE,
                "spike_gate": NO_GATE,
            }
        )

    def rank_result(item: tuple[dict[str, float], dict[str, Any]]) -> tuple[Any, ...]:
        config, metrics = item
        return (
            metrics["cer"],
            metrics["harmfulFlips"],
            -metrics["helpfulFlips"],
            metrics["changeRate"],
            sum(abs(value) for value in config_key(config)[:4]),
            config_key(config),
        )

    coarse_scored = [(config, aggregate_metrics(calibration, config)) for config in coarse]
    coarse_scored.sort(key=rank_result)
    coarse_best = coarse_scored[0][0]
    iterations.append(
        {
            "iteration": "coarse-blend",
            "evaluated": len(coarse_scored),
            "bestConfig": coarse_best,
            "metrics": coarse_scored[0][1],
        }
    )

    refined_weights: list[dict[str, float]] = []
    for field in ("w_ctc_phone", "w_ctc_frame", "w_phone_edit", "w_length"):
        center = coarse_best[field]
        values = sorted({max(0.0, center + delta) for delta in (-0.3, -0.15, 0.0, 0.15, 0.3)})
        if field == "w_ctc_phone":
            ctc_phone_values = values
        elif field == "w_ctc_frame":
            ctc_frame_values = values
        elif field == "w_phone_edit":
            edit_values = values
        else:
            length_values = values
    for values in itertools.product(ctc_phone_values, ctc_frame_values, edit_values, length_values):
        refined_weights.append(
            {
                "w_ctc_phone": values[0],
                "w_ctc_frame": values[1],
                "w_phone_edit": values[2],
                "w_length": values[3],
                "margin_gate": NO_GATE,
                "surprisal_gate": NO_GATE,
                "spike_gate": NO_GATE,
            }
        )
    refined_scored = [(config, aggregate_metrics(calibration, config)) for config in refined_weights]
    refined_scored.sort(key=rank_result)
    refined_best = refined_scored[0][0]
    iterations.append(
        {
            "iteration": "refined-blend",
            "evaluated": len(refined_scored),
            "bestConfig": refined_best,
            "metrics": refined_scored[0][1],
        }
    )

    gated_configs: list[dict[str, float]] = []
    for margin_gate, surprisal_gate, spike_gate in itertools.product(
        (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, NO_GATE),
        tuple(surprisal_thresholds) + (NO_GATE,),
        tuple(spike_thresholds) + (NO_GATE,),
    ):
        config = dict(refined_best)
        config.update(
            margin_gate=margin_gate,
            surprisal_gate=surprisal_gate,
            spike_gate=spike_gate,
        )
        gated_configs.append(config)
    gated_scored = [(config, aggregate_metrics(calibration, config)) for config in gated_configs]
    gated_scored.sort(key=rank_result)
    best = gated_scored[0][0]
    iterations.append(
        {
            "iteration": "uncertainty-gating",
            "evaluated": len(gated_scored),
            "bestConfig": best,
            "metrics": gated_scored[0][1],
        }
    )
    leaderboard = [
        {"config": config, "metrics": metrics}
        for config, metrics in gated_scored[:10]
    ]
    return best, iterations, leaderboard


def bootstrap_delta(
    examples: Sequence[ExampleRow], config: dict[str, float], *, iterations: int, seed: int
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        sample = [examples[rng.randrange(len(examples))] for _ in range(len(examples))]
        metrics = aggregate_metrics(sample, config)
        deltas.append(metrics["baselineCer"] - metrics["cer"])
    deltas.sort()
    return {
        "meanAbsoluteCerReduction": statistics.fmean(deltas),
        "p2_5": deltas[int(0.025 * (len(deltas) - 1))],
        "p50": deltas[int(0.5 * (len(deltas) - 1))],
        "p97_5": deltas[int(0.975 * (len(deltas) - 1))],
        "probabilityPositive": sum(value > 0 for value in deltas) / len(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--calibration", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--hypotheses", type=int, default=6)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hypotheses > args.beam_size:
        parser.error("--hypotheses cannot exceed --beam-size")
    torch.set_num_threads(args.cpu_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))

    started = time.time()
    samples = load_real_samples(
        {"train": args.train, "calibration": args.calibration, "test": args.test},
        max_duration=args.max_duration,
    )
    ctc_revision = HfApi().model_info(PHONEME_MODEL).sha
    processor = AutoProcessor.from_pretrained(PHONEME_MODEL, revision=ctc_revision)
    phoneme_model = AutoModelForCTC.from_pretrained(PHONEME_MODEL, revision=ctc_revision)
    phoneme_model.eval()
    vocab = processor.tokenizer.get_vocab()
    blank = int(phoneme_model.config.pad_token_id)
    ignored = {vocab[name] for name in ("PAD", "UNK", "SOS", "EOS", "sil", "pau") if name in vocab}

    whisper = FasterWhisperAdapter(
        model=WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        model_revision=WHISPER_REVISION,
        runtime_revision="real-ja-tuning-v1",
        cpu_threads=args.cpu_threads,
    )

    staged: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="semantic-asr-ja-real-") as temporary:
        work = Path(temporary)
        for position, sample in enumerate(samples, 1):
            path = work / f"sample-{position:04d}.wav"
            sf.write(path, sample["waveform"], 16_000, subtype="PCM_16")
            request = DecodeRequest(
                audio_path=str(path),
                language="ja",
                beam_size=args.beam_size,
                hypotheses=args.hypotheses,
            )
            candidates = whisper.decode(request)
            if not candidates:
                raise RuntimeError("Whisper returned no candidate")

            inputs = processor(sample["waveform"], sampling_rate=16_000, return_tensors="pt")
            with torch.inference_mode():
                logits = phoneme_model(**inputs).logits[0]
            if not torch.isfinite(logits).all():
                raise RuntimeError("phoneme CTC returned non-finite logits")
            log_probs = logits.log_softmax(dim=-1)
            observed = collapse_ctc(logits.argmax(dim=-1).tolist(), blank=blank, ignored=ignored)
            staged.append(
                {
                    "sample": sample,
                    "candidates": candidates,
                    "log_probs": log_probs,
                    "observed": observed,
                }
            )
            print(
                canonical_json(
                    {
                        "progress": f"{position}/{len(samples)}",
                        "sampleHash": sample["sample_hash"],
                        "split": sample["split"],
                        "duration": round(sample["duration"], 3),
                        "candidateCount": len(candidates),
                        "observedPhones": len(observed),
                    }
                ),
                flush=True,
            )

    train_sequences = [row["observed"] for row in staged if row["sample"]["split"] == "train"]
    lm = TrigramPhoneLM(train_sequences, vocabulary_size=len(vocab))
    train_surprisals = [value for seq in train_sequences for value in lm.surprisals(seq)]
    spike_threshold = float(np.quantile(train_surprisals, 0.90)) if train_surprisals else math.inf

    examples: list[ExampleRow] = []
    for staged_row in staged:
        sample = staged_row["sample"]
        reference_normalized = normalize_text(sample["reference"])
        observed = staged_row["observed"]
        surprisals = lm.surprisals(observed)
        surprisal_mean = statistics.fmean(surprisals) if surprisals else 0.0
        surprisal_std = statistics.pstdev(surprisals) if len(surprisals) > 1 else 0.0
        surprisal_spike_rate = (
            sum(value > spike_threshold for value in surprisals) / len(surprisals)
            if surprisals
            else 0.0
        )
        raw_candidates = staged_row["candidates"]
        phone_targets = [g2p_ids(candidate.text, vocab) for candidate in raw_candidates]
        valid_indices = [index for index, target in enumerate(phone_targets) if target]
        if not valid_indices:
            raise RuntimeError("all N-best candidates produced empty phoneme targets")
        raw_candidates = [raw_candidates[index] for index in valid_indices]
        phone_targets = [phone_targets[index] for index in valid_indices]
        log_probs = staged_row["log_probs"]
        time_steps = int(log_probs.shape[0])
        repeated = log_probs.unsqueeze(1).expand(time_steps, len(phone_targets), -1)
        flat_targets = torch.tensor(
            [unit for target in phone_targets for unit in target], dtype=torch.long
        )
        losses = F.ctc_loss(
            repeated,
            flat_targets,
            torch.full((len(phone_targets),), time_steps, dtype=torch.long),
            torch.tensor([len(target) for target in phone_targets], dtype=torch.long),
            blank=blank,
            reduction="none",
            zero_infinity=False,
        )
        candidate_rows: list[CandidateRow] = []
        for candidate, target, loss in zip(raw_candidates, phone_targets, losses, strict=True):
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                loss_value = 1e6
            hypothesis = normalize_text(candidate.text)
            edits = levenshtein(reference_normalized, hypothesis)
            asr_score = candidate.avg_logprob
            if asr_score is None:
                asr_score = candidate.acoustic
            if asr_score is None or not math.isfinite(float(asr_score)):
                raise RuntimeError("candidate lacks a finite Whisper score")
            candidate_rows.append(
                CandidateRow(
                    text=candidate.text,
                    asr_score=float(asr_score),
                    ctc_nll_phone=loss_value / max(1, len(target)),
                    ctc_nll_frame=loss_value / max(1, time_steps),
                    phone_edit=levenshtein(observed, target) / max(1, len(observed), len(target)),
                    phone_length_ratio=abs(
                        math.log((len(target) + 1.0) / (len(observed) + 1.0))
                    ),
                    char_edits=edits,
                    char_length=max(1, len(reference_normalized)),
                )
            )
        candidate_rows.sort(key=lambda row: -row.asr_score)
        margin = (
            candidate_rows[0].asr_score - candidate_rows[1].asr_score
            if len(candidate_rows) > 1
            else math.inf
        )
        examples.append(
            ExampleRow(
                sample_hash=sample["sample_hash"],
                split=sample["split"],
                duration_seconds=sample["duration"],
                observed_phones=observed,
                surprisal_mean=surprisal_mean,
                surprisal_std=surprisal_std,
                surprisal_spike_rate=surprisal_spike_rate,
                asr_margin=margin,
                candidates=candidate_rows,
            )
        )

    grouped = {
        split: [row for row in examples if row.split == split]
        for split in ("train", "calibration", "test")
    }
    train_means = sorted(row.surprisal_mean for row in grouped["train"])
    train_spikes = sorted(row.surprisal_spike_rate for row in grouped["train"])
    surprisal_thresholds = [float(np.quantile(train_means, q)) for q in (0.50, 0.70, 0.85, 0.95)]
    spike_thresholds = [float(np.quantile(train_spikes, q)) for q in (0.50, 0.70, 0.85, 0.95)]
    best, tuning_iterations, leaderboard = search_configs(
        grouped["calibration"],
        surprisal_thresholds=surprisal_thresholds,
        spike_thresholds=spike_thresholds,
    )

    train_metrics = aggregate_metrics(grouped["train"], best)
    calibration_metrics = aggregate_metrics(grouped["calibration"], best)
    test_metrics = aggregate_metrics(grouped["test"], best)
    bootstrap = bootstrap_delta(
        grouped["test"], best, iterations=args.bootstrap, seed=20260904
    )
    dataset_fingerprint = sha256_text(
        canonical_json(
            [
                {
                    "sampleHash": row.sample_hash,
                    "split": row.split,
                    "durationMillis": round(row.duration_seconds * 1000),
                }
                for row in examples
            ]
        )
    )
    report = {
        "schema": REPORT_SCHEMA,
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "fingerprint": dataset_fingerprint,
            "sampleCount": len(examples),
            "splitCounts": {key: len(value) for key, value in grouped.items()},
            "totalDurationSeconds": sum(row.duration_seconds for row in examples),
            "speakerDisjoint": False,
            "rawDataPersisted": False,
        },
        "models": {
            "whisper": {
                "id": WHISPER_MODEL,
                "revision": WHISPER_REVISION,
                "beamSize": args.beam_size,
                "hypotheses": args.hypotheses,
                "computeType": "int8",
            },
            "phonemeCtc": {
                "id": PHONEME_MODEL,
                "revision": ctc_revision,
                "vocabularySize": len(vocab),
                "blankId": blank,
            },
        },
        "featureContract": {
            "referenceBlindExtraction": True,
            "candidateFeatures": [
                "whisper-average-log-probability",
                "phoneme-ctc-nll-per-phone",
                "phoneme-ctc-nll-per-frame",
                "greedy-phone-edit-distance",
                "phone-length-log-ratio",
            ],
            "utteranceRouting": [
                "whisper-top-two-margin",
                "native-trigram-phone-surprisal-mean",
                "native-trigram-phone-surprisal-spike-rate",
            ],
            "surprisalSpikeThresholdBits": spike_threshold,
        },
        "iterations": tuning_iterations,
        "selectedConfig": best,
        "metrics": {
            "train": train_metrics,
            "calibration": calibration_metrics,
            "test": test_metrics,
            "testBootstrapAbsoluteCerReduction": bootstrap,
        },
        "calibrationLeaderboard": leaderboard,
        "runtime": {
            "elapsedSeconds": time.time() - started,
            "cpuThreads": args.cpu_threads,
            "python": os.sys.version,
            "torch": torch.__version__,
        },
        "sampleHashes": {
            split: [row.sample_hash for row in grouped[split]] for split in grouped
        },
        "limitations": [
            "The public test set has no speaker IDs, so the fixed split is duplicate-reference-disjoint but not speaker-disjoint.",
            "The phoneme CTC model was trained on ReazonSpeech-domain material; this run is an in-domain pilot rather than an external-corpus generalization claim.",
            "References are used only for calibration and evaluation, never for candidate generation or acoustic feature extraction.",
            "No raw audio, transcript, or hypothesis text is written to the aggregate report.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TUNING_REPORT=" + canonical_json(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
