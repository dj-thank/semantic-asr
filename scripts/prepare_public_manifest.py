"""Build a rights-annotated audio manifest from a public Hugging Face ASR dataset.

Writes 16 kHz mono WAV files plus a JSONL manifest that ``semantic-asr generate-candidates``
accepts. Splits are assigned deterministically from the sample identifier. The public test sets
used here carry no speaker labels, so ``groupId`` falls back to the sample identifier and the
resulting split is *not* speaker-disjoint; record that limitation with any quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

DATASETS = {
    "reazonspeech-test": {
        "path": "japanese-asr/ja_asr.reazonspeech_test",
        "domain": "broadcast",
        "license": "reazonspeech-apache-2.0",
    },
    "jsut-basic5000": {
        "path": "japanese-asr/ja_asr.jsut_basic5000",
        "domain": "read",
        "license": "jsut-research-use",
    },
    "common-voice-8": {
        "path": "japanese-asr/ja_asr.common_voice_8_0",
        "domain": "read-crowd",
        "license": "cc0-1.0",
    },
}


def assign_split(sample_id: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    bucket = digest[0] / 255.0
    if bucket < 0.6:
        return "train"
    if bucket < 0.8:
        return "calibration"
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = whole test split")
    parser.add_argument("--seed", default="semantic-asr-public-v1")
    parser.add_argument("--rights-decision", default="allow", choices=["allow", "review", "deny"])
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    out = Path(args.output_dir)
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(spec["path"], split="test")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if args.limit:
        dataset = dataset.shuffle(seed=20260902).select(range(min(args.limit, len(dataset))))

    manifest_path = out / "manifest.jsonl"
    counts = {"train": 0, "calibration": 0, "test": 0}
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            audio = row["audio"]
            array, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            if rate != 16000:
                array = resample_poly(array, 16000, rate).astype(np.float32)
            sample_id = f"{args.dataset}-{index:06d}"
            wav_path = wav_dir / f"{sample_id}.wav"
            sf.write(wav_path, array, 16000, subtype="PCM_16")
            split = assign_split(sample_id, args.seed)
            counts[split] += 1
            record = {
                "sampleId": sample_id,
                "groupId": sample_id,
                "sourceId": sample_id,
                "split": split,
                "audioPath": str(wav_path.resolve()),
                "reference": str(row["transcription"]).strip(),
                "domain": spec["domain"],
                "rightsDecision": args.rights_decision,
                "licenseId": spec["license"],
                "durationSeconds": round(len(array) / 16000, 3),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "rows": len(dataset), "splits": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
