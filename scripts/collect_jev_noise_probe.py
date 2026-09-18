"""Offline noise sensitivity on already collected real speech, not new speakers.

No API or new model download. Corrupted recordings retain parent identity. Gaussian
noise is synthetic; this is NOT a naturally noisy conversation benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path

from run_jev_phonetic_shadow import edits, file_digest, normalized, save


def corrupt(audio, snr_db: float, seed: int):
    """Add exact-power seeded noise, then canonicalize shared PCM16 samples."""
    import numpy as np

    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("finite nonempty mono audio required")
    if snr_db not in {10.0, 20.0}:
        raise ValueError("only preregistered 10/20 dB conditions are allowed")
    signal = audio.astype("float64")
    power = float(np.mean(signal * signal))
    if power <= 0:
        raise ValueError("cannot define SNR for silent source")
    noise = np.random.default_rng(seed).standard_normal(len(signal))
    noise *= np.sqrt(power / (10 ** (snr_db / 10)) / np.mean(noise * noise))
    mixed = signal + noise
    clipped = int(np.count_nonzero((mixed < -1) | (mixed > 32767 / 32768)))
    pcm = np.rint(np.clip(mixed, -1, 32767 / 32768) * 32768).astype("<i2")
    waveform = (pcm.astype("float32") / 32768).astype("float32")
    actual_noise = waveform.astype("float64") - signal
    achieved = float(10 * np.log10(power / np.mean(actual_noise * actual_noise)))
    return waveform, {"target_snr_db": snr_db, "achieved_snr_db": achieved,
                      "seed": seed, "clipped_samples": clipped,
                      "pcm16_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                      "float_pcm_sha256": hashlib.sha256(waveform.astype("<f4").tobytes()).hexdigest()}


def run(args):
    if not args.allow_public_fleurs:
        raise ValueError("explicit public FLEURS permission is required")
    source, out = Path(args.probe_dir), Path(args.output_dir)
    if out.exists():
        raise FileExistsError("do not overwrite an experiment")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("dataset") != "google/fleurs" or manifest.get("data_license") != "CC-BY-4.0":
        raise ValueError("requires the completed reviewed public FLEURS collection")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import numpy as np
    import pyarrow.parquet as pq
    import pyopenjtalk
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import HubertForCTC, Wav2Vec2FeatureExtractor

    from semantic_asr.api import load_transcriber, transcribe

    entries = [r for r in manifest["records"] if r["split"] == "validation"][:args.parent_records]
    parents = {}
    for entry in entries:
        file = source / (entry["id"] + ".json")
        if file_digest(file) != entry["record_sha256"]:
            raise ValueError("parent record digest mismatch")
        parent = json.loads(file.read_text(encoding="utf-8"))
        parents[parent["source_id"]] = parent
    torch.set_num_threads(2)
    torch.manual_seed(17)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(manifest["phone_model"],
        revision=manifest["phone_revision"], local_files_only=True)
    phone_model = HubertForCTC.from_pretrained(manifest["phone_model"],
        revision=manifest["phone_revision"], local_files_only=True, use_safetensors=True).eval()
    vocab = json.loads((source / "vocab.json").read_text(encoding="utf-8"))
    inverse = {index: symbol for symbol, index in vocab.items()}
    blank = phone_model.config.pad_token_id
    warm = load_transcriber("cpu-ja-v1")
    parquet = hf_hub_download("google/fleurs", "parquet-data/ja_jp/validation-00000-of-00001.parquet",
        repo_type="dataset", revision=manifest["dataset_revision"], local_files_only=True)
    out.mkdir(parents=True)
    receipt = {"schema": "jev-offline-noise-v1", "status": "running",
        "source_manifest_sha256": file_digest(source / "manifest.json"),
        "runner_sha256": file_digest(Path(__file__)), "dataset_revision": manifest["dataset_revision"],
        "phone_model": manifest["phone_model"], "phone_revision": manifest["phone_revision"],
        "whisper_revision": manifest["whisper_revision"], "data_license": "CC-BY-4.0",
        "attribution": "Google / Conneau et al., FLEURS (2022)",
        "synthetic_corruption_of_recorded_speech": True, "new_independent_recordings": 0,
        "reference_used_during_inference": False, "api_calls": 0, "audio_uploaded": False,
        "model_downloads_allowed": False, "max_derived_conditions": 2 * len(parents),
        "max_wall_seconds": args.max_wall_seconds, "evaluation_role": "development-exposed"}
    save(out / "receipt.json", receipt)
    results, completed = [], set()
    start = time.monotonic()
    try:
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                sid = raw["id"]
                if sid not in parents or sid in completed:
                    continue
                parent = parents[sid]
                audio, rate = sf.read(io.BytesIO(raw["audio"]["bytes"]), dtype="float32")
                if rate != 16000 or audio.ndim != 1:
                    raise ValueError("source audio contract mismatch")
                parent_hash = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
                if parent_hash != parent["pcm_sha256"]:
                    continue
                for condition, snr in enumerate((20.0, 10.0)):
                    if time.monotonic() - start >= args.max_wall_seconds:
                        raise TimeoutError("finite noise experiment wall budget exhausted")
                    seed = 20260919 + 2 * int(parent["id"].split("-")[-1]) + condition
                    waveform, corruption = corrupt(audio, snr, seed)
                    began = time.monotonic()
                    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt")
                    with torch.inference_mode():
                        logs = phone_model(**inputs).logits[0].float().log_softmax(-1).cpu().numpy()
                    phones = logs.argmax(-1)
                    greedy = [inverse[int(p)] for i, p in enumerate(phones)
                              if (i == 0 or p != phones[i - 1]) and int(p) != blank]
                    phone_seconds = time.monotonic() - began
                    began = time.monotonic()
                    asr = transcribe(waveform, transcriber=warm)
                    asr_seconds = time.monotonic() - began
                    if len(asr.longform.segments) != 1:
                        raise ValueError("partial/segmented output cannot replace a full parent")
                    observed = asr.longform.segments[0].observed
                    label = parent["id"] + "-snr" + str(int(snr))
                    np.savez_compressed(out / (label + ".npz"), log_probs=logs)
                    # Reference is evaluation only, after BOTH model inferences.
                    reference = normalized(parent["reference"])
                    ref_phones = parent["reference_phones"]
                    predicted_phones = pyopenjtalk.g2p(observed.text).split()
                    result = {"id": label, "parent_id": parent["id"], "parent_pcm_sha256": parent_hash,
                        "source_id": sid, "duration_seconds": len(audio) / rate, "corruption": corruption,
                        "reference_characters": len(reference), "clean_errors": edits(reference, normalized(parent["baseline_text"])),
                        "noisy_errors": edits(reference, normalized(observed.text)),
                        "clean_text": parent["baseline_text"], "noisy_text": observed.text,
                        "phone_greedy": greedy, "candidate_count": len(observed.candidates),
                        "observed_phone_changes_from_clean": edits(parent["phone_greedy"], greedy),
                        "g2p_reference_proxy_phone_count": len(ref_phones),
                        "clean_g2p_proxy_errors": edits(ref_phones, pyopenjtalk.g2p(parent["baseline_text"]).split()),
                        "noisy_g2p_proxy_errors": edits(ref_phones, predicted_phones),
                        "g2p_proxy_is_not_gold_phonetic_annotation": True,
                        "phone_seconds": phone_seconds, "asr_seconds": asr_seconds,
                        "asr_source_audio_sha256": asr.source_audio_sha256,
                        "posterior_sha256": file_digest(out / (label + ".npz"))}
                    save(out / (label + ".json"), result)
                    results.append(result)
                    save(out / "results.json", results)
                    print(json.dumps({"id": label, "clean_errors": result["clean_errors"],
                        "noisy_errors": result["noisy_errors"], "completed": len(results)}), flush=True)
                completed.add(sid)
            if len(completed) == len(parents):
                break
        receipt["status"] = "completed" if len(completed) == len(parents) else "partial-missing-parent"
    except Exception as exc:
        receipt.update({"status": "partial" if isinstance(exc, TimeoutError) else "failed",
                        "error_type": type(exc).__name__})
        raise
    finally:
        receipt.update({"completed_conditions": len(results), "completed_parents": len(completed),
                        "wall_seconds": time.monotonic() - start})
        save(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-public-fleurs", action="store_true")
    parser.add_argument("--parent-records", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.parent_records <= 12 or not 1 <= args.max_wall_seconds <= 900:
        parser.error("requires 1..12 parents and 1..900 wall seconds")
    print(json.dumps(run(args), indent=2))
