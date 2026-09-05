"""Collect frozen public FLEURS/Whisper/HuBERT evidence for offline error analysis.

No reference text is supplied to either acoustic model. Audio is not uploaded.
This is a bounded pilot, not an unbiased population-level accuracy estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
import time
from pathlib import Path


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=24)
    parser.add_argument("--phone-model", default="prj-beatrice/japanese-hubert-base-phoneme-ctc-v4")
    args = parser.parse_args()
    if not 1 <= args.per_split <= 64:
        parser.error("per-split must be in [1, 64]")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pyarrow.parquet as pq
    import pyopenjtalk
    import soundfile as sf
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import HubertForCTC, Wav2Vec2FeatureExtractor

    from semantic_asr.api import load_transcriber, transcribe
    from semantic_asr.revisions import FASTER_WHISPER_MODEL_REVISIONS

    torch.set_num_threads(2)
    torch.manual_seed(17)
    hub = HfApi()
    dataset = "google/fleurs"
    ds_info = hub.dataset_info(dataset)
    model_info = hub.model_info(args.phone_model)
    for revision in (ds_info.sha, model_info.sha):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError("expected an immutable Hub commit")
    ds_revision, phone_revision = ds_info.sha, model_info.sha
    files = [x.rfilename for x in ds_info.siblings]
    manifest = {
        "schema": "semantic-asr-public-phonetic-probe-v1",
        "dataset": dataset,
        "dataset_revision": ds_revision,
        "subset": "ja_jp",
        "data_license": "CC-BY-4.0",
        "attribution": "FLEURS, Google, Conneau et al. (2022)",
        "phone_model": args.phone_model,
        "phone_revision": phone_revision,
        "phone_model_license": "Apache-2.0",
        "base_profile": "cpu-ja-v1",
        "whisper_revision": FASTER_WHISPER_MODEL_REVISIONS["large-v3-turbo"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "requested_per_split": args.per_split,
        "selection": "first distinct-text-ID rows in sorted pinned parquet paths; 2-20 s",
        "reference_used_during_inference": False,
        "speaker_disjoint_verified": False,
        "pretraining_contamination_excluded": False,
        "context_policy": "isolated clips, no unrelated adjacent rows or reference as context",
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "faster-whisper",
                "ctranslate2",
                "pyopenjtalk-plus",
                "pyarrow",
                "numpy",
            )
        },
        "records": [],
    }
    write_json(out / "manifest.json", manifest)
    print(
        "FROZEN",
        json.dumps(
            {k: manifest[k] for k in ("dataset_revision", "phone_revision", "source_commit")}
        ),
        flush=True,
    )
    print("PARQUETS", [x for x in files if "ja_jp" in x and x.endswith(".parquet")], flush=True)
    vocab_path = hf_hub_download(args.phone_model, "vocab.json", revision=phone_revision)
    vocab = json.loads(Path(vocab_path).read_text())
    write_json(out / "vocab.json", vocab)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.phone_model, revision=phone_revision)
    phone_model = HubertForCTC.from_pretrained(
        args.phone_model, revision=phone_revision, use_safetensors=True
    ).eval()
    write_json(out / "phone-config.json", phone_model.config.to_dict())
    if sorted(vocab.values()) != list(range(phone_model.config.vocab_size)):
        raise RuntimeError("phone vocabulary and acoustic head differ")
    inverse = {v: k for k, v in vocab.items()}
    warm = load_transcriber("cpu-ja-v1")
    # Initialize the text frontend explicitly; it never sees the audio model input.
    pyopenjtalk.g2p("音声認識")
    seen_ids, seen_audio, seen_text = set(), set(), set()
    for split in ("validation", "test"):
        candidates = sorted(
            x for x in files if x.endswith(".parquet") and "ja_jp/" in x and f"/{split}" in x
        )
        if not candidates:
            raise RuntimeError(f"no pinned Japanese parquet files for {split}")
        count = 0
        for parquet_name in candidates:
            local = hf_hub_download(
                dataset, parquet_name, repo_type="dataset", revision=ds_revision
            )
            parquet = pq.ParquetFile(local)
            for batch in parquet.iter_batches(batch_size=8):
                for raw in batch.to_pylist():
                    ns = int(raw["num_samples"])
                    if not 32000 <= ns <= 320000 or raw["id"] in seen_ids:
                        continue
                    reference = raw["raw_transcription"] or raw["transcription"]
                    text_key = hashlib.sha256(reference.encode()).hexdigest()
                    if text_key in seen_text:
                        continue
                    audio, rate = sf.read(io.BytesIO(raw["audio"]["bytes"]), dtype="float32")
                    if (
                        rate != 16000
                        or audio.ndim != 1
                        or len(audio) != ns
                        or not np.isfinite(audio).all()
                    ):
                        raise RuntimeError("FLEURS audio contract mismatch")
                    pcm_sha = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
                    if pcm_sha in seen_audio:
                        continue
                    seen_ids.add(raw["id"])
                    seen_audio.add(pcm_sha)
                    seen_text.add(text_key)
                    sample_id = f"{split}-{count:03d}"
                    begun = time.perf_counter()
                    # Phonetic inference happens before candidates/references are processed.
                    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
                    with torch.inference_mode():
                        logits = (
                            phone_model(**inputs).logits[0].float().log_softmax(-1).cpu().numpy()
                        )
                    acoustic_seconds = time.perf_counter() - begun
                    np.savez_compressed(out / f"{sample_id}.npz", log_probs=logits)
                    phone_ids = logits.argmax(-1)
                    greedy = [
                        inverse[int(k)]
                        for i, k in enumerate(phone_ids)
                        if (i == 0 or k != phone_ids[i - 1])
                        and int(k) != phone_model.config.pad_token_id
                    ]
                    begun = time.perf_counter()
                    result = transcribe(audio, transcriber=warm)
                    decode_seconds = time.perf_counter() - begun
                    if len(result.longform.segments) != 1:
                        raise RuntimeError("probe must use one window per clip")
                    observed = result.longform.segments[0].observed
                    rows = []
                    for candidate in observed.candidates:
                        rows.append(
                            {
                                "id": candidate.candidate_id,
                                "text": candidate.text,
                                "phone_symbols": pyopenjtalk.g2p(candidate.text).split(),
                                "kana": pyopenjtalk.g2p(candidate.text, kana=True),
                                "avg_logprob": candidate.avg_logprob,
                                "sequence_score": candidate.sequence_score,
                                "acoustic": candidate.acoustic,
                                "rank": candidate.rank,
                                "evidence": candidate.as_dict(),
                            }
                        )
                    record = {
                        "id": sample_id,
                        "split": split,
                        "source_id": int(raw["id"]),
                        "source_path": Path(raw["path"]).name,
                        "duration_seconds": len(audio) / rate,
                        "pcm_sha256": pcm_sha,
                        "source_audio_sha256": result.source_audio_sha256,
                        "reference": reference,
                        "reference_phones": pyopenjtalk.g2p(reference).split(),
                        "reference_kana": pyopenjtalk.g2p(reference, kana=True),
                        "reference_role": "evaluation-only G2P proxy; not verified pronunciation",
                        "baseline_id": observed.selected_candidate_id,
                        "baseline_text": observed.text,
                        "phone_greedy": greedy,
                        "candidates": rows,
                        "phone_seconds": acoustic_seconds,
                        "asr_seconds": decode_seconds,
                        "posterior_sha256": hashlib.sha256(
                            (out / f"{sample_id}.npz").read_bytes()
                        ).hexdigest(),
                    }
                    write_json(out / f"{sample_id}.json", record)
                    manifest["records"].append(
                        {
                            "id": sample_id,
                            "split": split,
                            "pcm_sha256": pcm_sha,
                            "record_sha256": hashlib.sha256(
                                (out / f"{sample_id}.json").read_bytes()
                            ).hexdigest(),
                        }
                    )
                    write_json(out / "manifest.json", manifest)
                    print(
                        "SAMPLE",
                        sample_id,
                        "candidates",
                        len(rows),
                        "seconds",
                        round(decode_seconds, 2),
                        flush=True,
                    )
                    count += 1
                    if count >= args.per_split:
                        break
                if count >= args.per_split:
                    break
            if count >= args.per_split:
                break
        if count != args.per_split:
            raise RuntimeError(f"incomplete {split}: {count}/{args.per_split}")
    manifest["complete"] = True
    write_json(out / "manifest.json", manifest)
    print("COMPLETE", len(manifest["records"]), flush=True)


if __name__ == "__main__":
    main()
