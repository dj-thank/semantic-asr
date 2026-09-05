"""Second bounded experiment: retain beam-5 baseline and add beam-12 acoustic paths.

Development uses the prior development clips. Evaluation excludes EVERY previously
used text ID, waveform and reference surface. Reference text never enters inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import time
from dataclasses import replace
from pathlib import Path

from collect_phonetic_public_probe import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.per_split <= 32:
        parser.error("per-split must be in [1,32]")
    import numpy as np
    import pyarrow.parquet as pq
    import pyopenjtalk
    import soundfile as sf
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import HubertForCTC, Wav2Vec2FeatureExtractor
    from semantic_asr.adapters import DecodeRequest
    from semantic_asr.api import _materialise_audio, load_transcriber, transcribe

    torch.set_num_threads(2)
    torch.manual_seed(17)
    prior = json.loads((args.prior / "manifest.json").read_text())
    if not prior.get("complete"):
        raise ValueError("prior inference must be complete")
    excluded_ids, excluded_audio, excluded_text = set(), set(), set()
    dev_rows = {}
    for item in prior["records"]:
        path = args.prior / f"{item['id']}.json"
        if digest(path) != item["record_sha256"]:
            raise ValueError("prior evidence hash mismatch")
        row = json.loads(path.read_text())
        excluded_ids.add(row["source_id"])
        excluded_audio.add(row["pcm_sha256"])
        excluded_text.add(row["reference"])
        if row["split"] == "validation":
            dev_rows[row["source_id"]] = row["pcm_sha256"]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("do not overwrite an experiment")
    model_id, revision = prior["phone_model"], prior["phone_revision"]
    vocab = json.loads(Path(hf_hub_download(model_id,"vocab.json",revision=revision)).read_text())
    write_json(out/"vocab.json",vocab)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id,revision=revision)
    phone_model = HubertForCTC.from_pretrained(model_id,revision=revision,use_safetensors=True).eval()
    inverse = {i:s for s,i in vocab.items()}
    warm = load_transcriber("cpu-ja-v1")
    info = HfApi().dataset_info(prior["dataset"],revision=prior["dataset_revision"])
    files = [f.rfilename for f in info.siblings]
    manifest = {k:v for k,v in prior.items() if k not in ("records","complete")}
    manifest.update({
        "experiment":"candidate-expansion-wave2",
        "prior_manifest_sha256":digest(args.prior/"manifest.json"),
        "requested_per_split":args.per_split,
        "selection":"prior development audio; fresh test excludes all prior text IDs/audio/text",
        "prior_excluded_source_ids":sorted(excluded_ids),
        "candidate_expansion":{"baseline_beam":5,"extra_beam":12,"extra_hypotheses":12,
            "same_model_second_decode":True,"reference_prompted":False},
        "phone_input":"same canonical PCM16 WAV samples as Whisper",
        "packages":{p:importlib.metadata.version(p) for p in prior["packages"]},
        "records":[],
    })
    import subprocess
    manifest["source_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    write_json(out/"manifest.json",manifest)
    seen_ids, seen_audio, seen_text = set(), set(), set()
    for split in ("validation","test"):
        count = 0
        parquets = sorted(f for f in files if "ja_jp/" in f and f"/{split}" in f and f.endswith(".parquet"))
        for name in parquets:
            file = hf_hub_download(prior["dataset"],name,repo_type="dataset",revision=prior["dataset_revision"])
            for batch in pq.ParquetFile(file).iter_batches(batch_size=8):
                for raw in batch.to_pylist():
                    uid = int(raw["id"])
                    if not 32000 <= int(raw["num_samples"]) <= 320000 or uid in seen_ids:
                        continue
                    ref = raw["raw_transcription"] or raw["transcription"]
                    if split == "validation" and uid not in dev_rows:
                        continue
                    if split == "test" and (uid in excluded_ids or ref in excluded_text):
                        continue
                    audio, rate = sf.read(io.BytesIO(raw["audio"]["bytes"]),dtype="float32")
                    if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                        raise ValueError("unexpected audio format")
                    pcm_sha = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
                    if pcm_sha in seen_audio or ref in seen_text:
                        continue
                    if split == "validation" and pcm_sha != dev_rows[uid]:
                        continue
                    if split == "test" and pcm_sha in excluded_audio:
                        continue
                    seen_ids.add(uid); seen_audio.add(pcm_sha); seen_text.add(ref)
                    sid = f"{split}-{count:03d}"
                    path, temporary = _materialise_audio(audio)
                    try:
                        canonical, _ = sf.read(path,dtype="float32")
                        begun = time.perf_counter()
                        inputs = extractor(canonical,sampling_rate=16000,return_tensors="pt")
                        with torch.inference_mode():
                            logs = phone_model(**inputs).logits[0].float().log_softmax(-1).numpy()
                        phone_seconds = time.perf_counter()-begun
                        np.savez_compressed(out/f"{sid}.npz",log_probs=logs)
                        begun = time.perf_counter()
                        result = transcribe(path,transcriber=warm)
                        observed = result.longform.segments[0].observed
                        base_seconds = time.perf_counter()-begun
                        begun = time.perf_counter()
                        extra = warm.base_adapter.decode(DecodeRequest(
                            str(path),beam_size=12,hypotheses=12,start_ms=0,end_ms=result.duration_ms,
                        ))
                        extra_seconds = time.perf_counter()-begun
                    finally:
                        temporary.unlink(missing_ok=True)
                    candidates = list(observed.candidates)
                    texts = {c.text for c in candidates}
                    for candidate in extra:
                        if candidate.text not in texts:
                            texts.add(candidate.text)
                            candidates.append(replace(candidate,candidate_id="expanded:"+candidate.candidate_id,
                                metadata={**candidate.metadata,"decodeNamespace":"wave2-beam12",
                                    "decodeStartMs":0,"decodeEndMs":result.duration_ms}))
                    rows = [{"id":c.candidate_id,"text":c.text,"phone_symbols":pyopenjtalk.g2p(c.text).split(),
                        "kana":pyopenjtalk.g2p(c.text,kana=True),"avg_logprob":c.avg_logprob,
                        "sequence_score":c.sequence_score,"acoustic":c.acoustic,"rank":c.rank,
                        "evidence":c.as_dict()} for c in candidates]
                    ids = logs.argmax(-1)
                    record = {"id":sid,"split":split,"source_id":uid,"source_path":Path(raw["path"]).name,
                        "duration_seconds":len(audio)/rate,"pcm_sha256":pcm_sha,
                        "source_audio_sha256":result.source_audio_sha256,"reference":ref,
                        "reference_phones":pyopenjtalk.g2p(ref).split(),"reference_kana":pyopenjtalk.g2p(ref,kana=True),
                        "reference_role":"evaluation-only G2P proxy, not gold pronunciation",
                        "baseline_id":observed.selected_candidate_id,"baseline_text":observed.text,
                        "baseline_candidate_ids":[c.candidate_id for c in observed.candidates],
                        "phone_greedy":[inverse[int(k)] for i,k in enumerate(ids) if (i==0 or k!=ids[i-1]) and int(k)!=phone_model.config.pad_token_id],
                        "candidates":rows,"phone_seconds":phone_seconds,"asr_seconds":base_seconds,
                        "expansion_seconds":extra_seconds,"posterior_sha256":digest(out/f"{sid}.npz")}
                    write_json(out/f"{sid}.json",record)
                    manifest["records"].append({"id":sid,"split":split,"pcm_sha256":pcm_sha,"record_sha256":digest(out/f"{sid}.json")})
                    write_json(out/"manifest.json",manifest)
                    print("EXPANDED",sid,"candidates",len(observed.candidates),"->",len(rows),flush=True)
                    count += 1
                    if count >= args.per_split: break
                if count >= args.per_split: break
            if count >= args.per_split: break
        if count != args.per_split:
            raise ValueError(f"incomplete split: {split} {count}")
    manifest["complete"] = True
    write_json(out/"manifest.json",manifest)


if __name__ == "__main__":
    main()
