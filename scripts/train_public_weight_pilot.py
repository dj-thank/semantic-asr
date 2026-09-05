"""Bounded real-weight feasibility pilot, NOT a release-quality training recipe.

Only the official FLEURS train split is opened. A text-ID-disjoint portion is
reserved as development, not an unseen publication test. Frozen HuBERT features
train new phone/mora CTC heads; actual Whisper candidates train Qwen LoRA by
pairwise preference. References never enter the candidate scorer. There is no
hyperparameter search, auto-promotion, resume claim, or continuous background loop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import subprocess
import time
from pathlib import Path

from semantic_asr.contracts import sha256_json

DATASET = "google/fleurs"
DATA_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
PHONE = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
PHONE_REV = "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
LM = "Qwen/Qwen3-0.6B-Base"
LM_REV = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in sorted(items):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_step(loss, model, optimizer):
    import torch

    if not torch.isfinite(loss):
        raise ValueError("non-finite training loss")
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in trainable):
        raise ValueError("no trainable gradients")
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def candidate_tokens(tokenizer, candidate):
    """A fixed, separately tokenized prefix/target contract; no reference input."""
    prefix = (
        f"日本語ASR候補。音響スコア: {candidate['acoustic']:.4f}\n読み: {candidate['kana']}\n文章: "
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    text_ids = tokenizer.encode(candidate["text"], add_special_tokens=False)
    if not prefix_ids or not text_ids or len(prefix_ids) + len(text_ids) > 256:
        raise ValueError("candidate is empty or exceeds the fixed 256-token limit")
    return prefix_ids + text_ids, len(prefix_ids)


def language_score(model, encoded):
    import torch
    import torch.nn.functional as F

    ids, prefix = encoded
    tokens = torch.tensor([ids], dtype=torch.long)
    logits = model(input_ids=tokens, use_cache=False).logits[0, :-1].float()
    targets = tokens[0, 1:]
    # Score only the complete candidate, not acoustic/G2P metadata in the prefix.
    return -F.cross_entropy(logits[prefix - 1 :], targets[prefix - 1 :], reduction="mean")


def acoustic_model(hidden_size, vocab):
    from torch import nn

    from semantic_asr.training import SemanticASRMultiTask

    model = SemanticASRMultiTask(
        nn.Identity(),
        hidden_size=hidden_size,
        phone_vocab_size=len(vocab["phone"]),
        mora_vocab_size=len(vocab["mora"]),
    )
    # No fabricated accent/F0/boundary/preservation supervision or exported heads.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("phone_head.", "mora_head.")))
    return model


def acoustic_metrics(model, records, features, vocab):
    import torch

    from semantic_asr.evaluation import edit_distance

    totals = {k: {"errors": 0, "labels": 0} for k in ("phone", "mora")}
    losses = []
    model.eval()
    with torch.no_grad():
        for row in records:
            output = model(
                input_features=features[row["id"]].unsqueeze(0),
                phone_labels=torch.tensor([row["phone_ids"]]),
                mora_labels=torch.tensor([row["mora_ids"]]),
            )
            losses.append(float(output.loss))
            for kind, logits in (("phone", output.phone_logits), ("mora", output.mora_logits)):
                ids = logits[0].argmax(-1).tolist()
                decoded = [k for i, k in enumerate(ids) if k != 0 and (i == 0 or k != ids[i - 1])]
                target = row[f"{kind}_ids"]
                totals[kind]["errors"] += edit_distance(target, decoded)
                totals[kind]["labels"] += len(target)
    return {
        "loss": sum(losses) / len(losses),
        "proxy_error_counts": totals,
        "warning": "G2P-derived weak labels, NOT gold phonetic annotations",
    }


def prepare(out, exclusions, check_time):
    import numpy as np
    import pyarrow.parquet as pq
    import pyopenjtalk
    import soundfile as sf
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import HubertForCTC, Wav2Vec2FeatureExtractor

    from semantic_asr.api import _materialise_audio, load_transcriber, transcribe
    from semantic_asr.mora_phonology import MORA_PHONES, phones_to_moras
    from semantic_asr.rights import RightsRecord, RightsRegistry

    rights = RightsRegistry(
        [
            RightsRecord(
                "fleurs",
                DATASET,
                "https://huggingface.co/datasets/google/fleurs",
                DATA_REV,
                "CC-BY-4.0",
                "https://creativecommons.org/licenses/by/4.0/",
                "allow",
                "allow",
                "deny",
                "deny",
                "FLEURS, Google, Conneau et al. (2022)",
                "2026-09-05",
                "Pilot trains on licensed public speech; no raw audio or speaker IDs exported.",
            )
        ]
    )
    rights.require("fleurs", "train")
    rights.require("fleurs", "derive_features")
    hub = HfApi()
    info = hub.dataset_info(DATASET, revision=DATA_REV)
    if info.card_data.get("license") != "cc-by-4.0":
        raise PermissionError("pinned dataset license mismatch")
    for name, revision in ((PHONE, PHONE_REV), (LM, LM_REV)):
        if hub.model_info(name, revision=revision).card_data.get("license") != "apache-2.0":
            raise PermissionError("pinned base model license mismatch")
    phone_vocab = json.loads(
        Path(hf_hub_download(PHONE, "vocab.json", revision=PHONE_REV)).read_text()
    )
    if phone_vocab.get("PAD") != 0:
        raise ValueError("expected CTC blank zero")
    keys = {" ".join(p) for p in MORA_PHONES.values()} | {"pau", "sil"}
    for phones in tuple(MORA_PHONES.values()):
        if phones[-1] in ("i", "u"):
            keys.add(" ".join((*phones[:-1], phones[-1].upper())))
    mora_vocab = {k: i for i, k in enumerate(["<blank>", *sorted(keys)])}
    vocab = {"phone": phone_vocab, "mora": mora_vocab}
    write_json(out / "vocab.json", vocab)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONE, revision=PHONE_REV)
    encoder = HubertForCTC.from_pretrained(
        PHONE, revision=PHONE_REV, use_safetensors=True
    ).hubert.eval()
    encoder.requires_grad_(False)
    initial_encoder = tensor_digest(encoder.named_parameters())
    warm = load_transcriber("cpu-ja-v1")
    records, features, seen_ids, seen_audio, seen_text = [], {}, set(), set(), set()
    counts = {"train": 0, "development": 0}
    ignored = {"duration": 0, "excluded": 0, "unsupported": 0, "duplicate": 0}
    total_seconds = 0.0
    files = sorted(
        f.rfilename
        for f in info.siblings
        if "ja_jp/train" in f.rfilename and f.rfilename.endswith(".parquet")
    )
    if not files:
        raise ValueError("pinned official Japanese train parquet is missing")
    for file in files:
        local = hf_hub_download(DATASET, file, repo_type="dataset", revision=DATA_REV)
        for batch in pq.ParquetFile(local).iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                check_time()
                uid, count = int(raw["id"]), int(raw["num_samples"])
                role = "development" if uid % 5 == 0 else "train"
                if counts[role] >= (32 if role == "train" else 8):
                    continue
                if not 32000 <= count <= 160000:
                    ignored["duration"] += 1
                    continue
                reference = raw["raw_transcription"] or raw["transcription"]
                text_hash = hashlib.sha256(reference.encode()).hexdigest()
                if uid in exclusions["source_ids"] or text_hash in exclusions["reference_sha256"]:
                    ignored["excluded"] += 1
                    continue
                audio, rate = sf.read(io.BytesIO(raw["audio"]["bytes"]), dtype="float32")
                pcm_hash = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
                if (
                    rate != 16000
                    or audio.ndim != 1
                    or len(audio) != count
                    or not np.isfinite(audio).all()
                ):
                    raise ValueError("audio sample contract mismatch")
                if pcm_hash in exclusions["pcm_sha256"]:
                    ignored["excluded"] += 1
                    continue
                if uid in seen_ids or pcm_hash in seen_audio or text_hash in seen_text:
                    ignored["duplicate"] += 1
                    continue
                phones = tuple(pyopenjtalk.g2p(reference).split())
                moras = phones_to_moras(phones)
                if (
                    not phones
                    or any(p not in phone_vocab for p in phones)
                    or any(" ".join(u.phones) not in mora_vocab for u in moras)
                ):
                    ignored["unsupported"] += 1
                    continue
                path, temporary = _materialise_audio(audio)
                try:
                    canonical, _ = sf.read(path, dtype="float32")
                    with torch.inference_mode():
                        hidden = encoder(
                            **extractor(canonical, sampling_rate=16000, return_tensors="pt")
                        ).last_hidden_state[0].cpu()
                    result = transcribe(path, transcriber=warm)
                finally:
                    temporary.unlink(missing_ok=True)
                sid = f"{role}-{counts[role]:03d}"
                observed = result.longform.segments[0].observed
                rows = [
                    {
                        "id": c.candidate_id,
                        "text": c.text,
                        "kana": pyopenjtalk.g2p(c.text, kana=True),
                        "acoustic": c.avg_logprob if c.avg_logprob is not None else c.acoustic,
                    }
                    for c in observed.candidates
                ]
                if any(c["acoustic"] is None or not math.isfinite(c["acoustic"]) for c in rows):
                    raise ValueError("missing/non-finite acoustic candidate score")
                records.append(
                    {
                        "id": sid,
                        "role": role,
                        "official_split": "train",
                        "source_id": uid,
                        "source_audio_sha256": result.source_audio_sha256,
                        "pcm_sha256": pcm_hash,
                        "reference_sha256": text_hash,
                        "reference": reference,
                        "duration_seconds": count / rate,
                        "speaker_disjoint_verified": False,
                        "candidates": rows,
                        "baseline_id": observed.selected_candidate_id,
                        "phone_ids": [phone_vocab[p] for p in phones],
                        "mora_ids": [mora_vocab[" ".join(u.phones)] for u in moras],
                    }
                )
                features[sid] = hidden.clone()  # normal training tensor, not inference-mode storage
                counts[role] += 1
                total_seconds += count / rate
                if total_seconds > 400:
                    raise ValueError("pilot audio budget exceeded")
                seen_ids.add(uid)
                seen_audio.add(pcm_hash)
                seen_text.add(text_hash)
                print("PREPARED", sid, "candidates", len(rows), flush=True)
            if counts == {"train": 32, "development": 8}:
                break
        if counts == {"train": 32, "development": 8}:
            break
    if counts != {"train": 32, "development": 8}:
        raise ValueError(f"incomplete pilot split: {counts}")
    if initial_encoder != tensor_digest(encoder.named_parameters()):
        raise ValueError("frozen acoustic encoder changed")
    manifest = {
        "schema": "semantic-asr-real-weight-pilot-v1",
        "dataset": DATASET,
        "dataset_revision": DATA_REV,
        "roles": counts,
        "records": records,
        "audio_seconds": total_seconds,
        "ignored": ignored,
        "exclusions": exclusions,
        "split_rule": (
            "official train only; source_id mod 5 == 0 is development; unique text IDs/audio/text"
        ),
        "encoder_digest": initial_encoder,
        "encoder_unchanged": True,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "phone_model": PHONE,
        "phone_revision": PHONE_REV,
        "lm": LM,
        "lm_revision": LM_REV,
        "weights_license": "Apache-2.0; retain base notices and FLEURS CC-BY-4.0 attribution",
        "no_raw_audio_exported": True,
        "heldout_test_opened": False,
        "reference_used_in_candidate_inference": False,
        "development_is_publication_test": False,
    }
    write_json(out / "manifest.json", manifest)
    del encoder, warm, extractor
    gc.collect()
    return manifest, features, vocab


def run(out, exclusions, check_time):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from semantic_asr.candidate_pool import lenient_surface_key
    from semantic_asr.evaluation import edit_distance

    torch.set_num_threads(2)
    torch.manual_seed(17)
    manifest, features, vocab = prepare(out, exclusions, check_time)
    records = manifest["records"]
    train = [r for r in records if r["role"] == "train"]
    dev = [r for r in records if r["role"] == "development"]
    model = acoustic_model(next(iter(features.values())).shape[-1], vocab)
    initial = {
        k: v.detach().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("phone_head.", "mora_head."))
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003)
    acoustic_before = acoustic_metrics(model, dev, features, vocab)
    history = []
    for step in range(128):
        check_time()
        row = train[step % len(train)]
        model.train()
        output = model(
            input_features=features[row["id"]].unsqueeze(0),
            phone_labels=torch.tensor([row["phone_ids"]]),
            mora_labels=torch.tensor([row["mora_ids"]]),
        )
        norm = finite_step(output.loss, model, optimizer)
        history.append({"step": step + 1, "loss": float(output.loss.detach()), "grad_norm": norm})
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items() if k in initial}
    delta = sum(float((state[k] - initial[k]).square().sum()) for k in state) ** 0.5
    if not delta > 0:
        raise ValueError("acoustic weights did not change")
    save_file(state, str(out / "acoustic-heads.safetensors"))
    save_file(
        {"features": features[dev[0]["id"]].contiguous()},
        str(out / "reload-features.safetensors"),
    )
    model.eval()
    with torch.no_grad():
        sample = model(input_features=features[dev[0]["id"]].unsqueeze(0))
    save_file(
        {"phone": sample.phone_logits.contiguous(), "mora": sample.mora_logits.contiguous()},
        str(out / "reload-acoustic-logits.safetensors"),
    )
    acoustic = {
        "steps": 128,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_digest": tensor_digest(initial.items()),
        "trained_digest": tensor_digest(state.items()),
        "parameter_delta_l2": delta,
        "development_before": acoustic_before,
        "development_after": acoustic_metrics(model, dev, features, vocab),
        "history": history,
        "hidden_size": next(iter(features.values())).shape[-1],
        "encoder_unchanged": True,
    }
    write_json(out / "acoustic-training.json", acoustic)
    del model, optimizer, features, initial, state, sample
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(LM, revision=LM_REV, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        LM,
        revision=LM_REV,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    layers = [model.config.num_hidden_layers - 2, model.config.num_hidden_layers - 1]
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=layers,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.eval()
    frozen_before = tensor_digest(
        (n, p) for n, p in model.named_parameters() if not p.requires_grad
    )
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    tokenized = {}
    for row in records:
        tokenized[row["id"]] = {c["id"]: candidate_tokens(tokenizer, c) for c in row["candidates"]}
    # Ground truth is consulted only for training-pair labels and post-selection metrics.
    pairs = []
    for row in train:
        errors = {
            c["id"]: edit_distance(
                lenient_surface_key(row["reference"]), lenient_surface_key(c["text"])
            )
            for c in row["candidates"]
        }
        good = min(errors, key=lambda k: (errors[k], k))
        bad = max(errors, key=lambda k: (errors[k], k))
        if errors[good] < errors[bad]:
            pairs.append((row["id"], good, bad))
    if not pairs:
        raise ValueError("no non-tied real ASR training pairs; do not fabricate negative examples")

    def evaluate():
        results = []
        with torch.no_grad():
            for row in dev:
                scores = {
                    c["id"]: float(language_score(model, tokenized[row["id"]][c["id"]]))
                    for c in row["candidates"]
                }
                selected = max(scores, key=lambda k: (scores[k], k))
                text = next(c["text"] for c in row["candidates"] if c["id"] == selected)
                ref = lenient_surface_key(row["reference"])
                results.append(
                    {
                        "id": row["id"],
                        "selected_id": selected,
                        "scores": scores,
                        "errors": edit_distance(ref, lenient_surface_key(text)),
                        "characters": len(ref),
                    }
                )
        return results

    before = evaluate()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0002)
    history = []
    for step in range(16):
        check_time()
        sid, good, bad = pairs[step % len(pairs)]
        good_score = language_score(model, tokenized[sid][good])
        bad_score = language_score(model, tokenized[sid][bad])
        loss = torch.nn.functional.softplus(bad_score - good_score)
        norm = finite_step(loss, model, optimizer)
        history.append(
            {"step": step + 1, "sample_id": sid, "loss": float(loss.detach()), "grad_norm": norm}
        )
        print("LORA_UPDATE", step + 1, history[-1]["loss"], flush=True)
    trained = {n: p for n, p in model.named_parameters() if p.requires_grad}
    delta = sum(float((p.detach() - initial[n]).square().sum()) for n, p in trained.items()) ** 0.5
    if not delta > 0 or frozen_before != tensor_digest(
        (n, p) for n, p in model.named_parameters() if not p.requires_grad
    ):
        raise ValueError("LoRA did not change or frozen base changed")
    after = evaluate()
    model.save_pretrained(out / "lora", safe_serialization=True)
    probe = next(iter(tokenized[dev[0]["id"]].values()))
    with torch.no_grad():
        expected = float(language_score(model, probe))
    write_json(out / "reload-lora.json", {"encoded": probe, "expected": expected})
    write_json(
        out / "lora-training.json",
        {
            "steps": 16,
            "rank": 8,
            "layers": layers,
            "real_informative_train_pairs": len(pairs),
            "trainable_parameters": sum(p.numel() for p in trained.values()),
            "initial_digest": tensor_digest(initial.items()),
            "trained_digest": tensor_digest(trained.items()),
            "parameter_delta_l2": delta,
            "frozen_base_digest": frozen_before,
            "frozen_base_unchanged": True,
            "development_before": before,
            "development_after": after,
            "history": history,
            "limitation": (
                "1-seed ranker feasibility only; no full acoustic gate or publication test"
            ),
        },
    )


def verify(out):
    """Fresh-process proof that exported artifacts, not in-memory tensors, are used."""
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    vocab = json.loads((out / "vocab.json").read_text())
    info = json.loads((out / "acoustic-training.json").read_text())
    model = acoustic_model(info["hidden_size"], vocab)
    state = load_file(out / "acoustic-heads.safetensors")
    status = model.load_state_dict(state, strict=False)
    if status.unexpected_keys or any(
        n.startswith(("mora_head.", "phone_head.")) for n in status.missing_keys
    ):
        raise ValueError("incomplete acoustic checkpoint")
    if tensor_digest(state.items()) != info["trained_digest"]:
        raise ValueError("acoustic artifact identity mismatch")
    model.eval()
    with torch.no_grad():
        output = model(
            input_features=load_file(out / "reload-features.safetensors")["features"].unsqueeze(0)
        )
    expected = load_file(out / "reload-acoustic-logits.safetensors")
    acoustic_max = max(
        float((output.phone_logits - expected["phone"]).abs().max()),
        float((output.mora_logits - expected["mora"]).abs().max()),
    )
    if acoustic_max > 1e-6:
        raise ValueError("reloaded acoustic logits differ")
    del model
    base = AutoModelForCausalLM.from_pretrained(
        LM,
        revision=LM_REV,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base, out / "lora").eval()
    probe = json.loads((out / "reload-lora.json").read_text())
    with torch.no_grad():
        actual = float(language_score(model, probe["encoded"]))
    if abs(actual - probe["expected"]) > 1e-6:
        raise ValueError("reloaded LoRA score differs")
    write_json(
        out / "reload-verification.json",
        {
            "fresh_process": True,
            "acoustic_max_abs_difference": acoustic_max,
            "lora_score_abs_difference": abs(actual - probe["expected"]),
            "success": True,
            "promotion_approved": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()

    def check_time():
        if time.monotonic() - begun > 1800:
            raise TimeoutError("pilot exceeded 1800-second wall-clock launch budget")

    write_json(
        args.output / "protocol.json",
        {
            "seed": 17,
            "train": 32,
            "development": 8,
            "official_split": "train",
            "max_audio_seconds": 400,
            "max_wall_seconds": 1800,
            "acoustic_steps": 128,
            "lora_steps": 16,
            "acoustic_lr": 0.003,
            "lora_lr": 0.0002,
            "selection": "fixed final step, no dev checkpoint selection",
            "status": "preregistered-pilot",
            "full_issue_acceptance": False,
            "policy_changes": False,
        },
    )
    try:
        if args.exclusions is None:
            raise ValueError("cumulative exclusions file is required")
        exclusions = json.loads(args.exclusions.read_text())
        run(args.output, exclusions, check_time)
        write_json(
            args.output / "execution.json",
            {
                "status": "training-complete-reload-pending",
                "seconds": time.monotonic() - begun,
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in (
                        "torch",
                        "transformers",
                        "peft",
                        "safetensors",
                        "numpy",
                        "pyopenjtalk-plus",
                    )
                },
                "exclusions_digest": sha256_json(exclusions),
                "promotion_approved": False,
            },
        )
    except Exception as exc:
        write_json(
            args.output / "execution.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "seconds": time.monotonic() - begun,
                "promotion_approved": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
