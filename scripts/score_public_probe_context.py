"""Reference-blind full-candidate LM scoring on a previously frozen public probe.

This adds a linguistic preference, not acoustic evidence or correctness probability.
The evaluator must choose its policy on development data before viewing test outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    args = parser.parse_args()
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from semantic_asr.sequence_scorers import (
        CausalScoringConfig,
        TextCandidate,
        TransformersCausalSequenceScorer,
    )

    torch.set_num_threads(2)
    torch.manual_seed(17)
    manifest_bytes = (args.probe / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get("complete") or manifest["reference_used_during_inference"]:
        raise ValueError("requires a complete reference-blind probe")
    revision = HfApi().model_info(args.model).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("LM revision must be immutable")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, use_safetensors=True, trust_remote_code=False,
        torch_dtype=torch.float32,
    ).eval()
    scorer = TransformersCausalSequenceScorer(
        model, tokenizer,
        CausalScoringConfig(model_name=args.model, model_revision=revision, batch_size=1),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "semantic-asr-public-context-probe-v1",
        "probe_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "model": args.model,
        "revision": revision,
        "license": "Apache-2.0",
        "context": "日本語の文章：\n",
        "context_source": "fixed neutral prefix; never neighboring unrelated clips or references",
        "reference_conditioned": False,
        "source_role": "whole-candidate linguistic preference, not acoustic proof",
        "runtime": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "records": [],
    }
    for item in manifest["records"]:
        raw = (args.probe / f"{item['id']}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["record_sha256"]:
            raise ValueError("probe record integrity mismatch")
        # Only the candidate list is used; reference fields never reach the scorer.
        candidates = json.loads(raw)["candidates"]
        scores = scorer.score(
            [TextCandidate(c["id"], c["text"]) for c in candidates],
            context=result["context"],
        )
        result["records"].append({
            "id": item["id"],
            "record_sha256": item["record_sha256"],
            "scores": [
                {"candidate_id": s.candidate_id, "sum_logprob": s.cumulative.value,
                 "average_logprob": s.average.value, "token_count": s.token_count}
                for s in scores
            ],
        })
        (args.output / "context-scores.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("SCORED", item["id"], len(scores), flush=True)
    result["complete"] = True
    (args.output / "context-scores.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
