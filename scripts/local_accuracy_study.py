"""Bounded local study: prepare, develop, fresh-check, report. No cloud compute."""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import html
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from semantic_asr.evaluation import normalize_characters
from semantic_asr.experiment_runner import _checkpoint_writer_lock
from semantic_asr.local_accuracy import measure, normalized_trial, summarize, surface_review

REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
TRIALS = {
    "trial1-itn": ("single", False),
    "trial2-itn-stop": ("single", True),
    "trial3-dual-itn-stop": ("dual", True),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc():
    return dt.datetime.now(dt.UTC).isoformat()


def check(root):
    state = read(root / "state.json")
    if time.time() > state["deadline_epoch"]:
        raise TimeoutError("six-hour study wall-clock budget exhausted")
    return state


def import_cohort(records, audio_root, destination, output_audio):
    import numpy as np
    import soundfile as sf

    rows = []
    seen_ids, seen_refs, seen_pcm = set(), set(), set()
    for row in records:
        path = audio_root / Path(row["audio_path"]).name
        if sha(path) != row["wav_sha256"]:
            raise ValueError("materialized WAV hash mismatch")
        if hashlib.sha256(row["reference"].encode()).hexdigest() != row["reference_sha256"]:
            raise ValueError("reference hash mismatch")
        pcm, sr = sf.read(path, dtype="float32")
        if sr != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("native audio contract mismatch")
        ph = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
        for value, seen in [
            (row["id"], seen_ids),
            (row["reference_sha256"], seen_refs),
            (ph, seen_pcm),
        ]:
            if value in seen:
                raise ValueError("duplicate cohort row")
            seen.add(value)
        output_audio.mkdir(parents=True, exist_ok=True)
        target = output_audio / path.name
        shutil.copy2(path, target)
        rows.append({**row, "audio_path": str(target.resolve()), "decoded_pcm_sha256": ph})
    write(destination / "references.json", rows)
    write(
        destination / "inference.json",
        [{key: row[key] for key in ("id", "audio_path", "wav_sha256")} for row in rows],
    )
    return rows


def prepare(root, workspace):
    if root.exists():
        raise FileExistsError("choose a new study directory; evidence is never overwritten")
    root.mkdir(parents=True)
    write(
        root / "state.json",
        {
            "created_utc": utc(),
            "deadline_epoch": time.time() + 21600,
            "max_trials": 3,
            "status": "preparing",
            "cloud_spend": 0,
            "dataset": "google/fleurs",
            "revision": REV,
            "license": "CC-BY-4.0",
            "primary_metric": "NFKC whitespace-free strict CER, punctuation retained",
            "additional_metric": "raw codepoint CER and literal exact match",
            "heldout_policy": "prospective development check, never publisher final test",
            "speaker_independence": "unknown",
            "max_new_download_bytes": 2000000000,
            "workspace": str(workspace),
            "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip(),
        },
    )
    campaign = workspace / "work/campaign"
    raw = campaign / "gpu-006-results/fresh-dev006"
    records = read(raw / "manifest.json")
    assert len(records) == 64
    import_cohort(records, campaign / "fresh-dev006-audio", root / "dev", root / "audio/dev")
    model_hashes = read(campaign / "cpu-fresh006/freeze.json")["model_hashes"]
    models = {
        "reazon_dir": str(
            workspace / "work/models/sherpa-onnx-zipformer-ja-en-reazonspeech-2025-01-17"
        ),
        "reazon_sha256": model_hashes["reazon"],
        "parakeet_dir": str(
            workspace / "work/models/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
        ),
        "parakeet_sha256": model_hashes["parakeet_ja_ctc"],
    }
    write(root / "models.json", models)
    ex = read(campaign / "batch006-inputs/exclusions.json")
    ids, refs, pcm = (
        set(map(str, ex["source_ids"])),
        set(ex["reference_sha256"]),
        set(ex["pcm_sha256"]),
    )
    for row in records:
        ids.add(row["id"])
        refs.add(row["reference_sha256"])
        pcm.add(row["source_pcm_sha256"])
    write(
        root / "exclusions.json",
        {
            "ids": sorted(ids),
            "refs": sorted(refs),
            "source_pcm": sorted(pcm),
            "dev_wav": [r["wav_sha256"] for r in records],
        },
    )
    cached = {}
    gpu_report = read(raw / "report.json")
    for name in ("whisper-turbo", "qwen-base", "projection005"):
        path = raw / (name + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [r["id"] for r in rows] == [r["id"] for r in records]
        errors = sum(
            measure(r["reference"], h["text"])["strict_errors"]
            for r, h in zip(records, rows, strict=True)
        )
        expected = gpu_report["scores"][name]["strict"]["errors"]
        cached[name] = {
            "rows": rows,
            "source_sha256": sha(path),
            "original_errors": expected,
            "recomputed_errors": errors,
            "valid": errors == expected,
            "timing_scope": "historical whole-batch incl loading; per-utterance unavailable",
            "historical_batch_seconds": gpu_report["models"][name]["seconds"],
        }
    write(root / "dev/cached.json", cached)
    write(
        root / "preregistration.json",
        {
            "created_utc": utc(),
            "trials": TRIALS,
            "selection": (
                "lowest strict CER improving single normalized baseline with zero new "
                "strict-exact regressions; ties use listed order; otherwise keep baseline"
            ),
            "holdout_count": 64,
            "holdout_selection": (
                "first eligible official-train rows in fixed parquet order, "
                "2..15seconds, exclusions by id/ref/PCM/WAV"
            ),
            "not_training_weights": True,
            "max_trials": 3,
            "max_wall_seconds": 21600,
        },
    )
    print("prepared development64 and three fixed trial definitions", flush=True)


def worker(root, cohort, variant):
    # Only inferential inputs are opened in this process. Reference loading and
    # candidate selection live in the parent process after outputs are fixed.
    from semantic_asr.run_cli import build_parser, build_profile_adapters, run_transcription

    check(root)
    rows = read(root / cohort / "inference.json")
    assert all(set(r) == {"id", "audio_path", "wav_sha256"} for r in rows)
    models = read(root / "models.json")
    target = root / cohort / variant
    target.mkdir(exist_ok=False)
    options = [
        "--profile",
        "reazon-ja-v1" if variant == "single" else "reazon-ja-research-v1",
        "--reazon-model-dir",
        models["reazon_dir"],
        "--reazon-artifact-sha256",
        models["reazon_sha256"],
        "--formats",
        "json,observed,normalized",
        "--quiet",
    ]
    if variant == "dual":
        options += [
            "--second-ear-parakeet-model-dir",
            models["parakeet_dir"],
            "--second-ear-parakeet-artifact-sha256",
            models["parakeet_sha256"],
        ]
    args = build_parser().parse_args([rows[0]["audio_path"], *options])
    start = time.monotonic()
    primary, secondary = build_profile_adapters(args)
    load_seconds = time.monotonic() - start
    with (target / "predictions.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            check(root)
            assert sha(row["audio_path"]) == row["wav_sha256"]
            args.audio, args.output_dir = row["audio_path"], str(target / row["id"])
            start = time.monotonic()
            summary = run_transcription(args, adapter=primary, second_ear=secondary)
            seconds = time.monotonic() - start
            result = read(summary["outputs"]["transcript_json"])
            evidence = read(summary["outputs"]["json"])
            # The public serialization uses an observedTranscript object per segment.
            prediction = {
                "id": row["id"],
                "observed": result["observedText"],
                "normalized": result["normalizedText"],
                "status": "provisional" if summary["provisionalSegments"] else "accepted",
                "seconds": seconds,
                "evidence_sha256": summary["evidenceSha256"],
                "evidence_file": summary["outputs"]["json"],
                "provenance": result["provenance"],
                "candidate_texts": extract_candidates(evidence),
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{cohort}/{variant} {row['id']} complete", flush=True)
    write(
        target / "completed.json",
        {
            "count": len(rows),
            "model_load_seconds": load_seconds,
            "predictions_sha256": sha(target / "predictions.jsonl"),
        },
    )


def extract_candidates(evidence):
    values = []

    def visit(obj):
        if isinstance(obj, dict):
            if "candidate_id" in obj and "text" in obj:
                values.append(obj["text"])
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    visit(evidence)
    return list(dict.fromkeys(values))


def run_workers(root, cohort):
    for variant in ("single", "dual"):
        check(root)
        complete = root / cohort / variant / "completed.json"
        if complete.exists():
            saved = read(complete)
            assert saved["predictions_sha256"] == sha(complete.parent / "predictions.jsonl")
            continue
        with (root / f"{cohort}-{variant}.log").open("x", encoding="utf-8") as log:
            seconds = max(1, min(3600, check(root)["deadline_epoch"] - time.time()))
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "worker",
                    "--output",
                    str(root),
                    "--cohort",
                    cohort,
                    "--variant",
                    variant,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds,
                check=True,
                env={**os.environ, "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1"},
            )


def measurements(root, cohort, include_trials):
    references = read(root / cohort / "references.json")
    results = []
    for variant in ("single", "dual"):
        rows = [
            json.loads(line)
            for line in (root / cohort / variant / "predictions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [r["id"] for r in rows] == [r["id"] for r in references]
        for ref, row in zip(references, rows, strict=True):
            for layer in ("observed", "normalized"):
                results.append(
                    {
                        "id": row["id"],
                        "system": variant + "-" + layer,
                        "text": row[layer],
                        "status": row["status"],
                        "seconds": row["seconds"],
                        "execution": "new-local-CLI-path",
                        "layer": layer,
                        "reference": ref["reference"],
                        "audio_path": ref["audio_path"],
                        "candidates": row["candidate_texts"],
                        "evidence_sha256": row["evidence_sha256"],
                        **measure(ref["reference"], row[layer]),
                    }
                )
            for name in include_trials:
                source, stop = TRIALS[name]
                if source != variant:
                    continue
                text = normalized_trial(row["normalized"], terminal_stop=stop)
                results.append(
                    {
                        "id": row["id"],
                        "system": name,
                        "text": text,
                        "status": row["status"],
                        "seconds": row["seconds"],
                        "execution": "derived-no-new-inference",
                        "layer": "normalized-only",
                        "reference": ref["reference"],
                        "audio_path": ref["audio_path"],
                        "candidates": row["candidate_texts"],
                        "parent_evidence_sha256": row["evidence_sha256"],
                        **measure(ref["reference"], text),
                    }
                )
    if cohort == "dev":
        for name, cached in read(root / "dev/cached.json").items():
            if not cached["valid"]:
                continue
            for ref, row in zip(references, cached["rows"], strict=True):
                results.append(
                    {
                        "id": row["id"],
                        "system": "cached-" + name,
                        "text": row["text"],
                        "status": "unknown",
                        "seconds": None,
                        "execution": "historical-GPU-recomputed",
                        "layer": "observed",
                        "reference": ref["reference"],
                        "audio_path": ref["audio_path"],
                        "candidates": [],
                        **measure(ref["reference"], row["text"]),
                    }
                )
    scores = {
        name: summarize([r for r in results if r["system"] == name])
        for name in dict.fromkeys(r["system"] for r in results)
    }
    write(root / cohort / "measurements.json", results)
    write(root / cohort / "scores.json", scores)
    return results, scores


def develop(root):
    run_workers(root, "dev")
    rows, scores = measurements(root, "dev", list(TRIALS))
    baseline = {r["id"]: r for r in rows if r["system"] == "single-normalized"}
    chosen, best = "single-normalized", scores["single-normalized"]["strict_errors"]
    ledger = []
    for name in TRIALS:
        selected = [r for r in rows if r["system"] == name]
        harm = sum(baseline[r["id"]]["strict_exact"] and not r["strict_exact"] for r in selected)
        score = scores[name]
        improves = score["strict_errors"] < best and harm == 0
        ledger.append(
            {
                "trial": name,
                "scores": score,
                "new_false_corrections": harm,
                "improves_current_best": improves,
                "weights_trained": False,
            }
        )
        if improves:
            chosen, best = name, score["strict_errors"]
    write(root / "trials.json", ledger)
    write(
        root / "selection-freeze.json",
        {
            "created_utc": utc(),
            "chosen": chosen,
            "trials": TRIALS,
            "models_sha256": sha(root / "models.json"),
            "implementation_sha256": sha(Path(__file__).resolve()),
            "normalizer_module_sha256": sha(
                Path(__file__).resolve().parents[1] / "src/semantic_asr/local_accuracy.py"
            ),
            "development_scores_sha256": sha(root / "dev/scores.json"),
            "holdout_seen": False,
            "promotion": False,
        },
    )
    print(json.dumps({"chosen": chosen, "scores": scores}, ensure_ascii=False), flush=True)


def fresh(root):
    import fsspec
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    freeze = read(root / "selection-freeze.json")
    assert freeze["implementation_sha256"] == sha(Path(__file__).resolve())
    assert freeze["models_sha256"] == sha(root / "models.json")
    destination = root / "fresh"
    destination.mkdir(exist_ok=False)
    ex = read(root / "exclusions.json")
    ids, refs, pcms = set(ex["ids"]), set(ex["refs"]), set(ex["source_pcm"])
    wavs = set(ex["dev_wav"])
    readme_url = f"https://huggingface.co/datasets/google/fleurs/raw/{REV}/README.md"
    with urllib.request.urlopen(readme_url, timeout=60) as response:
        card = response.read().decode()
    if "cc-by-4.0" not in card.lower():
        raise ValueError("dataset license no longer matches preregistration")
    (root / "FLEURS-DATASET-CARD.md").write_text(card, encoding="utf-8")
    url = f"https://huggingface.co/datasets/google/fleurs/resolve/{REV}/parquet-data/ja_jp/train-00000-of-00001.parquet"
    records = []
    # HTTP ranges keep the full 1.69GB archive off disk. Only selected WAVs persist.
    with fsspec.open(url, mode="rb", block_size=8 * 1024 * 1024, cache_type="readahead") as remote:
        parquet = pq.ParquetFile(remote)
        for batch in parquet.iter_batches(batch_size=8):
            check(root)
            for row in batch.to_pylist():
                identifier = str(row["id"])
                ref = row.get("raw_transcription") or row["transcription"]
                rh = hashlib.sha256(ref.encode()).hexdigest()
                if identifier in ids or rh in refs or not 32000 <= row["num_samples"] <= 240000:
                    continue
                a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                if sr != 16000 or a.ndim != 1 or not np.isfinite(a).all():
                    raise ValueError("invalid fresh PCM")
                ph = hashlib.sha256(a.astype("<f4").tobytes()).hexdigest()
                if ph in pcms:
                    continue
                buffer = io.BytesIO()
                sf.write(buffer, a, sr, format="WAV", subtype="PCM_16")
                data = buffer.getvalue()
                wh = hashlib.sha256(data).hexdigest()
                if wh in wavs:
                    continue
                target = root / "audio/fresh" / (identifier + ".wav")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                records.append(
                    {
                        "id": identifier,
                        "reference": ref,
                        "reference_sha256": rh,
                        "source_pcm_sha256": ph,
                        "wav_sha256": wh,
                        "audio_path": str(target.resolve()),
                        "seconds": len(a) / sr,
                        "source_split": "train",
                        "role": "prospective-development-check",
                        "license": "CC-BY-4.0",
                    }
                )
                ids.add(identifier)
                refs.add(rh)
                pcms.add(ph)
                wavs.add(wh)
                if len(records) == 64:
                    break
            if len(records) == 64:
                break
    if len(records) != 64:
        write(destination / "selection-failure.json", {"selected": len(records)})
        raise ValueError("64 unused rows unavailable with fixed exclusions")
    write(destination / "references.json", records)
    write(
        destination / "inference.json",
        [{k: r[k] for k in ("id", "audio_path", "wav_sha256")} for r in records],
    )
    write(
        destination / "freeze.json",
        {
            "created_utc": utc(),
            "selection_freeze_sha256": sha(root / "selection-freeze.json"),
            "manifest_sha256": sha(destination / "references.json"),
            "count": 64,
            "speaker_disjoint": "unknown",
            "final_test": False,
        },
    )
    run_workers(root, "fresh")
    trials = [] if freeze["chosen"] == "single-normalized" else [freeze["chosen"]]
    measurements(root, "fresh", trials)
    print("fresh64 measured once with fixed chosen variant", flush=True)


def render(root):
    cohorts = {}
    all_rows = []
    for cohort in ("dev", "fresh", "whisper-dev32"):
        if not (root / cohort / "scores.json").exists():
            continue
        rows = read(root / cohort / "measurements.json")
        cohorts[cohort] = {
            name: summarize([r for r in rows if r["system"] == name])
            for name in dict.fromkeys(r["system"] for r in rows)
        }
        for row in rows:
            row["cohort"] = cohort
            row["audio"] = Path(row["audio_path"]).relative_to(root).as_posix()
            if row["raw_errors"] == 0:
                category = "一致"
            elif row["strict_errors"] == 0 or row["lenient_errors"] == 0:
                category = "表記差（採点対象）"
            elif any(
                normalize_characters(c) == normalize_characters(row["reference"])
                for c in row["candidates"]
            ):
                category = "候補選択"
            else:
                category = "候補不足／原因不明（聴取未確認）"
            row["category"] = category
            row["surface_review"] = surface_review(row["reference"], row["text"])
            row["user_note"] = (
                "過去の内容評価では英字・カナの表記差について指摘あり。"
                "今回の厳密CERでは元の採点を維持。発話全体は聴取未確認。"
                if row["id"] == "119" and cohort in {"dev", "whisper-dev32"}
                else ""
            )
            all_rows.append(row)
    summaries = []
    for cohort, scores in cohorts.items():
        for system, score in scores.items():
            accepted_cer = (
                f"{100 * score['accepted_raw_cer']:.3f}%"
                if score["accepted_raw_cer"] is not None
                else "対象なし"
            )
            accepted_exact = (
                f"{100 * score['accepted_raw_exact_rate']:.2f}%"
                if score["accepted_raw_exact_rate"] is not None
                else "対象なし"
            )
            summaries.append(
                f"<tr><td>{cohort}</td><td>{html.escape(system)}</td>"
                f"<td>{100 * score['primary_cer']:.3f}%</td>"
                f"<td>{score['raw_exact_count']}/{score['count']}"
                f" ({100 * score['raw_exact_rate']:.2f}%)</td>"
                f"<td>{score['provisional_count']}/{score['count']}"
                f" ({100 * score['provisional_rate']:.2f}%)</td>"
                f"<td>{score['unknown_status_count']}</td>"
                f"<td>{score['accepted_count']}</td><td>{accepted_cer}</td>"
                f"<td>{accepted_exact}</td><td>{100 * score['strict_cer']:.3f}%</td></tr>"
            )
    data = json.dumps(all_rows, ensure_ascii=False).replace("<", "\\u003c")
    page = """<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<title>Semantic ASR 音声・正解・出力の比較</title><style>
body{font:16px system-ui;margin:24px auto;max-width:1250px;padding:0 20px;
background:#f5f7fa;color:#182530}
table{border-collapse:collapse;background:white;width:100%;font-size:14px}
td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
article{background:white;border:1px solid #d4dce4;border-radius:10px;margin:15px 0;padding:18px}
p{line-height:1.7;overflow-wrap:anywhere}label{display:inline-block;margin:10px}
select,input,button{padding:8px;font:inherit}audio{width:100%}mark{background:#ffe18b}
</style><h1>音声・正解文・実出力を比較する</h1>
<p><strong>主指標：厳密CER（無加工の文字列）。句読点・数字表記・漢字かな・空白の差も数えます。</strong>
発話完全一致率は別の指標です。固定セットでCER 0かつ完全一致率100%のときだけ目標達成とします。
100%−CERを発話正解率や意味精度とは呼びません。今回は目標未達です。</p>
<p>旧指標のNFKC・空白除去CERは補助列です。当時の選択・固定記録は変更していません。
未確定発話も全て分母に含みます。cachedは過去GPU結果、single/dualは今回の本体CLI経路、
trialは正規化側だけの変換です。音を聴かずに音響誤りとは断定していません。</p>
<p>dev：既露出64音声。fresh：設定固定後に一度評価した追加64音声。現在は既露出です。
話者独立は未確認で、公式final testではありません。FLEURS / Google, CC BY 4.0。
出典：<a href="https://huggingface.co/datasets/google/fleurs">dataset card</a>。
モデル・出力来歴はraw evidenceとJSONを参照してください。</p>
<table><thead><tr><th>データ</th><th>経路</th><th>厳密CER</th><th>発話完全一致</th>
<th>未確定</th><th>状態不明</th><th>確定数</th><th>確定分のみCER</th>
<th>確定分のみ完全一致率</th><th>旧CER（補助）</th>
</tr></thead><tbody>SUMMARY</tbody></table>
<label>データ<select id="cohort"><option value="">全て</option>
<option>dev</option><option>fresh</option><option>whisper-dev32</option></select></label>
<label>経路<select id="system"></select></label>
<label><input id="errors" type="checkbox" checked>文字差がある例のみ</label>
<label>検索<input id="search" placeholder="音声ID・単語"></label><p id="count"></p>
<main id="rows"></main><button id="more">さらに20件</button>
<button id="export">手動確認結果をJSON保存</button>
<script>const data=DATA;let limit=20;const el=id=>document.getElementById(id);
const reviewKey='semantic-asr-local-review-v1';
let reviews={};try{reviews=JSON.parse(localStorage.getItem(reviewKey)||'{}')}catch(e){}
for(const s of [...new Set(data.map(r=>r.system))]){
let o=document.createElement('option');o.value=s;o.textContent=s;el('system').append(o)}
function text(parent,tag,value){let n=document.createElement(tag);
n.textContent=value;parent.append(n);return n}
function draw(){const q=el('search').value;
const rows=data.filter(r=>(!el('cohort').value||r.cohort===el('cohort').value)
&&r.system===el('system').value&&(!el('errors').checked||r.raw_errors>0)
&&(!q||(r.id+' '+r.reference+' '+r.text).includes(q)));el('rows').replaceChildren();
el('count').textContent=rows.length+'件 / 表示'+Math.min(rows.length,limit)+'件';
for(const r of rows.slice(0,limit)){let a=document.createElement('article');
text(a,'h2',r.cohort+' / '+r.id+' / '+r.system);let audio=document.createElement('audio');
audio.controls=true;audio.preload='none';audio.src=r.audio;a.append(audio);
text(a,'p','参照表記：'+r.reference);text(a,'p','出力：'+r.text);
let diff=document.createElement('p');diff.innerHTML=r.diff;a.append(diff);
text(a,'p','文字誤り '+r.raw_errors+' / '+r.raw_units+'文字 ・ '+r.status+' ・ '
+r.category+' ・ '+(r.seconds==null?'個別時間なし':r.seconds.toFixed(3)+'秒'));
text(a,'p','実行区分：'+(r.execution||'記録なし')+' ・ 出力層：'+(r.layer||'記録なし'));
if(r.user_note)text(a,'p',r.user_note);
const key=r.cohort+'/'+r.id+'/'+r.system;let select=document.createElement('select');
for(const label of ['未確認','表記差のみ・同じ内容','内容の聞き違い','参照文の疑い','判断困難']){
let option=document.createElement('option');option.textContent=label;select.append(option)}
select.value=reviews[key]?.judgment||'未確認';select.onchange=()=>{
reviews[key]={judgment:select.value,reference:r.reference,text:r.text};
localStorage.setItem(reviewKey,JSON.stringify(reviews))};a.append(select);el('rows').append(a)}}
for(const id of ['cohort','system','errors','search'])
el(id).addEventListener('input',()=>{limit=20;draw()});
el('more').onclick=()=>{limit+=20;draw()};el('export').onclick=()=>{
const url=URL.createObjectURL(new Blob([JSON.stringify(reviews,null,2)],{type:'application/json'}));
let a=document.createElement('a');a.href=url;a.download='manual-review.json';a.click();
setTimeout(()=>URL.revokeObjectURL(url),1000)};draw();</script></html>"""
    # Diff HTML is generated from escaped user/data text only.
    for row in all_rows:
        parts = []
        for op, i, j, k, end in difflib.SequenceMatcher(
            None, row["reference"], row["text"], autojunk=False
        ).get_opcodes():
            a, b = html.escape(row["reference"][i:j]), html.escape(row["text"][k:end])
            parts.append(b if op == "equal" else f"<mark><del>{a}</del> → <ins>{b}</ins></mark>")
        row["diff"] = "".join(parts)
    data = json.dumps(all_rows, ensure_ascii=False).replace("<", "\\u003c")
    (root / "index.html").write_text(
        page.replace("SUMMARY", "".join(summaries)).replace("DATA", data), encoding="utf-8"
    )
    write(root / "summary.json", cohorts)
    write(root / "remaining-errors.json", [r for r in all_rows if r["raw_errors"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "develop", "fresh", "report", "worker"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cohort", choices=("dev", "fresh"))
    parser.add_argument("--variant", choices=("single", "dual"))
    args = parser.parse_args()
    root = args.output.resolve()
    if args.stage == "worker":
        worker(root, args.cohort, args.variant)
        return
    with _checkpoint_writer_lock(root.parent / (root.name + "-writer")):
        try:
            if args.stage == "prepare":
                prepare(root, args.workspace.resolve())
            elif args.stage == "develop":
                develop(root)
                render(root)
            elif args.stage == "fresh":
                fresh(root)
                render(root)
            else:
                render(root)
        except Exception as exc:
            if root.exists():
                write(
                    root / f"failure-{args.stage}-{time.time_ns()}.json",
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "utc": utc(),
                    },
                )
            raise


if __name__ == "__main__":
    main()
