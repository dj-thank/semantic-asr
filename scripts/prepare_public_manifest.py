"""Build a rights-annotated local manifest from a public Hugging Face ASR dataset.

Materialising audio and a manifest containing references and absolute paths is a
local-research operation.  It is therefore disabled unless ``--allow-raw-export``
is supplied, and the output directory must resolve outside this checkout.  Rights
default to ``review``; the one dataset/revision explicitly supported by this
repository is allowed only when no stricter registry decision is supplied.

The public test sets used here carry no speaker labels, so ``groupId`` falls back
to the sample identifier and the resulting split is *not* speaker-disjoint.  Record
that limitation with any quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any

try:  # Optional: installed by the ``public-data`` extra.
    import numpy as np
except ImportError:  # pragma: no cover - exercised in an environment without extras
    np = None

try:  # Optional: installed by the ``public-data`` extra.
    import soundfile as sf
except ImportError:  # pragma: no cover - exercised in an environment without extras
    sf = None

try:  # Optional: installed by the ``public-data`` extra.
    from datasets import Audio, load_dataset
except ImportError:  # pragma: no cover - exercised in an environment without extras
    Audio = None
    load_dataset = None

try:  # Optional: installed by the ``public-data`` extra.
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - exercised in an environment without extras
    resample_poly = None

from semantic_asr.revisions import PUBLIC_DATASET_REVISIONS, resolve_hugging_face_revision
from semantic_asr.rights import RightsRegistry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DATASETS: dict[str, dict[str, str]] = {
    "reazonspeech-test": {
        "path": "japanese-asr/ja_asr.reazonspeech_test",
        "domain": "broadcast",
        "license": "reazonspeech-apache-2.0",
        "rightsAssetId": "reazonspeech-release",
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


def exact_supported_public_asset(dataset: str, dataset_revision: str) -> bool:
    """Return whether ``dataset`` is a repository-supported exact public snapshot."""

    spec = DATASETS.get(dataset)
    if spec is None:
        return False
    return PUBLIC_DATASET_REVISIONS.get(spec["path"]) == dataset_revision.lower()


def resolve_rights_decision(
    dataset: str,
    dataset_revision: str,
    requested: str | None = None,
) -> str:
    """Resolve the CLI decision, keeping unknown assets fail-closed.

    An explicit ``allow`` is an operator decision.  Without one, only the exact
    pinned public snapshot known by this repository may use ``allow``; every other
    dataset/revision remains ``review``.
    """

    if requested is not None:
        return requested
    return "allow" if exact_supported_public_asset(dataset, dataset_revision) else "review"


def ensure_safe_output_dir(output_dir: str | Path) -> Path:
    """Resolve a raw-data destination and reject paths inside this checkout."""

    resolved = Path(output_dir).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("raw public-data output must not be a filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(resolved)
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError(
        "raw public-data output must be outside the repository checkout; "
        "use an external local-research directory"
    )


def validate_rights_for_export(
    dataset: str,
    dataset_revision: str,
    *,
    requested: str | None = None,
    registry_path: str | Path | None = None,
    asset_id: str | None = None,
) -> tuple[str, str | None]:
    """Resolve and enforce rights for an export containing audio and references."""

    decision = resolve_rights_decision(dataset, dataset_revision, requested)
    if decision not in {"allow", "deny", "review"}:
        raise ValueError(f"unknown rights decision: {decision}")
    spec = DATASETS[dataset]
    selected_asset_id = asset_id or spec.get("rightsAssetId")
    if asset_id is not None and registry_path is None:
        raise ValueError("--rights-asset-id requires --rights-registry")
    if registry_path is not None:
        if not selected_asset_id:
            raise ValueError("--rights-registry requires a rights asset ID")
        registry = RightsRegistry.load(registry_path)
        # Materialising a WAV and a reference-bearing manifest is both a derived
        # feature operation and a raw-data operation.  A registry review/deny
        # must never be overridden by the CLI's explicit ``allow``.
        registry.require(selected_asset_id, "derive_features")
        registry.require(selected_asset_id, "redistribute_raw")
    if decision != "allow":
        raise PermissionError(
            f"raw public-data export requires allow rights; dataset {dataset!r} is {decision!r}"
        )
    return decision, selected_asset_id


def _require_public_data_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    missing = [
        name
        for name, value in (
            ("numpy", np),
            ("soundfile", sf),
            ("datasets", load_dataset),
            ("scipy", resample_poly),
        )
        if value is None
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            f"public-data preparation requires {missing_text}; install with "
            "python -m pip install -e '.[public-data]'"
        )
    return np, sf, Audio, load_dataset, resample_poly


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = whole test split")
    parser.add_argument("--seed", default="semantic-asr-public-v1")
    parser.add_argument("--dataset-revision")
    parser.add_argument(
        "--rights-decision",
        choices=["allow", "review", "deny"],
        default=None,
        help=(
            "default: allow only for the exact pinned repository-supported asset, otherwise review"
        ),
    )
    parser.add_argument("--rights-registry", help="optional JSON rights registry")
    parser.add_argument(
        "--rights-asset-id",
        help="asset ID in --rights-registry (defaults to the dataset mapping when available)",
    )
    parser.add_argument(
        "--allow-raw-export",
        "--allow-raw",
        "--export-raw",
        action="store_true",
        help="explicitly materialise WAV/reference/path outputs in an external directory",
    )
    args = parser.parse_args()

    if not args.allow_raw_export:
        parser.error(
            "raw WAV/reference/path output is disabled by default; "
            "pass --allow-raw-export and use an output directory outside the checkout"
        )
    out = ensure_safe_output_dir(args.output_dir)
    spec = DATASETS[args.dataset]
    dataset_revision = resolve_hugging_face_revision(
        spec["path"],
        args.dataset_revision,
        PUBLIC_DATASET_REVISIONS,
    )
    rights_decision, rights_asset_id = validate_rights_for_export(
        args.dataset,
        dataset_revision,
        requested=args.rights_decision,
        registry_path=args.rights_registry,
        asset_id=args.rights_asset_id,
    )
    numpy_module, soundfile_module, audio_type, load_dataset_fn, resample_poly_fn = (
        _require_public_data_dependencies()
    )

    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_fn(spec["path"], revision=dataset_revision, split="test")
    dataset = dataset.cast_column("audio", audio_type(decode=False))
    if args.limit:
        dataset = dataset.shuffle(seed=20260902).select(range(min(args.limit, len(dataset))))

    manifest_path = out / "manifest.jsonl"
    counts = {"train": 0, "calibration": 0, "test": 0}
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            audio = row["audio"]
            array, rate = soundfile_module.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            if rate != 16000:
                array = resample_poly_fn(array, 16000, rate).astype(numpy_module.float32)
            sample_id = f"{args.dataset}-{index:06d}"
            wav_path = wav_dir / f"{sample_id}.wav"
            soundfile_module.write(wav_path, array, 16000, subtype="PCM_16")
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
                "rightsDecision": rights_decision,
                "rightsAssetId": rights_asset_id,
                "licenseId": spec["license"],
                "durationSeconds": round(len(array) / 16000, 3),
                "datasetName": spec["path"],
                "datasetRevision": dataset_revision,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "rows": len(dataset),
                "splits": counts,
                "dataset": spec["path"],
                "datasetRevision": dataset_revision,
                "rightsDecision": rights_decision,
                "rightsAssetId": rights_asset_id,
                "rawExport": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
