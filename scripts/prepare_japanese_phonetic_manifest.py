#!/usr/bin/env python3
"""Materialize rights-approved Japanese reading labels for dual CTC training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_asr.phonetic_runtime.japanese_labels import JapanesePhoneticLabelProfile
from semantic_asr.phonetic_runtime.materialize import materialize_japanese_phonetic_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--allow-derived-phonetic-labels",
        action="store_true",
        help="Acknowledge creation of source-text-derived phone/mora labels.",
    )
    args = parser.parse_args()
    if not args.allow_derived_phonetic_labels:
        parser.error("--allow-derived-phonetic-labels is required")

    profile = JapanesePhoneticLabelProfile()
    result = materialize_japanese_phonetic_manifest(
        args.input,
        args.output_dir,
        name=args.name,
        revision=args.revision,
        profile=profile,
    )
    print(
        json.dumps(
            {
                "outputDirectory": str(result.output_directory),
                "manifest": str(result.manifest_path),
                "manifestDigest": result.manifest.digest,
                "materializationDigest": result.digest,
                "labelProfileDigest": result.label_profile_digest,
                "phoneInventoryDigest": result.phone_inventory_digest,
                "moraInventoryDigest": result.mora_inventory_digest,
                "rows": len(result.manifest.rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
