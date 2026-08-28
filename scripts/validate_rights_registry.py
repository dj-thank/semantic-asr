#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry")
    parser.add_argument("--schema", default="schemas/rights-registry.schema.json")
    args = parser.parse_args()
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("install jsonschema to validate the registry") from exc
    payload = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)
    identifiers = [row["assetId"] for row in payload["assets"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("rights asset IDs must be unique")
    print(f"validated {len(identifiers)} rights assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
