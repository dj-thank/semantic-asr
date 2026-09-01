from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .research_registry import default_research_registry

RESEARCH_COMMANDS = {"research-ledger"}


def research_ledger_payload() -> dict[str, Any]:
    registry = default_research_registry()
    return {
        "schemaVersion": "semantic-asr-research-ledger-v1",
        "version": registry.version,
        "digest": registry.digest,
        "sources": [
            {
                **asdict(source),
                "status": source.status.value,
            }
            for source in registry.sources
        ],
        "translations": [
            {
                **asdict(translation),
                "kind": translation.kind.value,
            }
            for translation in registry.translations
        ],
        "metadata": registry.metadata,
    }


def research_ledger_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Semantic ASR research ledger",
        "",
        f"- Version: `{payload['version']}`",
        f"- Digest: `{payload['digest']}`",
        "",
        "## Sources",
        "",
    ]
    for source in payload["sources"]:
        location = f" — {source['primary_url']}" if source.get("primary_url") else ""
        lines.append(
            f"- **{source['source_id']}** [{source['status']}]: "
            f"{source['title']}{location}"
        )
    lines.extend(("", "## Architecture translations", ""))
    for translation in payload["translations"]:
        lines.extend(
            (
                f"### {translation['translation_id']}",
                "",
                f"- Kind: `{translation['kind']}`",
                f"- Sources: {', '.join(translation['source_ids'])}",
                f"- Semantic ASR mechanism: {translation['semantic_asr_mechanism']}",
                f"- Claim boundary: {translation['claim_boundary']}",
                f"- Falsification test: {translation['falsification_test']}",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def command_research_ledger(args: argparse.Namespace) -> int:
    payload = research_ledger_payload()
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.format == "json"
        else research_ledger_markdown(payload)
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-asr",
        description="Semantic ASR research provenance commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ledger = commands.add_parser("research-ledger")
    ledger.add_argument("--format", choices=["json", "markdown"], default="json")
    ledger.add_argument("--output")
    ledger.set_defaults(func=command_research_ledger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
