"""Opt-in joint document decoding example.

The demonstration scorer is intentionally neutral. Replace it with an immutable, held-out-evaluated
complete-document scorer before treating its preferences as decision evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_asr.api import load_transcriber
from semantic_asr.document_joint_deliberation import (
    DocumentDeliberationConfig,
    with_joint_document_deliberation,
)
from semantic_asr.global_scorer import CallableGlobalSequenceScorer, frozen_profile_digest
from semantic_asr.outputs import write_outputs


def neutral_document_scorer(path, context) -> float:
    del path, context
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--profile", default="cpu-ja-v1")
    parser.add_argument("--output-dir", default="runs/document-joint-deliberation")
    args = parser.parse_args()

    first_pass = load_transcriber(args.profile)
    scorer = CallableGlobalSequenceScorer(
        neutral_document_scorer,
        source="example-neutral-document-scorer",
        profile_digest=frozen_profile_digest(
            "example-neutral-document-scorer",
            "r1",
            {"warning": "demonstration-only-not-evaluated"},
        ),
    )
    transcriber = with_joint_document_deliberation(
        first_pass,
        document_scorer=scorer,
        config=DocumentDeliberationConfig(
            apply_provisional=False,
            fail_closed_to_first_pass=True,
        ),
    )
    result = transcriber.transcribe(Path(args.audio))
    outputs = write_outputs(result, Path(args.output_dir))
    print(result.observed_text)
    print(result.diagnostics.get("documentJointDeliberation"))
    print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
