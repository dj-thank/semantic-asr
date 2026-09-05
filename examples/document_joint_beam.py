"""Opt-in joint document deliberation example.

The scorer below is deliberately neutral and only demonstrates the integration contract. Replace
it with an immutable, held-out-evaluated complete-document scorer before making quality decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_asr import (
    CallableGlobalSequenceScorer,
    DocumentBeamConfig,
    DocumentContext,
    frozen_profile_digest,
    load_transcriber,
    with_document_deliberation,
)
from semantic_asr.outputs import write_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--profile", default="cpu-ja-v1")
    parser.add_argument("--output-dir", default="runs/document-joint-beam")
    args = parser.parse_args()

    scorer = CallableGlobalSequenceScorer(
        lambda path, context: 0.0,
        source="example-neutral-document-scorer",
        profile_digest=frozen_profile_digest(
            "example-neutral-document-scorer",
            "r1",
            {"warning": "integration-only-not-evaluated"},
        ),
    )
    transcriber = with_document_deliberation(
        load_transcriber(args.profile),
        sequence_scorer=scorer,
        config=DocumentBeamConfig(
            apply_provisional=False,
            fail_closed_to_first_pass=True,
        ),
        declared_context=DocumentContext(
            topic_summary="caller-declared context frozen before evaluation",
        ),
    )
    result = transcriber.transcribe(Path(args.audio))
    outputs = write_outputs(result, Path(args.output_dir))
    print(result.observed_text)
    print(result.diagnostics.get("globalDeliberation"))
    print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
