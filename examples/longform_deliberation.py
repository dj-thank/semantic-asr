"""Opt-in long-form deliberation example.

The scorer below is a deterministic integration demonstration, not a quality model. Replace it with
an immutable, held-out-evaluated complete-path scorer before using the second pass for decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from semantic_asr import (
    CallableGlobalSequenceScorer,
    DocumentContext,
    LongformDeliberationConfig,
    frozen_profile_digest,
    load_transcriber,
    with_global_deliberation,
)
from semantic_asr.outputs import write_outputs


def contextual_preference(path, context: DocumentContext) -> float:
    text = "".join(arc.text for arc in path)
    # Demonstrate that the scorer receives the complete path and frozen right context.
    if "まだ" in text and any(marker in context.right_context for marker in ("承認後", "完了後")):
        return 0.5
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--profile", default="cpu-ja-v1")
    parser.add_argument("--output-dir", default="runs/longform-deliberation")
    args = parser.parse_args()

    first_pass = load_transcriber(args.profile)
    scorer = CallableGlobalSequenceScorer(
        contextual_preference,
        source="example-contextual-preference",
        profile_digest=frozen_profile_digest(
            "example-contextual-preference",
            "r1",
            {"warning": "demonstration-only-not-evaluated"},
        ),
    )
    transcriber = with_global_deliberation(
        first_pass,
        sequence_scorer=scorer,
        config=LongformDeliberationConfig(
            apply_provisional=False,
            fail_closed_to_first_pass=True,
        ),
        declared_context=DocumentContext(
            topic_summary="caller-declared meeting topic",
        ),
    )
    result = transcriber.transcribe(Path(args.audio))
    outputs = write_outputs(result, Path(args.output_dir))
    print(result.observed_text)
    print(result.diagnostics["globalDeliberation"])
    print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
