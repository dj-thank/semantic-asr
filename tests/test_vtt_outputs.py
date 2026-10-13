from __future__ import annotations

import json
from html import unescape
from pathlib import Path

import pytest

from semantic_asr.contracts import CandidateEvidence, NormalizedTranscript, ObservedTranscript
from semantic_asr.fusion import fuse_candidates
from semantic_asr.longform import LongformResult, LongformSegment, Window
from semantic_asr.outputs import render_vtt, write_outputs


def result_with_text(text: str) -> LongformResult:
    audio_digest = "a" * 64
    ranked = fuse_candidates([CandidateEvidence("spoken", text)])
    observed = ObservedTranscript.create(
        selected=ranked[0],
        ranked=ranked,
        uncertainty_spans=[],
        source_audio_sha256=audio_digest,
    )
    normalized = NormalizedTranscript.attach(observed, text=f"字幕: {text}", mode="deterministic")
    return LongformResult.create(
        source_name="meeting.wav",
        source_audio_sha256=audio_digest,
        duration_ms=1000,
        segments=(LongformSegment(Window(0, 0, 1000), observed, normalized, {}),),
        diagnostics={"provisionalWindowCount": 0},
    )


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "text,payload",
    [
        ("<b>音声</b><v 太郎>です", "&lt;b&gt;音声&lt;/b&gt;&lt;v 太郎&gt;です"),
        ("&amp; &lt; &gt; & '引用'", "&amp;amp; &amp;lt; &amp;gt; &amp; '引用'"),
        ("入力 --> 出力", "入力 --&gt; 出力"),
        ("はい。はい。\n続きです。", "はい。はい。\n続きです。"),
    ],
)
def test_vtt_preserves_literal_transcript_text(text, payload, normalized):
    result = result_with_text(text)
    before = result.as_dict()
    prefix = "字幕: " if normalized else ""

    document = render_vtt(result, normalized=normalized)

    assert document == f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{prefix}{payload}\n"
    # Cue text entities decode once: literal entity spellings must survive too.
    cue_text = document.split("\n", 3)[3].rstrip("\n")
    assert unescape(cue_text) == prefix + text
    assert result.as_dict() == before
    result.verify()


def test_output_bundle_escapes_only_vtt_serialization(tmp_path):
    result = result_with_text("<b>音声</b> &amp;")
    before = result.as_dict()

    paths = write_outputs(result, tmp_path, formats={"vtt", "json", "observed", "normalized"})

    assert Path(paths["vtt"]).read_text(encoding="utf-8") == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n字幕: &lt;b&gt;音声&lt;/b&gt; &amp;amp;\n"
    )
    assert Path(paths["observed"]).read_text(encoding="utf-8") == result.observed_text + "\n"
    assert Path(paths["normalized"]).read_text(encoding="utf-8") == result.normalized_text + "\n"
    exported = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert exported["observed_text"] == result.observed_text
    assert exported["normalized_text"] == result.normalized_text
    assert exported["evidence_sha256"] == result.evidence_sha256
    assert result.as_dict() == before
    result.verify()
