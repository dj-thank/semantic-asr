import importlib.util
from pathlib import Path


def load_study():
    path = Path(__file__).resolve().parents[1] / "scripts/local_accuracy_study.py"
    spec = importlib.util.spec_from_file_location("local_accuracy_study_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_escapes_untrusted_transcription(tmp_path):
    from semantic_asr.local_accuracy import measure, summarize

    study = load_study()
    text = '</script><img src=x onerror="alert(1)">'
    row = {
        "id": "1",
        "system": "single-observed",
        "reference": text,
        "text": text,
        "status": "provisional",
        "seconds": 1,
        "candidates": [],
        "audio_path": str(tmp_path / "audio/dev/1.wav"),
        **measure(text, text),
    }
    study.write(tmp_path / "dev/scores.json", {"single-observed": summarize([row])})
    study.write(tmp_path / "dev/measurements.json", [row])
    study.render(tmp_path)
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert text not in page
    assert "\\u003c/script>" in page
    assert "audio/dev/1.wav" in page
    assert "provisional" in page


def test_candidate_extraction_preserves_strings_without_interpreting_them():
    study = load_study()
    assert study.extract_candidates(
        {
            "segments": [
                {
                    "observed": {
                        "candidates": [
                            {"candidate_id": "a", "text": "話しました"},
                            {"candidate_id": "b", "text": "話しませんでした"},
                        ]
                    }
                }
            ]
        }
    ) == ["話しました", "話しませんでした"]


def test_report_counts_literal_space_difference_and_accepted_subset(tmp_path):
    from semantic_asr.local_accuracy import measure, summarize

    study = load_study()
    rows = [
        {
            "id": "1",
            "system": "single-observed",
            "reference": "あ 。",
            "text": "あ。",
            "status": "accepted",
            "seconds": 1,
            "candidates": [],
            "audio_path": str(tmp_path / "audio/dev/1.wav"),
            **measure("あ 。", "あ。"),
        },
        {
            "id": "2",
            "system": "single-observed",
            "reference": "う。",
            "text": "う。",
            "status": "provisional",
            "seconds": 1,
            "candidates": [],
            "audio_path": str(tmp_path / "audio/dev/2.wav"),
            **measure("う。", "う。"),
        },
    ]
    study.write(tmp_path / "dev/scores.json", {"single-observed": summarize(rows)})
    study.write(tmp_path / "dev/measurements.json", rows)
    study.render(tmp_path)
    score = study.read(tmp_path / "summary.json")["dev"]["single-observed"]
    assert score["primary_cer"] == 0.2
    assert score["accepted_raw_cer"] == 1 / 3
    assert score["provisional_rate"] == 0.5
    assert score["accepted_raw_exact_rate"] == 0
    assert not score["all_exact"]
    assert [r["id"] for r in study.read(tmp_path / "remaining-errors.json")] == ["1"]
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "主指標：厳密CER" in page
    assert "確定分のみCER" in page
