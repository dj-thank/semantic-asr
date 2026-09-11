from __future__ import annotations

import semantic_asr


def test_japanese_phonetic_export_surface_is_public() -> None:
    expected = {
        "JapanesePronunciationPolicy",
        "JapanesePronunciationTarget",
        "LoadedSourceRecording",
        "PhoneticFeatureExportConfig",
        "PhoneticFeatureExportResult",
        "PhoneticFeatureExporter",
        "PhoneticFeatureReceipt",
        "PhoneticSourceItem",
        "PhoneticSourceManifest",
        "PhoneticSourceResourcePolicy",
        "SoundFileSourceAudioLoader",
        "japanese_pronunciation_target",
        "load_phonetic_source_manifest",
    }

    assert expected.issubset(set(semantic_asr.__all__))
    assert all(hasattr(semantic_asr, name) for name in expected)
