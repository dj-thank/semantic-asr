from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_phonetic_features.py"
    spec = importlib.util.spec_from_file_location("export_phonetic_features_script", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def config():
    return {
        "schemaVersion": "1",
        "encoder": {
            "modelId": "test/frozen-encoder",
            "modelRevision": "1" * 40,
            "modelArtifactSha256": None,
            "revisionPolicy": "exact-commit",
            "layerIndex": 9,
            "sampleRate": 16000,
            "featureDimension": 768,
            "frameStrideMs": 20.0,
            "device": "cpu",
            "localFilesOnly": True,
        },
        "pronunciationPolicy": {
            "schemaVersion": "1",
            "blankSymbol": "<blk>",
            "nasalSymbol": "N",
            "sokuonSymbol": "q",
            "longVowelSymbol": ":",
            "ignorePunctuation": True,
            "mappingRevision": "ja-kana-mora-phone-v1",
        },
        "export": {
            "schemaVersion": "1",
            "featureDtype": "float32",
            "featureSubdirectory": "features",
            "maximumCachedRecordings": 2,
            "fsyncEachRow": True,
        },
        "resources": {
            "maximumItems": 100,
            "maximumReadingCharacters": 1000,
            "maximumSegmentDurationMs": 30000,
            "maximumTotalAudioSamples": 10000000,
            "maximumRecordingSamples": 10000000,
        },
    }


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_exact_config_loads_without_coercing_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write(path, config())

    feature, pronunciation, export, resources, runtime = module().load_config(path)

    assert feature.model_revision == "1" * 40
    assert feature.layer_index == 9
    assert pronunciation.mapping_revision == "ja-kana-mora-phone-v1"
    assert export.maximum_cached_recordings == 2
    assert resources.maximum_items == 100
    assert runtime == {"device": "cpu", "local_files_only": True}


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    value = config()
    value["encoder"]["unknown"] = True
    path = tmp_path / "config.json"
    write(path, value)

    with pytest.raises(ValueError, match="schema is not exact"):
        module().load_config(path)


def test_string_boolean_and_integer_are_rejected(tmp_path: Path) -> None:
    value = config()
    value["encoder"]["localFilesOnly"] = "false"
    path = tmp_path / "boolean.json"
    write(path, value)
    with pytest.raises(ValueError, match="localFilesOnly must be a boolean"):
        module().load_config(path)

    value = config()
    value["encoder"]["layerIndex"] = "9"
    path = tmp_path / "integer.json"
    write(path, value)
    with pytest.raises(ValueError, match="layerIndex must be an integer"):
        module().load_config(path)


def test_placeholder_revision_cannot_be_executed(tmp_path: Path) -> None:
    value = config()
    value["encoder"]["modelRevision"] = "0" * 40
    path = tmp_path / "placeholder.json"
    write(path, value)

    with pytest.raises(ValueError, match="placeholder"):
        module().load_config(path)
