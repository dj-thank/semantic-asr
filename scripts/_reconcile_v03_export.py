#!/usr/bin/env python3
"""One-shot reconciler for frozen Japanese target and feature export contracts."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def reconcile_targets() -> None:
    path = Path("src/semantic_asr/japanese_phonetic_targets.py")
    text = path.read_text(encoding="utf-8")
    marker = '''    def __post_init__(self) -> None:
        if not self.blank_symbol:
            raise ValueError("blank_symbol is required")
'''
    if "mapping_revision is not implemented" not in text:
        replacement = '''    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("pronunciation policy schema_version must be '1'")
        if not isinstance(self.ignore_punctuation, bool):
            raise TypeError("ignore_punctuation must be a boolean")
        if self.mapping_revision != "ja-kana-mora-phone-v1":
            raise ValueError("mapping_revision is not implemented by this frozen mapping")
        if not self.blank_symbol:
            raise ValueError("blank_symbol is required")
'''
        text = replace_once(text, marker, replacement, "pronunciation policy guard")
    marker = '''        if len(self.policy_digest) != 64:
            raise ValueError("policy_digest must be a SHA-256 value")
'''
    if "policy_digest must be hexadecimal" not in text:
        text = replace_once(
            text,
            marker,
            marker
            + '''        try:
            int(self.policy_digest, 16)
        except ValueError as exc:
            raise ValueError("policy_digest must be hexadecimal") from exc
''',
            "pronunciation digest hex guard",
        )
    path.write_text(text, encoding="utf-8")


def reconcile_runtime() -> None:
    path = Path("src/semantic_asr/joint_phonetic_runtime_optional.py")
    text = path.read_text(encoding="utf-8")
    if 'execution_device: str = "cpu"' not in text:
        text = replace_once(
            text,
            "    frame_stride_ms: float\n    model_artifact_sha256: str | None = None",
            "    frame_stride_ms: float\n"
            '    execution_device: str = "cpu"\n'
            '    compute_dtype: str = "float32"\n'
            "    model_artifact_sha256: str | None = None",
            "feature runtime fields",
        )
    marker = '''        stride = float(self.frame_stride_ms)
        if not math.isfinite(stride) or stride <= 0.0:
            raise ValueError("frame_stride_ms must be finite and positive")
        object.__setattr__(self, "frame_stride_ms", stride)
'''
    if "compute_dtype must be float32" not in text:
        replacement = '''        stride = float(self.frame_stride_ms)
        if not math.isfinite(stride) or stride <= 0.0:
            raise ValueError("frame_stride_ms must be finite and positive")
        if not self.execution_device or (
            self.execution_device not in {"cpu", "mps", "cuda"}
            and not self.execution_device.startswith("cuda:")
        ):
            raise ValueError("execution_device must be cpu, mps, cuda, or cuda:<index>")
        if self.compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("compute_dtype must be float32, float16, or bfloat16")
        if self.execution_device == "cpu" and self.compute_dtype == "float16":
            raise ValueError("float16 feature extraction is not supported on CPU")
        object.__setattr__(self, "frame_stride_ms", stride)
'''
        text = replace_once(text, marker, replacement, "feature runtime validation")
    init_marker = '''        self.config = config
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
'''
    if "runtime device differs from frozen feature config" not in text:
        init_replacement = '''        if device != config.execution_device:
            raise ValueError("runtime device differs from frozen feature config")
        self.config = config
        self.device = device
        self.local_files_only = local_files_only
        self.dtype = getattr(torch, config.compute_dtype)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
'''
        text = replace_once(text, init_marker, init_replacement, "transformers runtime identity")
    model_marker = '''        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(device)
'''
    if "torch_dtype=self.dtype" not in text:
        model_replacement = '''        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
            torch_dtype=self.dtype,
        ).to(device=device, dtype=self.dtype)
'''
        text = replace_once(text, model_marker, model_replacement, "transformers dtype binding")
    inputs_marker = '''        inputs = {
            name: value.to(self.device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
'''
    if "value.is_floating_point()" not in text:
        inputs_replacement = '''        inputs = {
            name: (
                value.to(
                    self.device,
                    dtype=self.dtype if value.is_floating_point() else value.dtype,
                )
                if hasattr(value, "to")
                else value
            )
            for name, value in inputs.items()
        }
'''
        text = replace_once(text, inputs_marker, inputs_replacement, "transformers input dtype")
    if "def runtime_provenance(self)" not in text:
        anchor = '''    def extract_features(
        self,
        samples: Sequence[float],
'''
        property_text = '''    @property
    def runtime_provenance(self) -> dict[str, object]:
        import transformers

        return {
            "backend": "transformers-auto-model-hidden-state-v1",
            "torchVersion": torch.__version__,
            "transformersVersion": transformers.__version__,
            "executionDevice": self.config.execution_device,
            "computeDtype": self.config.compute_dtype,
            "localFilesOnly": self.local_files_only,
        }

'''
        text = replace_once(text, anchor, property_text + anchor, "runtime provenance property")
    protocol_marker = '''class FrozenAudioFeatureBackend(Protocol):
    config: FrozenAudioFeatureConfig

    def extract_features(
'''
    if "runtime_provenance: dict[str, object]" not in text:
        protocol_replacement = '''class FrozenAudioFeatureBackend(Protocol):
    config: FrozenAudioFeatureConfig
    runtime_provenance: dict[str, object]

    def extract_features(
'''
        text = replace_once(text, protocol_marker, protocol_replacement, "backend protocol provenance")
    path.write_text(text, encoding="utf-8")


def reconcile_exporter() -> None:
    path = Path("src/semantic_asr/phonetic_feature_export.py")
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "from collections.abc import Sequence",
        "from collections.abc import Mapping, Sequence",
        1,
    )
    if "feature_backend_runtime_digest: str" not in text:
        text = replace_once(
            text,
            "    feature_backend_config_digest: str\n    feature_matrix_digest: str",
            "    feature_backend_config_digest: str\n"
            "    feature_backend_runtime_digest: str\n"
            "    feature_matrix_digest: str",
            "receipt runtime digest field",
        )
        text = replace_once(
            text,
            "            self.feature_backend_config_digest,\n            self.feature_matrix_digest,",
            "            self.feature_backend_config_digest,\n"
            "            self.feature_backend_runtime_digest,\n"
            "            self.feature_matrix_digest,",
            "receipt runtime digest validation",
        )
        text = replace_once(
            text,
            "    feature_backend_config_digest: str\n    pronunciation_policy_digest: str",
            "    feature_backend_config_digest: str\n"
            "    feature_backend_runtime_digest: str\n"
            "    pronunciation_policy_digest: str",
            "result runtime digest field",
        )
        text = replace_once(
            text,
            "            self.feature_backend_config_digest,\n            self.pronunciation_policy_digest,",
            "            self.feature_backend_config_digest,\n"
            "            self.feature_backend_runtime_digest,\n"
            "            self.pronunciation_policy_digest,",
            "result runtime digest validation",
        )
    config_marker = '''    def __post_init__(self) -> None:
        if self.feature_dtype not in {"float16", "float32", "float64"}:
'''
    if "feature export schema_version" not in text:
        config_replacement = '''    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("feature export schema_version must be '1'")
        if not isinstance(self.fsync_each_row, bool):
            raise TypeError("fsync_each_row must be a boolean")
        if self.feature_dtype not in {"float16", "float32", "float64"}:
'''
        text = replace_once(text, config_marker, config_replacement, "export config validation")
    if "def backend_runtime_provenance(self)" not in text:
        anchor = '''    @property
    def feature_revision(self) -> str:
'''
        helper = '''    @property
    def backend_runtime_provenance(self) -> dict[str, object]:
        value = getattr(self.feature_backend, "runtime_provenance", None)
        if callable(value):
            value = value()
        if not isinstance(value, Mapping):
            raise TypeError(
                "feature backend must expose an explicit runtime_provenance mapping"
            )
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))

    @property
    def backend_runtime_digest(self) -> str:
        return sha256_json(self.backend_runtime_provenance)

'''
        text = replace_once(text, anchor, helper + anchor, "backend runtime helper")
    text = text.replace(
        '                "pronunciationPolicyDigest": self.pronunciation_policy.digest,\n'
        '                "exportConfigDigest": self.config.digest,',
        '                "pronunciationPolicyDigest": self.pronunciation_policy.digest,\n'
        '                "featureBackendRuntimeDigest": self.backend_runtime_digest,\n'
        '                "exportConfigDigest": self.config.digest,',
    )
    text = text.replace(
        '                "featureBackendConfigDigest": self.feature_backend.config.digest,\n'
        '                "pronunciationPolicyDigest": self.pronunciation_policy.digest,',
        '                "featureBackendConfigDigest": self.feature_backend.config.digest,\n'
        '                "featureBackendRuntimeDigest": self.backend_runtime_digest,\n'
        '                "pronunciationPolicyDigest": self.pronunciation_policy.digest,',
    )
    if "feature_backend_runtime_digest=self.backend_runtime_digest" not in text:
        text = replace_once(
            text,
            "            feature_backend_config_digest=self.feature_backend.config.digest,\n"
            "            feature_matrix_digest=matrix.digest,",
            "            feature_backend_config_digest=self.feature_backend.config.digest,\n"
            "            feature_backend_runtime_digest=self.backend_runtime_digest,\n"
            "            feature_matrix_digest=matrix.digest,",
            "receipt runtime digest construction",
        )
    start = text.find("    def export(\n")
    if start < 0:
        raise RuntimeError("export method missing")
    replacement = '''    def _validated_rows(
        self,
        path: Path,
        source: PhoneticSourceManifest,
        *,
        output_root: Path,
        require_complete: bool,
    ) -> tuple[list[PhoneticFeatureItem], list[PhoneticFeatureReceipt], int]:
        items: list[PhoneticFeatureItem] = []
        receipts: list[PhoneticFeatureReceipt] = []
        total_samples = 0
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError("feature manifest contains an empty checkpoint row")
                if index >= len(source.items):
                    raise ValueError("feature manifest contains more rows than the source")
                item = _feature_item(json.loads(line))
                source_item = source.items[index]
                expected_pairs = {
                    "utterance_id": source_item.utterance_id,
                    "split": source_item.split,
                    "speaker_id": source_item.speaker_id,
                    "source_id": source_item.source_id,
                    "rights_decision": source_item.rights_decision,
                    "license_id": source_item.license_id,
                    "feature_revision": self.feature_revision,
                    "phone_inventory_digest": self.phone_inventory.digest,
                    "mora_inventory_digest": self.mora_inventory.digest,
                }
                for name, expected in expected_pairs.items():
                    if getattr(item, name) != expected:
                        raise ValueError(
                            f"checkpoint feature row differs from source/config: {name}"
                        )
                feature = (output_root / item.feature_path).resolve()
                try:
                    feature.relative_to(output_root.resolve())
                except ValueError as exc:
                    raise ValueError("checkpoint feature path escapes output root") from exc
                if not feature.exists() or file_sha256(feature) != item.feature_sha256:
                    raise ValueError("checkpoint feature file is absent or corrupted")
                sidecar = feature.with_suffix(".receipt.json")
                if not sidecar.exists():
                    raise ValueError("checkpoint feature receipt is absent")
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                expected_receipt_keys = {
                    *PhoneticFeatureReceipt.__dataclass_fields__,
                    "receiptDigest",
                }
                if set(payload) != expected_receipt_keys:
                    raise ValueError("feature receipt schema is not exact")
                receipt_digest = payload.pop("receiptDigest")
                receipt = PhoneticFeatureReceipt(**payload)
                if receipt.digest != receipt_digest:
                    raise ValueError("feature receipt digest mismatch")
                sample_start = math.floor(
                    source_item.segment_start_ms * source_item.sample_rate / 1000
                )
                sample_end = math.ceil(
                    source_item.segment_end_ms * source_item.sample_rate / 1000
                )
                if receipt.source_item_digest != source_item.digest:
                    raise ValueError("feature receipt is bound to a different source item")
                if receipt.source_manifest_digest != source.digest:
                    raise ValueError("feature receipt is bound to a different source manifest")
                if receipt.source_recording_file_sha256 != source_item.audio_sha256:
                    raise ValueError("feature receipt is bound to a different recording")
                if (receipt.sample_start, receipt.sample_end) != (sample_start, sample_end):
                    raise ValueError("feature receipt sample range differs from source row")
                if receipt.sample_rate != source_item.sample_rate:
                    raise ValueError("feature receipt sample rate differs from source row")
                if item.source_audio_sha256 != receipt.source_clip_sha256:
                    raise ValueError("trainer row and receipt use different clip hashes")
                if receipt.feature_path != item.feature_path:
                    raise ValueError("trainer row and receipt use different feature paths")
                if receipt.feature_sha256 != item.feature_sha256:
                    raise ValueError("trainer row and receipt use different feature hashes")
                if (
                    receipt.feature_backend_config_digest
                    != self.feature_backend.config.digest
                ):
                    raise ValueError("receipt uses a different feature backend config")
                if receipt.feature_backend_runtime_digest != self.backend_runtime_digest:
                    raise ValueError("receipt uses a different feature backend runtime")
                if receipt.export_config_digest != self.config.digest:
                    raise ValueError("receipt uses a different export config")
                total_samples += receipt.sample_end - receipt.sample_start
                if total_samples > self.source_resources.maximum_total_audio_samples:
                    raise ValueError("feature export exceeds maximum_total_audio_samples")
                items.append(item)
                receipts.append(receipt)
        if require_complete and len(items) != len(source.items):
            raise ValueError("completed feature manifest is not complete")
        return items, receipts, total_samples

    def _result(
        self,
        source: PhoneticSourceManifest,
        output: Path,
        *,
        run_digest: str,
        receipt_digests: Sequence[str],
    ) -> PhoneticFeatureExportResult:
        return PhoneticFeatureExportResult(
            output_manifest=output,
            output_manifest_sha256=file_sha256(output),
            item_count=len(receipt_digests),
            source_manifest_digest=source.digest,
            feature_backend_config_digest=self.feature_backend.config.digest,
            feature_backend_runtime_digest=self.backend_runtime_digest,
            pronunciation_policy_digest=self.pronunciation_policy.digest,
            phone_inventory_digest=self.phone_inventory.digest,
            mora_inventory_digest=self.mora_inventory.digest,
            feature_revision=self.feature_revision,
            export_config_digest=self.config.digest,
            run_digest=run_digest,
            receipt_digests=tuple(receipt_digests),
        )

    def _final_metadata(
        self,
        source: PhoneticSourceManifest,
        result: PhoneticFeatureExportResult,
    ) -> dict[str, object]:
        return {
            "schemaVersion": "2",
            "runDigest": result.run_digest,
            "resultDigest": result.digest,
            "outputManifestSha256": result.output_manifest_sha256,
            "itemCount": result.item_count,
            "receiptDigests": result.receipt_digests,
            "phoneInventory": asdict(self.phone_inventory),
            "moraInventory": asdict(self.mora_inventory),
            "pronunciationPolicy": asdict(self.pronunciation_policy),
            "pronunciationPolicyDigest": self.pronunciation_policy.digest,
            "featureBackendConfig": asdict(self.feature_backend.config),
            "featureBackendConfigDigest": self.feature_backend.config.digest,
            "featureBackendRuntime": self.backend_runtime_provenance,
            "featureBackendRuntimeDigest": self.backend_runtime_digest,
            "featureRevision": self.feature_revision,
            "sourceManifestDigest": source.digest,
            "exportConfig": asdict(self.config),
            "exportConfigDigest": self.config.digest,
            "claimBoundary": (
                "derived training features only; no model quality or runtime promotion claim"
            ),
        }

    def export(
        self,
        source: PhoneticSourceManifest,
        output_manifest: str | Path,
        *,
        allow_derived_export: bool,
        resume: bool = True,
    ) -> PhoneticFeatureExportResult:
        if not allow_derived_export:
            raise PermissionError(
                "phonetic feature export requires allow_derived_export=True"
            )
        output = Path(output_manifest)
        if output.suffix != ".jsonl":
            raise ValueError("output manifest must use the .jsonl suffix")
        output_root = output.resolve().parent
        if output_root == Path(output_root.anchor):
            raise ValueError("output manifest must not be written at filesystem root")
        output_root.mkdir(parents=True, exist_ok=True)
        run_digest = self._run_digest(source, output)
        partial = output.with_suffix(output.suffix + ".partial")
        checkpoint = output.with_suffix(output.suffix + ".partial.meta.json")
        final_meta = output.with_suffix(output.suffix + ".export.json")
        checkpoint_payload = {
            "schemaVersion": "2",
            "runDigest": run_digest,
            "sourceManifestDigest": source.digest,
            "featureBackendConfigDigest": self.feature_backend.config.digest,
            "featureBackendRuntimeDigest": self.backend_runtime_digest,
            "pronunciationPolicyDigest": self.pronunciation_policy.digest,
            "phoneInventoryDigest": self.phone_inventory.digest,
            "moraInventoryDigest": self.mora_inventory.digest,
            "featureRevision": self.feature_revision,
            "exportConfigDigest": self.config.digest,
        }

        if output.exists():
            if not final_meta.exists():
                if not checkpoint.exists():
                    raise ValueError(
                        "completed output is missing both export and recovery metadata"
                    )
                if json.loads(checkpoint.read_text(encoding="utf-8")) != checkpoint_payload:
                    raise ValueError("completed output recovery metadata differs from this run")
            items, receipts, _ = self._validated_rows(
                output,
                source,
                output_root=output_root,
                require_complete=True,
            )
            result = self._result(
                source,
                output,
                run_digest=run_digest,
                receipt_digests=tuple(receipt.digest for receipt in receipts),
            )
            if len(items) != result.item_count:
                raise ValueError("completed result count mismatch")
            expected_metadata = self._final_metadata(source, result)
            if final_meta.exists():
                metadata = json.loads(final_meta.read_text(encoding="utf-8"))
                if sha256_json(metadata) != sha256_json(expected_metadata):
                    raise ValueError("completed export metadata mismatch")
            else:
                _atomic_write_text(
                    final_meta,
                    json.dumps(
                        expected_metadata,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            checkpoint.unlink(missing_ok=True)
            return result

        completed: list[PhoneticFeatureItem] = []
        receipts: list[PhoneticFeatureReceipt] = []
        total_samples = 0
        if partial.exists():
            if not resume or not checkpoint.exists():
                raise ValueError(
                    "partial export exists without an enabled matching checkpoint"
                )
            if json.loads(checkpoint.read_text(encoding="utf-8")) != checkpoint_payload:
                raise ValueError("partial export checkpoint belongs to a different run")
            completed, receipts, total_samples = self._validated_rows(
                partial,
                source,
                output_root=output_root,
                require_complete=False,
            )
        else:
            _atomic_write_text(
                checkpoint,
                json.dumps(
                    checkpoint_payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )

        mode = "a" if completed else "w"
        with partial.open(mode, encoding="utf-8", newline="\n") as handle:
            for source_item in source.items[len(completed) :]:
                output_item, receipt = self._export_item(
                    source,
                    source_item,
                    output_root=output_root,
                )
                total_samples += receipt.sample_end - receipt.sample_start
                if total_samples > self.source_resources.maximum_total_audio_samples:
                    raise ValueError("feature export exceeds maximum_total_audio_samples")
                handle.write(
                    json.dumps(
                        _feature_row(output_item),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                handle.flush()
                if self.config.fsync_each_row:
                    os.fsync(handle.fileno())
                completed.append(output_item)
                receipts.append(receipt)

        os.replace(partial, output)
        result = self._result(
            source,
            output,
            run_digest=run_digest,
            receipt_digests=tuple(receipt.digest for receipt in receipts),
        )
        _atomic_write_text(
            final_meta,
            json.dumps(
                self._final_metadata(source, result),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        checkpoint.unlink(missing_ok=True)
        return result
'''
    text = text[:start] + replacement
    path.write_text(text, encoding="utf-8")


def reconcile_cli_and_examples() -> None:
    path = Path("scripts/export_phonetic_features.py")
    text = path.read_text(encoding="utf-8")
    if "def _json_boolean(" not in text:
        anchor = '''def _exact_object(value: object, expected: set[str], *, name: str) -> dict[str, object]:
'''
        helpers = '''def _json_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _json_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _json_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


'''
        text = replace_once(text, anchor, helpers + anchor, "CLI strict JSON helpers")
    if '"computeDtype"' not in text:
        text = replace_once(
            text,
            '            "device",\n            "localFilesOnly",',
            '            "device",\n            "computeDtype",\n            "localFilesOnly",',
            "CLI compute dtype schema",
        )
    replacements = {
        '        layer_index=int(encoder["layerIndex"]),': '        layer_index=_json_integer(encoder["layerIndex"], name="layerIndex"),',
        '        sample_rate=int(encoder["sampleRate"]),': '        sample_rate=_json_integer(encoder["sampleRate"], name="sampleRate"),',
        '        feature_dimension=int(encoder["featureDimension"]),': '        feature_dimension=_json_integer(\n            encoder["featureDimension"], name="featureDimension"\n        ),',
        '        frame_stride_ms=float(encoder["frameStrideMs"]),': '        frame_stride_ms=_json_number(encoder["frameStrideMs"], name="frameStrideMs"),\n        execution_device=str(encoder["device"]),\n        compute_dtype=str(encoder["computeDtype"]),',
        '        ignore_punctuation=bool(pronunciation["ignorePunctuation"]),': '        ignore_punctuation=_json_boolean(\n            pronunciation["ignorePunctuation"], name="ignorePunctuation"\n        ),',
        '        maximum_cached_recordings=int(export["maximumCachedRecordings"]),': '        maximum_cached_recordings=_json_integer(\n            export["maximumCachedRecordings"], name="maximumCachedRecordings"\n        ),',
        '        fsync_each_row=bool(export["fsyncEachRow"]),': '        fsync_each_row=_json_boolean(export["fsyncEachRow"], name="fsyncEachRow"),',
        '        maximum_items=int(resources["maximumItems"]),': '        maximum_items=_json_integer(resources["maximumItems"], name="maximumItems"),',
        '        maximum_reading_characters=int(resources["maximumReadingCharacters"]),': '        maximum_reading_characters=_json_integer(\n            resources["maximumReadingCharacters"], name="maximumReadingCharacters"\n        ),',
        '        maximum_segment_duration_ms=int(resources["maximumSegmentDurationMs"]),': '        maximum_segment_duration_ms=_json_integer(\n            resources["maximumSegmentDurationMs"], name="maximumSegmentDurationMs"\n        ),',
        '        maximum_total_audio_samples=int(resources["maximumTotalAudioSamples"]),': '        maximum_total_audio_samples=_json_integer(\n            resources["maximumTotalAudioSamples"], name="maximumTotalAudioSamples"\n        ),',
        '        maximum_recording_samples=int(resources["maximumRecordingSamples"]),': '        maximum_recording_samples=_json_integer(\n            resources["maximumRecordingSamples"], name="maximumRecordingSamples"\n        ),',
        '        "local_files_only": bool(encoder["localFilesOnly"]),': '        "local_files_only": _json_boolean(\n            encoder["localFilesOnly"], name="localFilesOnly"\n        ),',
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new, 1)
        elif new not in text:
            raise RuntimeError(f"CLI replacement anchor missing: {old}")
    if "placeholder encoder model revision" not in text:
        text = replace_once(
            text,
            "    feature_config = FrozenAudioFeatureConfig(\n",
            '    if encoder["modelRevision"] == "0" * 40:\n'
            '        raise ValueError("placeholder encoder model revision cannot be executed")\n'
            "    feature_config = FrozenAudioFeatureConfig(\n",
            "CLI placeholder revision guard",
        )
    text = text.replace(
        '        device=runtime["device"],',
        "        device=feature_config.execution_device,",
        1,
    )
    path.write_text(text, encoding="utf-8")

    path = Path("examples/phonetic_feature_export.config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["encoder"]["computeDtype"] = "float32"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    path = Path("tests/test_export_phonetic_features_config.py")
    text = path.read_text(encoding="utf-8")
    if '"computeDtype": "float32"' not in text:
        text = replace_once(
            text,
            '            "device": "cpu",\n            "localFilesOnly": True,',
            '            "device": "cpu",\n'
            '            "computeDtype": "float32",\n'
            '            "localFilesOnly": True,',
            "config test compute dtype",
        )
    path.write_text(text, encoding="utf-8")
    path = Path("docs/PHONETIC_FEATURE_EXPORT.md")
    text = path.read_text(encoding="utf-8").replace(
        '    "device": "cpu",\n    "localFilesOnly": true',
        '    "device": "cpu",\n    "computeDtype": "float32",\n    "localFilesOnly": true',
    )
    path.write_text(text, encoding="utf-8")


def reconcile_tests_and_package() -> None:
    path = Path("tests/test_phonetic_feature_export.py")
    text = path.read_text(encoding="utf-8")
    if "def runtime_provenance(self)" not in text:
        marker = '''        self.calls = 0

    def extract_features(self, samples, *, sample_rate, source_audio_sha256):
'''
        replacement = '''        self.calls = 0

    @property
    def runtime_provenance(self):
        return {
            "backend": "fake-feature-backend-v1",
            "executionDevice": "cpu",
            "computeDtype": "float32",
            "runtimeVersion": "fixture-r1",
        }

    def extract_features(self, samples, *, sample_rate, source_audio_sha256):
'''
        text = replace_once(text, marker, replacement, "fake backend provenance")
    path.write_text(text, encoding="utf-8")

    path = Path("pyproject.toml")
    text = path.read_text(encoding="utf-8")
    if "phonetic-export = [" not in text:
        marker = "\npublic-data = ["
        block = '''
phonetic-export = [
  "numpy>=1.26",
  "soundfile>=0.12",
  "torch>=2.4",
  "transformers>=4.45",
  "safetensors>=0.4",
]
'''
        if marker not in text:
            raise RuntimeError("pyproject public-data extras anchor missing")
        text = text.replace(marker, "\n" + block.lstrip() + marker.lstrip(), 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    reconcile_targets()
    reconcile_runtime()
    reconcile_exporter()
    reconcile_cli_and_examples()
    reconcile_tests_and_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
