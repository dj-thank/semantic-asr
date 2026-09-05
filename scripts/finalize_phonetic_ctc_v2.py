#!/usr/bin/env python3
"""One-shot idempotent source migration for the dual CTC v2 audit."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def save(path: str, text: str) -> None:
    (ROOT / path).write_text(text.rstrip() + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(f"missing migration marker: {label}")
    if text.count(old) != 1:
        raise RuntimeError(f"non-unique migration marker: {label}")
    return text.replace(old, new, 1)


def patch_manifest() -> None:
    path = "src/semantic_asr/phonetic_runtime/manifest.py"
    text = load(path)
    text = text.replace(
        'SplitName = Literal["train", "calibration", "test"]',
        'SplitName = Literal["train", "validation", "calibration", "test"]',
    )
    text = text.replace(
        '        if self.split not in {"train", "calibration", "test"}:',
        '        if self.split not in {"train", "validation", "calibration", "test"}:',
    )
    pattern = re.compile(
        r"def validate_split_isolation\(manifest: PhoneticSplitManifest\) -> None:\n"
        r".*?\n\ndef load_phonetic_manifest",
        re.DOTALL,
    )
    replacement = """def validate_split_isolation(
    manifest: PhoneticSplitManifest,
    *,
    required_splits: tuple[SplitName, ...] = ("train", "validation", "calibration"),
) -> None:
    split_names: tuple[SplitName, ...] = (
        "train",
        "validation",
        "calibration",
        "test",
    )
    if not required_splits:
        raise ValueError("required_splits must not be empty")
    if len(required_splits) != len(set(required_splits)):
        raise ValueError("required_splits must be unique")
    unknown = set(required_splits) - set(split_names)
    if unknown:
        raise ValueError(f"unknown required phonetic splits: {sorted(unknown)}")
    rows_by_split = {split: manifest.rows_for(split) for split in split_names}
    for split in required_splits:
        if not rows_by_split[split]:
            raise ValueError(f"phonetic manifest requires a non-empty {split} split")
    speakers = {
        split: {row.speaker_id for row in rows}
        for split, rows in rows_by_split.items()
    }
    sessions = {
        split: {row.session_id for row in rows}
        for split, rows in rows_by_split.items()
    }
    sources = {
        split: {row.source_id for row in rows}
        for split, rows in rows_by_split.items()
    }
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            if speakers[left].intersection(speakers[right]):
                raise ValueError(f"speaker leakage between {left} and {right}")
            if sessions[left].intersection(sessions[right]):
                raise ValueError(f"session leakage between {left} and {right}")
            if sources[left].intersection(sources[right]):
                raise ValueError(f"source leakage between {left} and {right}")


def load_phonetic_manifest"""
    if "required_splits: tuple[SplitName, ...]" not in text:
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError("could not replace split isolation function")
    save(path, text)


def patch_training() -> None:
    path = "src/semantic_asr/phonetic_runtime/training.py"
    text = load(path)
    replacements = {
        "calibration_total_loss": "validation_total_loss",
        "calibration_phone_loss": "validation_phone_loss",
        "calibration_mora_loss": "validation_mora_loss",
        "best_calibration_loss": "best_validation_loss",
        'calibration_rows = manifest.rows_for("calibration")': (
            'validation_rows = manifest.rows_for("validation")'
        ),
        "calibration_total, calibration_phone, calibration_mora, _ = _run_split(\n"
        "            model,\n"
        "            calibration_rows,": (
            "validation_total, validation_phone, validation_mora, _ = _run_split(\n"
            "            model,\n"
            "            validation_rows,"
        ),
        "if calibration_total < best_loss:": "if validation_total < best_loss:",
        "best_loss = calibration_total": "best_loss = validation_total",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(
        "validation_total_loss=calibration_total,\n"
        "                validation_phone_loss=calibration_phone,\n"
        "                validation_mora_loss=calibration_mora,",
        "validation_total_loss=validation_total,\n"
        "                validation_phone_loss=validation_phone,\n"
        "                validation_mora_loss=validation_mora,",
    )
    text = text.replace(
        '    if not rows:\n        raise ValueError("training/calibration split must not be empty")',
        '    if not rows:\n        raise ValueError("training/validation split must not be empty")',
    )
    save(path, text)


def patch_torch_model() -> None:
    path = "src/semantic_asr/phonetic_runtime/torch_model.py"
    text = load(path)
    if "def minimum_ctc_frames(" not in text:
        marker = "\ndef multitask_ctc_loss(\n"
        addition = '''

def minimum_ctc_frames(targets: Tensor, target_lengths: Tensor) -> Tensor:
    """Return the minimum CTC frame count, including repeated-label separators."""

    if targets.ndim != 1:
        raise ValueError("CTC targets must be one-dimensional and concatenated")
    if target_lengths.ndim != 1:
        raise ValueError("CTC target_lengths must have shape [batch]")
    if torch.any(target_lengths < 1):
        raise ValueError("CTC target lengths must be positive")
    if int(target_lengths.sum().item()) != targets.numel():
        raise ValueError("CTC target lengths do not match the concatenated target tensor")
    output: list[int] = []
    cursor = 0
    for length in target_lengths.tolist():
        row = targets[cursor : cursor + length]
        repeats = int((row[1:] == row[:-1]).sum().item()) if length > 1 else 0
        output.append(int(length) + repeats)
        cursor += int(length)
    return torch.tensor(output, dtype=torch.long, device=target_lengths.device)


def _validate_ctc_feasibility(
    *,
    name: str,
    targets: Tensor,
    target_lengths: Tensor,
    output_lengths: Tensor,
    blank_id: int,
) -> None:
    if output_lengths.ndim != 1 or output_lengths.shape != target_lengths.shape:
        raise ValueError(f"{name} output and target lengths have incompatible shapes")
    if torch.any(targets == blank_id):
        raise ValueError(f"{name} targets must not contain the CTC blank")
    required = minimum_ctc_frames(targets, target_lengths)
    available = output_lengths.to(required.device)
    if torch.any(required > available):
        index = int(torch.nonzero(required > available, as_tuple=False)[0].item())
        raise ValueError(
            f"{name} target requires more CTC frames than available for batch row {index}: "
            f"required={int(required[index])}, available={int(available[index])}"
        )
'''
        if marker not in text:
            raise RuntimeError("multitask CTC loss insertion marker missing")
        text = text.replace(marker, addition + marker, 1)
    signature_marker = (
        "    if phone_weight < 0.0 or mora_weight < 0.0 or phone_weight + mora_weight <= 0.0:\n"
    )
    validation = """    _validate_ctc_feasibility(
        name="phone",
        targets=phone_targets,
        target_lengths=phone_target_lengths,
        output_lengths=output.output_lengths,
        blank_id=phone_blank_id,
    )
    _validate_ctc_feasibility(
        name="mora",
        targets=mora_targets,
        target_lengths=mora_target_lengths,
        output_lengths=output.output_lengths,
        blank_id=mora_blank_id,
    )
"""
    if 'name="phone",\n        targets=phone_targets' not in text:
        text = replace_once(
            text,
            signature_marker,
            validation + signature_marker,
            label="CTC feasibility validation",
        )
    text = text.replace("zero_infinity=True", "zero_infinity=False")
    save(path, text)


def patch_artifact() -> None:
    path = "src/semantic_asr/phonetic_runtime/artifact.py"
    text = load(path)
    if "import io\n" not in text:
        text = text.replace("import hashlib\n", "import hashlib\nimport io\n", 1)
    if "import zipfile\n" not in text:
        text = text.replace("import tempfile\n", "import tempfile\nimport zipfile\n", 1)
    if "def _write_deterministic_npz(" not in text:
        marker = (
            "\ndef _metadata_payload(metadata: DualCTCArtifactMetadata) -> dict[str, object]:\n"
        )
        addition = '''

def _write_deterministic_npz(path: Path, arrays: dict[str, Any]) -> None:
    """Write stable NPZ bytes for identical ordered tensor arrays."""

    import numpy as np

    with path.open("wb") as raw_stream:
        with zipfile.ZipFile(
            raw_stream,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(arrays):
                array_value = arrays[name]
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, array_value, allow_pickle=False)
                info = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        raw_stream.flush()
        os.fsync(raw_stream.fileno())
'''
        if marker not in text:
            raise RuntimeError("deterministic NPZ insertion marker missing")
        text = text.replace(marker, addition + marker, 1)
    text = text.replace(
        "        np.savez_compressed(weights_path, **arrays)",
        "        _write_deterministic_npz(weights_path, arrays)",
    )
    save(path, text)


def patch_inference() -> None:
    path = "src/semantic_asr/phonetic_runtime/inference.py"
    text = load(path)
    old = """        distribution = {
            symbol: max(0.0, value / total)
            for symbol, value in zip(inventory.symbols, row, strict=True)
        }
        renormalizer = sum(distribution.values())
        distribution[inventory.symbols[-1]] += 1.0 - renormalizer
"""
    new = """        normalized = [max(0.0, value) / total for value in row]
        renormalizer = sum(normalized)
        normalized = [value / renormalizer for value in normalized]
        anchor = max(range(len(normalized)), key=lambda index: normalized[index])
        normalized[anchor] += 1.0 - sum(normalized)
        distribution = dict(zip(inventory.symbols, normalized, strict=True))
"""
    if old in text:
        text = text.replace(old, new, 1)
    save(path, text)


def patch_provider() -> None:
    path = "src/semantic_asr/phonetic_runtime/provider.py"
    text = load(path)
    text = text.replace(
        "from dataclasses import dataclass\n",
        "from dataclasses import dataclass\n",
    )
    if "maximum_total_audio_ms" not in text:
        text = text.replace(
            "    maximum_lexicon_entries: int = 2_048\n",
            "    maximum_lexicon_entries: int = 2_048\n    maximum_total_audio_ms: int = 8_000\n",
            1,
        )
        text = text.replace(
            '            "maximum_lexicon_entries",\n',
            '            "maximum_lexicon_entries",\n            "maximum_total_audio_ms",\n',
            1,
        )
    if "utility_artifact_digest" not in text:
        text = text.replace(
            "    config: PhoneticProposalProviderConfig = PhoneticProposalProviderConfig()\n",
            "    config: PhoneticProposalProviderConfig = PhoneticProposalProviderConfig()\n"
            "    utility_artifact_digest: str | None = None\n",
            1,
        )
        text = text.replace(
            '        if self.mora_calibration.channel != "mora":\n'
            '            raise ValueError("mora calibration must emit the mora utility channel")\n',
            '        if self.mora_calibration.channel != "mora":\n'
            '            raise ValueError("mora calibration must emit the mora utility channel")\n'
            "        if self.utility_artifact_digest is not None and not _is_sha256(\n"
            "            self.utility_artifact_digest\n"
            "        ):\n"
            '            raise ValueError("utility_artifact_digest must be a SHA-256 value")\n',
            1,
        )
        class_marker = "    def _selected_spans(self, build: SemanticDeliberationBuild) -> tuple[DeliberationSpan, ...]:\n"
        classmethod = """    @classmethod
    def from_utility_artifact(
        cls,
        *,
        runtime: PhoneMoraPosteriorRuntime,
        lexicon_provider: SpanLexiconProvider,
        utility_artifact,
        config: PhoneticProposalProviderConfig | None = None,
    ) -> SourceAudioPhoneticProposalProvider:
        if utility_artifact.runtime_profile_digest != runtime.profile_digest:
            raise ValueError("utility artifact belongs to a different runtime profile")
        return cls(
            runtime=runtime,
            lexicon_provider=lexicon_provider,
            phone_calibration=utility_artifact.phone_profile,
            mora_calibration=utility_artifact.mora_profile,
            config=config or PhoneticProposalProviderConfig(),
            utility_artifact_digest=utility_artifact.digest,
        )

"""
        if class_marker not in text:
            raise RuntimeError("provider classmethod insertion marker missing")
        text = text.replace(class_marker, classmethod + class_marker, 1)
    if "total_audio_ms = 0" not in text:
        text = text.replace(
            "        output: dict[str, tuple[VerifiedSpanProposal, ...]] = {}\n",
            "        output: dict[str, tuple[VerifiedSpanProposal, ...]] = {}\n"
            "        total_audio_ms = 0\n",
            1,
        )
        budget_marker = '            if end_ms <= start_ms:\n                raise ValueError("phonetic proposal crop has a non-positive duration")\n'
        budget = (
            "            crop_duration_ms = end_ms - start_ms\n"
            "            if total_audio_ms + crop_duration_ms > self.config.maximum_total_audio_ms:\n"
            "                break\n"
            "            total_audio_ms += crop_duration_ms\n"
        )
        text = replace_once(
            text,
            budget_marker,
            budget_marker + budget,
            label="provider total audio budget",
        )
    save(path, text)


def patch_scripts_and_docs() -> None:
    training_script = "scripts/train_dual_phonetic_ctc.py"
    text = load(training_script).replace(
        '"bestCalibrationLoss": result.best_calibration_loss,',
        '"bestValidationLoss": result.best_validation_loss,',
    )
    save(training_script, text)

    evaluation_script = "scripts/evaluate_dual_phonetic_ctc.py"
    text = load(evaluation_script).replace(
        'choices=("train", "calibration", "test")',
        'choices=("train", "validation", "calibration", "test")',
    )
    save(evaluation_script, text)

    documentation = "docs/PHONETIC_CTC_RUNTIME.md"
    text = load(documentation)
    text = text.replace(
        "Train, calibration, and test partitions reject shared speakers, sessions, or\nsource recordings.",
        "Train, validation, calibration, and test partitions reject shared speakers, sessions, or\nsource recordings.",
    )
    text = text.replace(
        "evaluates calibration loss after each epoch, keeps the\nbest calibration checkpoint",
        "evaluates validation loss after each epoch, keeps the\nbest validation checkpoint",
    )
    if "Four-way split boundary" not in text:
        text += """

## Four-way split boundary

The model uses four separate evidence partitions:

```text
train       gradient updates
validation  checkpoint and hyperparameter selection
calibration CTC utility normalization for candidate fusion
test        final PER/MER and end-to-end ASR evaluation
```

Validation and calibration must not share speakers, sessions, or source recordings. The test split
is never used for checkpoint selection or utility normalization. This separation prevents an
apparently held-out utility profile from being fitted on the same rows already used to select the
model checkpoint.

## Repeated-label CTC feasibility

A target such as `a a` needs a blank-separated path and therefore at least three acoustic frames.
The runtime computes `target_length + adjacent_repeat_count` before calling the CTC loss. Impossible
alignments fail explicitly; `zero_infinity` is not used to silently turn them into zero-loss rows.

## Deterministic weight archives

Tensor arrays are written in sorted order to a ZIP archive with fixed timestamps, permissions, and
compression settings. Identical state dictionaries therefore produce byte-identical `weights.npz`
files and the same weight-file SHA-256.
"""
    save(documentation, text)


def patch_tests() -> None:
    split_test = "tests/test_phonetic_split_v2.py"
    text = load(split_test).replace(
        "def test_four_way_split_isolation_passes() -> None:\n"
        '    value = four_way_manifest(Path(pytest.ensuretemp("four-way-split")))',
        "def test_four_way_split_isolation_passes(tmp_path: Path) -> None:\n"
        "    value = four_way_manifest(tmp_path)",
    )
    save(split_test, text)

    training_test = "tests/test_phonetic_training_optional.py"
    text = load(training_test)
    if 'manifest_row(tmp_path, "validation", 3)' not in text:
        text = text.replace(
            '            manifest_row(tmp_path, "calibration", 3),\n'
            '            manifest_row(tmp_path, "calibration", 4),\n',
            '            manifest_row(tmp_path, "validation", 3),\n'
            '            manifest_row(tmp_path, "validation", 4),\n'
            '            manifest_row(tmp_path, "calibration", 5),\n',
            1,
        )
    text = text.replace("result.best_calibration_loss", "result.best_validation_loss")
    save(training_test, text)

    evaluation_test = "tests/test_phonetic_runtime_evaluation.py"
    text = load(evaluation_test).replace(
        'for index, split in enumerate(("train", "calibration", "test"), 1):',
        'for index, split in enumerate(("train", "validation", "calibration", "test"), 1):',
    )
    save(evaluation_test, text)

    manifest_test = "tests/test_phonetic_runtime_manifest.py"
    text = load(manifest_test)
    if 'row(tmp_path, "validation", 2)' not in text:
        text = text.replace(
            '            row(tmp_path, "train", 1),\n'
            '            row(tmp_path, "calibration", 2),\n'
            '            row(tmp_path, "test", 3),\n',
            '            row(tmp_path, "train", 1),\n'
            '            row(tmp_path, "validation", 2),\n'
            '            row(tmp_path, "calibration", 3),\n'
            '            row(tmp_path, "test", 4),\n',
            1,
        )
    if "validation = dict(payload)" not in text:
        marker = "    calibration = dict(payload)\n"
        addition = """    validation = dict(payload)
    validation.update(
        {
            "utteranceId": "validation-1",
            "audioPath": str((tmp_path / "validation.wav").resolve()),
            "sourceAudioSha256": write_wav(tmp_path / "validation.wav", frequency=500.0),
            "speakerId": "speaker-validation",
            "sessionId": "session-validation",
            "sourceId": "source-validation",
            "split": "validation",
        }
    )
"""
        text = replace_once(text, marker, addition + marker, label="manifest validation row")
        text = text.replace(
            '        json.dumps(payload) + "\\n" + json.dumps(calibration) + "\\n",',
            "        json.dumps(payload)\n"
            '        + "\\n"\n'
            "        + json.dumps(validation)\n"
            '        + "\\n"\n'
            "        + json.dumps(calibration)\n"
            '        + "\\n",',
            1,
        )
        text = text.replace("assert len(loaded.rows) == 2", "assert len(loaded.rows) == 3")
    save(manifest_test, text)


def main() -> int:
    patch_manifest()
    patch_training()
    patch_torch_model()
    patch_artifact()
    patch_inference()
    patch_provider()
    patch_scripts_and_docs()
    patch_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
