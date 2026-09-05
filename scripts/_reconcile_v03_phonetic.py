#!/usr/bin/env python3
"""One-shot reconciler for joint phone/mora training contracts."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label}, found {count}")
    return text.replace(old, new, 1)


def reconcile_dataset() -> None:
    path = Path("src/semantic_asr/phonetic_dataset.py")
    text = path.read_text(encoding="utf-8")
    if "def minimum_ctc_frames(" not in text:
        marker = '''def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
'''
        helper = marker + '''

def minimum_ctc_frames(target: Sequence[int]) -> int:
    if not target:
        raise ValueError("CTC target must not be empty")
    return len(target) + sum(
        left == right for left, right in zip(target, target[1:], strict=False)
    )


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _target_ids(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty JSON integer array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{name} must contain non-negative integer IDs")
    return tuple(value)
'''
        text = replace_once(text, marker, helper, "phonetic dataset helpers")
    marker = '''    def __post_init__(self) -> None:
        if not self.utterance_id or not self.feature_path:
'''
    if "schema_version must be '1'" not in text:
        text = replace_once(
            text,
            marker,
            '''    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError("schema_version must be '1'")
        if not self.utterance_id or not self.feature_path:
''',
            "phonetic item schema version",
        )
    old = '''        for name in ("frame_count", "feature_dimension"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
'''
    if old in text:
        text = text.replace(
            old,
            '''        for name in ("frame_count", "feature_dimension"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
''',
            1,
        )
    if "phone target requires more CTC frames" not in text:
        marker = '''        if not self.speaker_id or not self.source_id or not self.feature_revision:
            raise ValueError("speaker, source, and feature revision are required")
'''
        replacement = '''        if self.frame_count < minimum_ctc_frames(self.phone_targets):
            raise ValueError("phone target requires more CTC frames than available")
        if self.frame_count < minimum_ctc_frames(self.mora_targets):
            raise ValueError("mora target requires more CTC frames than available")
        if not self.speaker_id or not self.source_id or not self.feature_revision:
            raise ValueError("speaker, source, and feature revision are required")
'''
        text = replace_once(text, marker, replacement, "CTC feasibility guard")
    text = text.replace(
        '        frame_count=int(row["frameCount"]),\n'
        '        feature_dimension=int(row["featureDimension"]),',
        '        frame_count=_positive_integer(row["frameCount"], name="frameCount"),\n'
        '        feature_dimension=_positive_integer(\n'
        '            row["featureDimension"], name="featureDimension"\n'
        '        ),',
    )
    text = text.replace(
        '        phone_targets=tuple(int(value) for value in row["phoneTargets"]),  # type: ignore[union-attr]\n'
        '        mora_targets=tuple(int(value) for value in row["moraTargets"]),  # type: ignore[union-attr]',
        '        phone_targets=_target_ids(row["phoneTargets"], name="phoneTargets"),\n'
        '        mora_targets=_target_ids(row["moraTargets"], name="moraTargets"),',
    )
    path.write_text(text, encoding="utf-8")


def reconcile_metrics() -> None:
    path = Path("src/semantic_asr/phonetic_training.py")
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"@dataclass\(frozen=True, slots=True\)\nclass PhoneticValidationMetrics:.*?"
        r"(?=\n\n@dataclass\(frozen=True, slots=True\)|\n\ndef )",
        re.S,
    )
    replacement = '''
@dataclass(frozen=True, slots=True)
class PhoneticValidationMetrics:
    phone_error_rate: float
    mora_error_rate: float
    phone_candidate_auc: float
    mora_candidate_auc: float
    critical_false_accept_rate: float
    validation_sample_count: int

    def __post_init__(self) -> None:
        for name in ("phone_error_rate", "mora_error_rate"):
            value = _strict_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "phone_candidate_auc",
            "mora_candidate_auc",
            "critical_false_accept_rate",
        ):
            value = _strict_real(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.validation_sample_count, bool)
            or not isinstance(self.validation_sample_count, int)
            or self.validation_sample_count < 1
        ):
            raise ValueError("validation_sample_count must be a positive integer")
'''
    match = pattern.search(text)
    if match is None:
        raise RuntimeError("PhoneticValidationMetrics class not found")
    text = text[: match.start()] + replacement.lstrip() + text[match.end() :]
    path.write_text(text, encoding="utf-8")


def reconcile_trainer() -> None:
    path = Path("src/semantic_asr/phonetic_trainer_optional.py")
    text = path.read_text(encoding="utf-8")
    if "calibration: PhoneticSequenceCalibration | None = None," not in text:
        text = replace_once(
            text,
            "    fit_calibration: bool = False,\n    target_true_accept_rate: float = 0.95,",
            "    fit_calibration: bool = False,\n"
            "    calibration: PhoneticSequenceCalibration | None = None,\n"
            "    target_true_accept_rate: float = 0.95,",
            "trainer calibration argument",
        )
    if "cannot fit and apply sequence calibration simultaneously" not in text:
        marker = '''    if not 0.0 < target_true_accept_rate <= 1.0:
        raise ValueError("target_true_accept_rate must be in (0, 1]")
'''
        replacement = marker + '''    if fit_calibration and calibration is not None:
        raise ValueError("cannot fit and apply sequence calibration simultaneously")
    if (
        calibration is not None
        and calibration.calibration_manifest_sha256 == manifest.manifest_sha256
    ):
        raise ValueError("locked evaluation must not fit thresholds on its own manifest")
'''
        text = replace_once(text, marker, replacement, "trainer calibration guard")
    if "int(self.calibration_manifest_sha256, 16)" not in text:
        marker = '''        if len(self.calibration_manifest_sha256) != 64:
            raise ValueError("calibration_manifest_sha256 must be a SHA-256 value")
'''
        replacement = marker + '''        try:
            int(self.calibration_manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("calibration_manifest_sha256 must be hexadecimal") from exc
'''
        text = replace_once(text, marker, replacement, "calibration digest hex check")
    old = '''    phone_threshold, phone_false_accept = _threshold(
        phone_positive,
        phone_negative,
        true_accept_rate=target_true_accept_rate,
    )
    mora_threshold, mora_false_accept = _threshold(
        mora_positive,
        mora_negative,
        true_accept_rate=target_true_accept_rate,
    )
    calibration = (
        PhoneticSequenceCalibration(
            phone_threshold=phone_threshold,
            mora_threshold=mora_threshold,
            target_true_accept_rate=target_true_accept_rate,
            calibration_manifest_sha256=manifest.manifest_sha256,
            revision=calibration_revision,
            phone_false_accept_rate=phone_false_accept,
            mora_false_accept_rate=mora_false_accept,
        )
        if fit_calibration
        else None
    )
'''
    if old in text:
        new = '''    if fit_calibration:
        phone_threshold, phone_false_accept = _threshold(
            phone_positive,
            phone_negative,
            true_accept_rate=target_true_accept_rate,
        )
        mora_threshold, mora_false_accept = _threshold(
            mora_positive,
            mora_negative,
            true_accept_rate=target_true_accept_rate,
        )
        used_calibration = PhoneticSequenceCalibration(
            phone_threshold=phone_threshold,
            mora_threshold=mora_threshold,
            target_true_accept_rate=target_true_accept_rate,
            calibration_manifest_sha256=manifest.manifest_sha256,
            revision=calibration_revision,
            phone_false_accept_rate=phone_false_accept,
            mora_false_accept_rate=mora_false_accept,
        )
    elif calibration is not None:
        used_calibration = calibration
        phone_false_accept = (
            sum(value >= calibration.phone_threshold for value in phone_negative)
            / len(phone_negative)
            if phone_negative
            else 0.0
        )
        mora_false_accept = (
            sum(value >= calibration.mora_threshold for value in mora_negative)
            / len(mora_negative)
            if mora_negative
            else 0.0
        )
    else:
        used_calibration = None
        phone_false_accept = 0.0
        mora_false_accept = 0.0
'''
        text = text.replace(old, new, 1)
    text = text.replace("        calibration=calibration,", "        calibration=used_calibration,", 1)
    path.write_text(text, encoding="utf-8")


def reconcile_head_loss() -> None:
    path = Path("src/semantic_asr/phonetic_heads_optional.py")
    text = path.read_text(encoding="utf-8")
    if "def _validate_ctc_batch(" not in text:
        anchor = "\n\nclass JointPhoneMoraCTCHead"
        helper = '''

def _validate_ctc_batch(
    targets: Tensor,
    target_lengths: Tensor,
    input_lengths: Tensor,
    *,
    name: str,
) -> None:
    offset = 0
    for index, length_value in enumerate(target_lengths.tolist()):
        length = int(length_value)
        sequence = tuple(int(value) for value in targets[offset : offset + length].tolist())
        offset += length
        minimum = length + sum(
            left == right for left, right in zip(sequence, sequence[1:], strict=False)
        )
        if int(input_lengths[index]) < minimum:
            raise ValueError(f"{name} target requires more CTC frames than available")
    if offset != targets.numel():
        raise ValueError(f"{name} target lengths do not cover the flattened target tensor")
'''
        if anchor not in text:
            raise RuntimeError("joint head class anchor missing")
        text = text.replace(anchor, helper + anchor, 1)
    if "_validate_ctc_batch(\n            phone_targets" not in text:
        marker = '''        phone_log_probs = torch.log_softmax(output.phone_logits, dim=-1).transpose(0, 1)
'''
        insertion = '''        _validate_ctc_batch(
            phone_targets,
            phone_target_lengths,
            input_lengths,
            name="phone",
        )
        _validate_ctc_batch(
            mora_targets,
            mora_target_lengths,
            input_lengths,
            name="mora",
        )
'''
        text = replace_once(text, marker, insertion + marker, "joint head CTC validation")
    path.write_text(text, encoding="utf-8")


def reconcile_cli() -> None:
    path = Path("scripts/train_joint_phonetic_head.py")
    text = path.read_text(encoding="utf-8")
    if "calibration=calibration_result.calibration," not in text:
        text = replace_once(
            text,
            "        fit_calibration=False,\n        target_true_accept_rate=args.target_true_accept_rate,",
            "        fit_calibration=False,\n"
            "        calibration=calibration_result.calibration,\n"
            "        target_true_accept_rate=args.target_true_accept_rate,",
            "locked test calibration use",
        )
    if "if calibration_result.calibration is None:" not in text:
        marker = '''    test_result = evaluate_joint_phonetic_head(
'''
        check = '''    if calibration_result.calibration is None:
        raise RuntimeError("calibration split did not produce a sequence calibration profile")
    test_result = evaluate_joint_phonetic_head(
'''
        text = replace_once(text, marker, check, "calibration result assertion")
    path.write_text(text, encoding="utf-8")


def reconcile_package() -> None:
    path = Path("pyproject.toml")
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^train\s*=\s*\[(.*?)^\]", text)
    if match is None:
        raise RuntimeError("train optional dependency block not found")
    body = match.group(1)
    additions = []
    if '"numpy' not in body:
        additions.append('  "numpy>=1.26",\n')
    if '"safetensors' not in body:
        additions.append('  "safetensors>=0.4",\n')
    if additions:
        text = text[: match.start(1)] + "".join(additions) + body + text[match.end(1) :]
    path.write_text(text, encoding="utf-8")


def main() -> int:
    reconcile_dataset()
    reconcile_metrics()
    reconcile_trainer()
    reconcile_head_loss()
    reconcile_cli()
    reconcile_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
