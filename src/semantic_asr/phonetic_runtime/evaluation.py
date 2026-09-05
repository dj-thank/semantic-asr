"""Phone/mora sequence and runtime evaluation for a frozen dual CTC artifact."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from ..contracts import sha256_json
from ..phonetic_evidence import PosteriorSequence
from .inference import DualCTCPosteriorRuntime
from .manifest import PhoneticSplitManifest, SplitName


def _edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_item in enumerate(left, 1):
        current = [row_index]
        for column_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[column_index] + 1,
                    current[column_index - 1] + 1,
                    previous[column_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def greedy_posterior_symbols(posterior: PosteriorSequence) -> tuple[str, ...]:
    output: list[str] = []
    previous = posterior.blank_symbol
    for frame in posterior.frames:
        symbol = max(frame.probabilities, key=lambda item: (item[1], item[0]))[0]
        if symbol != posterior.blank_symbol and symbol != previous:
            output.append(symbol)
        previous = symbol
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PhoneticUtteranceEvaluation:
    utterance_id: str
    source_audio_sha256: str
    phone_reference_count: int
    phone_edits: int
    mora_reference_count: int
    mora_edits: int
    phone_prediction: tuple[str, ...]
    mora_prediction: tuple[str, ...]
    phone_posterior_digest: str
    mora_posterior_digest: str
    latency_ms: float
    python_peak_bytes: int

    def __post_init__(self) -> None:
        if not self.utterance_id:
            raise ValueError("utterance_id is required")
        for name in (
            "phone_reference_count",
            "phone_edits",
            "mora_reference_count",
            "mora_edits",
            "python_peak_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.phone_reference_count < 1 or self.mora_reference_count < 1:
            raise ValueError("phonetic references must be non-empty")
        latency = float(self.latency_ms)
        if not math.isfinite(latency) or latency < 0.0:
            raise ValueError("latency_ms must be finite and non-negative")
        object.__setattr__(self, "latency_ms", latency)

    @property
    def phone_error_rate(self) -> float:
        return self.phone_edits / self.phone_reference_count

    @property
    def mora_error_rate(self) -> float:
        return self.mora_edits / self.mora_reference_count

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class PhoneticEvaluationReport:
    runtime_profile_digest: str
    manifest_digest: str
    split: SplitName
    utterances: tuple[PhoneticUtteranceEvaluation, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.utterances:
            raise ValueError("phonetic evaluation requires utterances")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("phonetic evaluation split is invalid")

    @property
    def phone_error_rate(self) -> float:
        return sum(row.phone_edits for row in self.utterances) / sum(
            row.phone_reference_count for row in self.utterances
        )

    @property
    def mora_error_rate(self) -> float:
        return sum(row.mora_edits for row in self.utterances) / sum(
            row.mora_reference_count for row in self.utterances
        )

    @property
    def total_latency_ms(self) -> float:
        return sum(row.latency_ms for row in self.utterances)

    @property
    def maximum_python_peak_bytes(self) -> int:
        return max(row.python_peak_bytes for row in self.utterances)

    @property
    def digest(self) -> str:
        return sha256_json(self.as_dict(include_digest=False, include_predictions=False))

    def as_dict(
        self,
        *,
        include_digest: bool = True,
        include_predictions: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "schemaVersion": self.schema_version,
            "runtimeProfileDigest": self.runtime_profile_digest,
            "manifestDigest": self.manifest_digest,
            "split": self.split,
            "phoneErrorRate": self.phone_error_rate,
            "moraErrorRate": self.mora_error_rate,
            "totalLatencyMs": self.total_latency_ms,
            "maximumPythonPeakBytes": self.maximum_python_peak_bytes,
            "utterances": [
                {
                    "utteranceId": row.utterance_id,
                    "sourceAudioSha256": row.source_audio_sha256,
                    "phoneReferenceCount": row.phone_reference_count,
                    "phoneEdits": row.phone_edits,
                    "moraReferenceCount": row.mora_reference_count,
                    "moraEdits": row.mora_edits,
                    "phonePredictionSha256": sha256_json({"symbols": row.phone_prediction}),
                    "moraPredictionSha256": sha256_json({"symbols": row.mora_prediction}),
                    "phonePosteriorDigest": row.phone_posterior_digest,
                    "moraPosteriorDigest": row.mora_posterior_digest,
                    "latencyMs": row.latency_ms,
                    "pythonPeakBytes": row.python_peak_bytes,
                    **(
                        {
                            "phonePrediction": row.phone_prediction,
                            "moraPrediction": row.mora_prediction,
                        }
                        if include_predictions
                        else {}
                    ),
                    "digest": row.digest,
                }
                for row in self.utterances
            ],
        }
        if include_digest:
            payload["reportDigest"] = self.digest
        return payload

    def write(
        self,
        path: str | Path,
        *,
        include_predictions: bool = False,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    self.as_dict(include_predictions=include_predictions),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        finally:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()
        return destination


def evaluate_phonetic_runtime(
    runtime: DualCTCPosteriorRuntime,
    manifest: PhoneticSplitManifest,
    *,
    split: SplitName = "test",
) -> PhoneticEvaluationReport:
    rows = manifest.rows_for(split)
    if not rows:
        raise ValueError(f"phonetic evaluation split is empty: {split}")
    output: list[PhoneticUtteranceEvaluation] = []
    for row in rows:
        tracemalloc.start()
        started = time.perf_counter_ns()
        try:
            phone, mora = runtime.infer(
                row.audio_path,
                expected_source_audio_sha256=row.source_audio_sha256,
            )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        latency = (time.perf_counter_ns() - started) / 1_000_000.0
        phone_prediction = greedy_posterior_symbols(phone)
        mora_prediction = greedy_posterior_symbols(mora)
        output.append(
            PhoneticUtteranceEvaluation(
                utterance_id=row.utterance_id,
                source_audio_sha256=row.source_audio_sha256,
                phone_reference_count=len(row.phone_symbols),
                phone_edits=_edit_distance(row.phone_symbols, phone_prediction),
                mora_reference_count=len(row.mora_symbols),
                mora_edits=_edit_distance(row.mora_symbols, mora_prediction),
                phone_prediction=phone_prediction,
                mora_prediction=mora_prediction,
                phone_posterior_digest=phone.digest,
                mora_posterior_digest=mora.digest,
                latency_ms=latency,
                python_peak_bytes=peak,
            )
        )
    return PhoneticEvaluationReport(
        runtime_profile_digest=runtime.profile_digest,
        manifest_digest=manifest.digest,
        split=split,
        utterances=tuple(output),
    )
