from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .adapters import ASRAdapter, DecodeRequest
from .candidate_pool import aggregate_surface_candidates
from .contracts import CandidateEvidence, canonical_json

DatasetSplit = Literal["train", "calibration", "test"]
RightsDecision = Literal["allow", "deny", "review"]


@dataclass(frozen=True, slots=True)
class AudioManifestRecord:
    sample_id: str
    group_id: str
    source_id: str
    split: DatasetSplit
    audio_path: str
    reference: str
    domain: str = "unknown"
    near_duplicate_id: str | None = None
    rights_decision: RightsDecision = "review"
    license_id: str | None = None
    dataset_name: str | None = None
    dataset_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id or not self.source_id:
            raise ValueError("sample, group, and source IDs are required")
        if self.split not in {"train", "calibration", "test"}:
            raise ValueError("unknown audio-manifest split")
        if not self.audio_path or not self.reference:
            raise ValueError("audio path and reference are required")
        if self.rights_decision not in {"allow", "deny", "review"}:
            raise ValueError("unknown rights decision")
        if bool(self.dataset_name) != bool(self.dataset_revision):
            raise ValueError("dataset name and revision must be provided together")


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    language: str | None = "ja"
    beam_size: int = 12
    hypotheses: int = 12
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()
    return_timestamps: bool = False
    fail_on_non_allow_rights: bool = True
    model_revision: str | None = None
    model_artifact_sha256: str | None = None
    runtime_revision: str | None = None

    def __post_init__(self) -> None:
        if self.beam_size < 1 or self.hypotheses < 1:
            raise ValueError("beam size and hypothesis count must be positive")
        if self.hypotheses > self.beam_size:
            raise ValueError("hypothesis count cannot exceed beam size")
        if self.model_revision is not None and self.model_artifact_sha256 is not None:
            raise ValueError("model revision and local artifact digest are mutually exclusive")
        if self.model_artifact_sha256 is not None and (
            len(self.model_artifact_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF"
                for character in self.model_artifact_sha256
            )
        ):
            raise ValueError("model artifact digest must be SHA-256 hex")

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GeneratedCandidateRecord:
    sample_id: str
    group_id: str
    source_id: str
    split: DatasetSplit
    reference: str
    domain: str
    near_duplicate_id: str | None
    audio_sha256: str
    candidates: tuple[CandidateEvidence, ...]
    adapter: str
    model: str
    generation_config_sha256: str
    elapsed_ms: int
    rights_decision: RightsDecision
    license_id: str | None
    dataset_name: str | None
    dataset_revision: str | None

    def __post_init__(self) -> None:
        if len(self.audio_sha256) != 64:
            raise ValueError("audio digest must be SHA-256 hex")
        if len(self.generation_config_sha256) != 64:
            raise ValueError("generation configuration digest must be SHA-256 hex")
        if not self.candidates:
            raise ValueError("generated record requires candidates")
        if self.elapsed_ms < 0:
            raise ValueError("generation elapsed time must be non-negative")
        if self.rights_decision not in {"allow", "deny", "review"}:
            raise ValueError("unknown generated-record rights decision")

    def as_benchmark_row(self) -> dict[str, Any]:
        return {
            "sampleId": self.sample_id,
            "groupId": self.group_id,
            "sourceId": self.source_id,
            "split": self.split,
            "reference": self.reference,
            "domain": self.domain,
            "nearDuplicateId": self.near_duplicate_id,
            "audioSha256": self.audio_sha256,
            "datasetName": self.dataset_name,
            "datasetRevision": self.dataset_revision,
            "rightsDecision": self.rights_decision,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "generation": {
                "adapter": self.adapter,
                "model": self.model,
                "modelRevision": self.candidates[0].metadata.get("modelRevision"),
                "modelArtifactSha256": self.candidates[0].metadata.get("modelArtifactSha256"),
                "runtimeRevision": self.candidates[0].metadata.get("runtimeRevision"),
                "configSha256": self.generation_config_sha256,
                "elapsedMs": self.elapsed_ms,
                "licenseId": self.license_id,
            },
        }

    def as_ranker_row(self) -> dict[str, Any]:
        return {
            "exampleId": self.sample_id,
            "groupId": self.group_id,
            "split": self.split,
            "reference": self.reference,
            "context": "",
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "audioSha256": self.audio_sha256,
            "datasetName": self.dataset_name,
            "datasetRevision": self.dataset_revision,
            "rightsDecision": self.rights_decision,
            "sourceId": self.source_id,
            "domain": self.domain,
            "nearDuplicateId": self.near_duplicate_id,
            "generation": {
                "adapter": self.adapter,
                "model": self.model,
                "configSha256": self.generation_config_sha256,
                "elapsedMs": self.elapsed_ms,
                "licenseId": self.license_id,
            },
        }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_isolation(records: Sequence[AudioManifestRecord]) -> None:
    if not records:
        raise ValueError("audio manifest is empty")
    seen_samples: set[str] = set()
    dimensions: dict[str, dict[str, set[str]]] = {
        "group": {},
        "source": {},
        "near-duplicate": {},
        "reference": {},
    }
    for record in records:
        if record.sample_id in seen_samples:
            raise ValueError(f"duplicate sample ID: {record.sample_id}")
        seen_samples.add(record.sample_id)
        for kind, identifier in (
            ("group", record.group_id),
            ("source", record.source_id),
            ("near-duplicate", record.near_duplicate_id),
            (
                "reference",
                hashlib.sha256(
                    "".join(
                        character
                        for character in unicodedata.normalize("NFKC", record.reference)
                        if not character.isspace()
                    ).encode("utf-8")
                ).hexdigest(),
            ),
        ):
            if not identifier:
                continue
            dimensions[kind].setdefault(identifier, set()).add(record.split)
    for kind, values in dimensions.items():
        leaking = {
            identifier: sorted(splits) for identifier, splits in values.items() if len(splits) > 1
        }
        if leaking:
            identifier = sorted(leaking)[0]
            raise ValueError(f"{kind} leakage across splits: {identifier} -> {leaking[identifier]}")


def _bind_generation_config(
    adapter: ASRAdapter,
    config: CandidateGenerationConfig | None,
) -> CandidateGenerationConfig:
    bound = config or CandidateGenerationConfig()
    adapter_revision = getattr(adapter, "model_revision", None)
    adapter_artifact = getattr(adapter, "model_artifact_sha256", None)
    adapter_runtime = getattr(adapter, "runtime_revision", None)
    if bound.model_revision is not None and bound.model_revision != adapter_revision:
        raise ValueError(
            "candidate-generation model revision does not match the loaded adapter revision"
        )
    if bound.model_revision is None and adapter_revision is not None:
        bound = replace(bound, model_revision=str(adapter_revision))
    if bound.model_artifact_sha256 is not None and bound.model_artifact_sha256 != adapter_artifact:
        raise ValueError(
            "candidate-generation model artifact digest does not match the loaded adapter"
        )
    if bound.model_artifact_sha256 is None and adapter_artifact is not None:
        bound = replace(bound, model_artifact_sha256=str(adapter_artifact))
    if bound.model_revision is not None and bound.model_artifact_sha256 is not None:
        raise ValueError("loaded adapter cannot claim both Hub revision and local artifact digest")
    if bound.runtime_revision is not None and (
        adapter_runtime is not None and bound.runtime_revision != adapter_runtime
    ):
        raise ValueError("candidate-generation runtime revision does not match the loaded adapter")
    if bound.runtime_revision is None and adapter_runtime is not None:
        bound = replace(bound, runtime_revision=str(adapter_runtime))
    if hasattr(adapter, "runtime_revision") and bound.runtime_revision is None:
        raise ValueError("candidate-generation runtime revision is required")
    return bound


def generate_candidates(
    record: AudioManifestRecord,
    adapter: ASRAdapter,
    *,
    config: CandidateGenerationConfig | None = None,
) -> GeneratedCandidateRecord:
    config = _bind_generation_config(adapter, config)
    if record.rights_decision == "deny":
        raise PermissionError(f"sample {record.sample_id} rights decision is 'deny'")
    if config.fail_on_non_allow_rights and record.rights_decision != "allow":
        raise PermissionError(
            f"sample {record.sample_id} rights decision is {record.rights_decision!r}"
        )
    if not record.license_id:
        raise PermissionError(
            f"sample {record.sample_id} is missing license provenance for candidate generation"
        )
    source = Path(record.audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    audio_sha256 = sha256_file(source)
    started = time.perf_counter()
    candidates = adapter.decode(
        DecodeRequest(
            audio_path=str(source),
            language=config.language,
            beam_size=config.beam_size,
            hypotheses=config.hypotheses,
            initial_prompt=config.initial_prompt,
            hotwords=config.hotwords,
            return_timestamps=config.return_timestamps,
        )
    )
    pooled = aggregate_surface_candidates(candidates, id_prefix="experiment")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    model = str(getattr(adapter, "model_name", adapter.name))
    annotated: list[CandidateEvidence] = []
    for candidate in pooled:
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "experimentSampleId": record.sample_id,
                "audioSha256": audio_sha256,
                "generationConfigSha256": config.digest,
                "modelRevision": config.model_revision,
                "modelArtifactSha256": config.model_artifact_sha256,
                "runtimeRevision": config.runtime_revision,
            }
        )
        annotated.append(CandidateEvidence.from_dict({**candidate.as_dict(), "metadata": metadata}))
    return GeneratedCandidateRecord(
        sample_id=record.sample_id,
        group_id=record.group_id,
        source_id=record.source_id,
        split=record.split,
        reference=record.reference,
        domain=record.domain,
        near_duplicate_id=record.near_duplicate_id,
        audio_sha256=audio_sha256,
        candidates=tuple(annotated),
        adapter=adapter.name,
        model=model,
        generation_config_sha256=config.digest,
        elapsed_ms=elapsed_ms,
        rights_decision=record.rights_decision,
        license_id=record.license_id,
        dataset_name=record.dataset_name,
        dataset_revision=record.dataset_revision,
    )


def generate_manifest(
    records: Sequence[AudioManifestRecord],
    adapter: ASRAdapter,
    *,
    config: CandidateGenerationConfig | None = None,
) -> list[GeneratedCandidateRecord]:
    verify_manifest_isolation(records)
    return [generate_candidates(record, adapter, config=config) for record in records]


def _load_checkpoint_rows(
    path: Path, *, repair_unterminated_tail: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    offset = 0
    encoded_lines = payload.splitlines(keepends=True)
    for line_number, encoded_line in enumerate(encoded_lines, 1):
        next_offset = offset + len(encoded_line)
        if not encoded_line.strip():
            if (
                repair_unterminated_tail
                and line_number == len(encoded_lines)
                and not encoded_line.endswith((b"\n", b"\r"))
            ):
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                return rows
            offset = next_offset
            continue
        try:
            value = json.loads(encoded_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if not (
                repair_unterminated_tail
                and line_number == len(encoded_lines)
                and not encoded_line.endswith((b"\n", b"\r"))
            ):
                raise
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            return rows
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint row {line_number} must be an object")
        rows.append(value)
        offset = next_offset
    if repair_unterminated_tail and payload and not payload.endswith((b"\n", b"\r")):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return rows


@contextmanager
def _checkpoint_writer_lock(checkpoint: Path):
    """Hold an OS-released advisory lock for one checkpoint writer/finalizer."""

    lock_path = Path(str(checkpoint) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"checkpoint already has an active writer: {checkpoint}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _validate_checkpoint_prefix(
    records: Sequence[AudioManifestRecord],
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter: ASRAdapter,
    config: CandidateGenerationConfig,
) -> None:
    if len(rows) > len(records):
        raise ValueError("checkpoint contains more rows than the input manifest")
    for index, row in enumerate(rows):
        record = records[index]
        if row.get("sampleId") != record.sample_id:
            raise ValueError(f"checkpoint sample order mismatch at row {index + 1}")
        if row.get("groupId") != record.group_id:
            raise ValueError(f"checkpoint group mismatch at row {index + 1}")
        if row.get("sourceId") != record.source_id:
            raise ValueError(f"checkpoint source mismatch at row {index + 1}")
        if row.get("split") != record.split:
            raise ValueError(f"checkpoint split mismatch at row {index + 1}")
        if row.get("reference") != record.reference:
            raise ValueError(f"checkpoint reference mismatch at row {index + 1}")
        if row.get("domain") != record.domain:
            raise ValueError(f"checkpoint domain mismatch at row {index + 1}")
        if row.get("nearDuplicateId") != record.near_duplicate_id:
            raise ValueError(f"checkpoint near-duplicate mismatch at row {index + 1}")
        if row.get("rightsDecision") != record.rights_decision:
            raise ValueError(f"checkpoint rights mismatch at row {index + 1}")
        if row.get("datasetName") != record.dataset_name:
            raise ValueError(f"checkpoint dataset mismatch at row {index + 1}")
        if row.get("datasetRevision") != record.dataset_revision:
            raise ValueError(f"checkpoint dataset revision mismatch at row {index + 1}")
        generation = row.get("generation")
        if not isinstance(generation, Mapping) or generation.get("configSha256") != config.digest:
            raise ValueError(f"checkpoint configuration mismatch at row {index + 1}")
        if generation.get("adapter") != adapter.name:
            raise ValueError(f"checkpoint adapter mismatch at row {index + 1}")
        if generation.get("model") != str(getattr(adapter, "model_name", adapter.name)):
            raise ValueError(f"checkpoint model mismatch at row {index + 1}")
        if generation.get("licenseId") != record.license_id:
            raise ValueError(f"checkpoint license mismatch at row {index + 1}")
        if generation.get("modelRevision") != config.model_revision:
            raise ValueError(f"checkpoint generation model revision mismatch at row {index + 1}")
        if generation.get("modelArtifactSha256") != config.model_artifact_sha256:
            raise ValueError(f"checkpoint generation model artifact mismatch at row {index + 1}")
        if generation.get("runtimeRevision") != config.runtime_revision:
            raise ValueError(f"checkpoint generation runtime revision mismatch at row {index + 1}")
        source = Path(record.audio_path).expanduser().resolve()
        audio_sha256 = sha256_file(source)
        if row.get("audioSha256") != audio_sha256:
            raise ValueError(f"checkpoint audio digest mismatch at row {index + 1}")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"checkpoint candidates missing at row {index + 1}")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError(f"checkpoint candidate invalid at row {index + 1}")
            CandidateEvidence.from_dict(candidate)
            metadata = candidate.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"checkpoint candidate metadata missing at row {index + 1}")
            if metadata.get("experimentSampleId") != record.sample_id:
                raise ValueError(f"checkpoint candidate sample mismatch at row {index + 1}")
            if metadata.get("audioSha256") != audio_sha256:
                raise ValueError(f"checkpoint candidate audio mismatch at row {index + 1}")
            if metadata.get("generationConfigSha256") != config.digest:
                raise ValueError(f"checkpoint candidate configuration mismatch at row {index + 1}")
            if metadata.get("runtimeRevision") != config.runtime_revision:
                raise ValueError(f"checkpoint runtime revision mismatch at row {index + 1}")
            if metadata.get("modelArtifactSha256") != config.model_artifact_sha256:
                raise ValueError(f"checkpoint model artifact mismatch at row {index + 1}")
        if config.model_revision is not None and any(
            not isinstance(candidate, Mapping)
            or not isinstance(candidate.get("metadata"), Mapping)
            or candidate["metadata"].get("modelRevision") != config.model_revision
            for candidate in candidates
        ):
            raise ValueError(f"checkpoint model revision mismatch at row {index + 1}")


def generate_manifest_to_checkpoint(
    records: Sequence[AudioManifestRecord],
    adapter: ASRAdapter,
    *,
    checkpoint_path: str | Path,
    config: CandidateGenerationConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate one durable row at a time and resume only a verified manifest prefix."""

    verify_manifest_isolation(records)
    bound_config = _bind_generation_config(adapter, config)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with _checkpoint_writer_lock(checkpoint):
        rows = _load_checkpoint_rows(checkpoint, repair_unterminated_tail=True)
        _validate_checkpoint_prefix(records, rows, adapter=adapter, config=bound_config)
        if progress is not None and rows:
            progress(len(rows), len(records))
        with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records[len(rows) :]:
                generated = generate_candidates(record, adapter, config=bound_config)
                row = generated.as_benchmark_row()
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
                if progress is not None:
                    progress(len(rows), len(records))
    return rows


def _ranker_row_from_benchmark(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exampleId": row["sampleId"],
        "groupId": row["groupId"],
        "split": row["split"],
        "reference": row["reference"],
        "context": "",
        "candidates": row["candidates"],
        "audioSha256": row["audioSha256"],
        "datasetName": row.get("datasetName"),
        "datasetRevision": row.get("datasetRevision"),
        "rightsDecision": row.get("rightsDecision"),
        "sourceId": row.get("sourceId"),
        "domain": row.get("domain"),
        "nearDuplicateId": row.get("nearDuplicateId"),
        "generation": row.get("generation"),
    }


def finalize_generated_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path,
    records: Sequence[AudioManifestRecord],
    adapter: ASRAdapter,
    config: CandidateGenerationConfig | None = None,
    ranker_path: str | Path | None = None,
) -> None:
    """Promote a complete checkpoint last, after any derived ranker file is durable."""

    checkpoint = Path(checkpoint_path)
    with _checkpoint_writer_lock(checkpoint):
        rows = _load_checkpoint_rows(checkpoint)
        if not rows:
            raise ValueError("generated checkpoint is empty")
        if len(rows) != len(records):
            raise ValueError(
                f"generated checkpoint row count mismatch: expected {len(records)}, got {len(rows)}"
            )
        bound_config = _bind_generation_config(adapter, config)
        _validate_checkpoint_prefix(records, rows, adapter=adapter, config=bound_config)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if ranker_path is not None:
            ranker = Path(ranker_path)
            ranker.parent.mkdir(parents=True, exist_ok=True)
            ranker_partial = Path(str(ranker) + ".partial")
            with ranker_partial.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(
                        json.dumps(
                            _ranker_row_from_benchmark(row),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(ranker_partial, ranker)
        os.replace(checkpoint, output)


def audio_record_from_row(row: Mapping[str, Any], *, line_number: int = 0) -> AudioManifestRecord:
    return AudioManifestRecord(
        sample_id=str(row.get("sampleId") or row.get("sample_id") or line_number),
        group_id=str(row.get("groupId") or row.get("group_id") or ""),
        source_id=str(row.get("sourceId") or row.get("source_id") or ""),
        split=str(row.get("split") or "train"),
        audio_path=str(row.get("audioPath") or row.get("audio_path") or ""),
        reference=str(row.get("reference") or ""),
        domain=str(row.get("domain") or "unknown"),
        near_duplicate_id=(
            str(row.get("nearDuplicateId") or row.get("near_duplicate_id"))
            if row.get("nearDuplicateId") or row.get("near_duplicate_id")
            else None
        ),
        rights_decision=str(row.get("rightsDecision") or row.get("rights_decision") or "review"),
        license_id=(
            str(row.get("licenseId") or row.get("license_id"))
            if row.get("licenseId") or row.get("license_id")
            else None
        ),
        dataset_name=(
            str(row.get("datasetName") or row.get("dataset_name"))
            if row.get("datasetName") or row.get("dataset_name")
            else None
        ),
        dataset_revision=(
            str(row.get("datasetRevision") or row.get("dataset_revision"))
            if row.get("datasetRevision") or row.get("dataset_revision")
            else None
        ),
    )


def load_audio_manifest(path: str | Path) -> list[AudioManifestRecord]:
    output: list[AudioManifestRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"audio manifest row {line_number} must be an object")
        output.append(audio_record_from_row(payload, line_number=line_number))
    verify_manifest_isolation(output)
    return output


def write_generated_manifests(
    records: Iterable[GeneratedCandidateRecord],
    *,
    benchmark_path: str | Path,
    ranker_path: str | Path | None = None,
) -> None:
    rows = list(records)
    benchmark = Path(benchmark_path)
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_text(
        "\n".join(
            json.dumps(record.as_benchmark_row(), ensure_ascii=False, separators=(",", ":"))
            for record in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if ranker_path is not None:
        ranker = Path(ranker_path)
        ranker.parent.mkdir(parents=True, exist_ok=True)
        ranker.write_text(
            "\n".join(
                json.dumps(record.as_ranker_row(), ensure_ascii=False, separators=(",", ":"))
                for record in rows
            )
            + ("\n" if rows else ""),
            encoding="utf-8",
        )
