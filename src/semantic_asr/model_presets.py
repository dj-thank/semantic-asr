"""Auditable model-role registry for the Semantic ASR cascade.

The registry records intended roles and governance state. It never downloads a
model automatically; an immutable revision, rights approval, and held-out
validation are required before a preset can become a runtime default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelRole = Literal[
    "ngram",
    "cpu-reranker",
    "quality-reranker",
    "causal-scorer",
    "audio-second-ear",
    "offline-teacher",
]
RightsStatus = Literal["review", "approved", "denied"]
BenchmarkStatus = Literal["unverified", "smoke", "heldout-validated"]


@dataclass(frozen=True, slots=True)
class ModelPreset:
    name: str
    model_id: str
    role: ModelRole
    parameter_scale: str
    backend: str
    languages: tuple[str, ...]
    license_note: str
    revision: str | None = None
    rights_status: RightsStatus = "review"
    benchmark_status: BenchmarkStatus = "unverified"
    notes: str = ""

    def __post_init__(self) -> None:
        required = (
            self.name,
            self.model_id,
            self.parameter_scale,
            self.backend,
            self.license_note,
        )
        if any(not value.strip() for value in required):
            raise ValueError("model preset identity fields are required")
        if not self.languages:
            raise ValueError("model preset languages are required")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must be None or a non-empty immutable revision")

    @property
    def ready_for_runtime_default(self) -> bool:
        return (
            self.revision is not None
            and self.rights_status == "approved"
            and self.benchmark_status == "heldout-validated"
        )


PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        name="kenlm-char-mora",
        model_id="local:kenlm",
        role="ngram",
        parameter_scale="corpus-dependent",
        backend="kenlm",
        languages=("ja",),
        license_note="Depends on the approved training corpus and lexical sources.",
        notes="Ultra-light character, mora, and subword baseline.",
    ),
    ModelPreset(
        name="modernbert-ja-130m",
        model_id="sbintuitions/modernbert-ja-130m",
        role="cpu-reranker",
        parameter_scale="132.5M",
        backend="transformers-sequence-classification",
        languages=("ja", "en"),
        license_note="MIT model card at the pinned revision.",
        notes="Pairwise/listwise Japanese candidate encoder tier.",
    ),
    ModelPreset(
        name="lfm2-350m",
        model_id="LiquidAI/LFM2-350M",
        role="causal-scorer",
        parameter_scale="354.5M",
        backend="transformers-or-llama.cpp",
        languages=("ja", "en", "zh", "ko", "ar", "fr", "de", "es"),
        license_note="LFM model license; deployment terms require explicit review.",
        notes="Edge full-sequence likelihood and compact distillation student.",
    ),
    ModelPreset(
        name="lfm2-700m",
        model_id="LiquidAI/LFM2-700M",
        role="causal-scorer",
        parameter_scale="742.5M",
        backend="transformers-or-llama.cpp",
        languages=("ja", "en", "zh", "ko", "ar", "fr", "de", "es"),
        license_note="LFM model license; deployment terms require explicit review.",
        notes="Higher-quality edge sequence scorer candidate.",
    ),
    ModelPreset(
        name="qwen3-reranker-0.6b",
        model_id="Qwen/Qwen3-Reranker-0.6B",
        role="quality-reranker",
        parameter_scale="595.8M",
        backend="transformers-reranker",
        languages=("ja", "multilingual"),
        license_note="Apache-2.0 model card at the pinned revision.",
        notes="Instruction-aware quality tier; calibrate guarded logits.",
    ),
    ModelPreset(
        name="lfm2.5-1.2b-jp",
        model_id="LiquidAI/LFM2.5-1.2B-JP",
        role="causal-scorer",
        parameter_scale="1.17B",
        backend="transformers-or-llama.cpp",
        languages=("ja", "en"),
        license_note="LFM model license; deployment terms require explicit review.",
        notes="Japanese-specialized quality scorer or offline teacher.",
    ),
    ModelPreset(
        name="qwen3-asr-0.6b",
        model_id="Qwen/Qwen3-ASR-0.6B-hf",
        role="audio-second-ear",
        parameter_scale="0.6B",
        backend="transformers-qwen-asr",
        languages=("ja", "multilingual"),
        license_note="Apache-2.0 repository and pinned model revision.",
        notes="Ambiguity-only independent audio evidence, not decoder N-best.",
    ),
    ModelPreset(
        name="lfm2.5-audio-1.5b-jp",
        model_id="LiquidAI/LFM2.5-Audio-1.5B-JP",
        role="audio-second-ear",
        parameter_scale="1.47B",
        backend="liquid-audio",
        languages=("ja",),
        license_note="LFM and component licenses require explicit review.",
        notes="Continuous-audio verifier/second-ear research tier.",
    ),
)


def preset_by_name(name: str) -> ModelPreset:
    for preset in PRESETS:
        if preset.name == name:
            return preset
    raise KeyError(name)


def recommended_edge_cascade() -> tuple[ModelPreset, ...]:
    names = (
        "kenlm-char-mora",
        "modernbert-ja-130m",
        "lfm2-350m",
        "qwen3-reranker-0.6b",
        "qwen3-asr-0.6b",
    )
    return tuple(preset_by_name(name) for name in names)
