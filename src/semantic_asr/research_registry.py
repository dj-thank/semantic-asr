from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class SourceStatus(StrEnum):
    PINNED_PRIMARY = "pinned-primary"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"


class TranslationKind(StrEnum):
    DIRECT_IMPLEMENTATION = "direct-implementation"
    ARCHITECTURE_ANALOGY = "architecture-analogy"
    EXPERIMENTAL_HYPOTHESIS = "experimental-hypothesis"


@dataclass(frozen=True, slots=True)
class ResearchSource:
    source_id: str
    title: str
    status: SourceStatus
    primary_url: str | None = None
    revision: str | None = None
    publication: str | None = None
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.source_id or not self.title:
            raise ValueError("source_id and title are required")
        if self.status == SourceStatus.PINNED_PRIMARY and not self.primary_url:
            raise ValueError("pinned primary sources require a URL")
        if self.status == SourceStatus.PROVISIONAL and self.revision:
            raise ValueError("provisional sources cannot claim a pinned revision")


@dataclass(frozen=True, slots=True)
class ArchitectureTranslation:
    translation_id: str
    source_ids: tuple[str, ...]
    source_mechanism: str
    semantic_asr_mechanism: str
    kind: TranslationKind
    claim_boundary: str
    falsification_test: str
    implementation_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.translation_id or not self.source_ids:
            raise ValueError("translation ID and source IDs are required")
        if not all(
            (
                self.source_mechanism,
                self.semantic_asr_mechanism,
                self.claim_boundary,
                self.falsification_test,
            )
        ):
            raise ValueError("translation descriptions must not be empty")


@dataclass(frozen=True, slots=True)
class ResearchRegistry:
    sources: tuple[ResearchSource, ...]
    translations: tuple[ArchitectureTranslation, ...]
    version: str = "2026-08-31-v0.2"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_ids = {source.source_id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("research source IDs must be unique")
        if len({row.translation_id for row in self.translations}) != len(self.translations):
            raise ValueError("translation IDs must be unique")
        for translation in self.translations:
            missing = set(translation.source_ids) - source_ids
            if missing:
                raise ValueError(
                    f"translation {translation.translation_id} references missing sources: {missing}"
                )
            provisional = [
                source.source_id
                for source in self.sources
                if source.source_id in translation.source_ids
                and source.status != SourceStatus.PINNED_PRIMARY
            ]
            if provisional:
                raise ValueError(
                    "unverified or rejected sources cannot justify implementation: "
                    + ", ".join(provisional)
                )

    @property
    def digest(self) -> str:
        payload = {
            "version": self.version,
            "sources": [
                {
                    **asdict(source),
                    "status": source.status.value,
                }
                for source in self.sources
            ],
            "translations": [
                {
                    **asdict(translation),
                    "kind": translation.kind.value,
                }
                for translation in self.translations
            ],
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def source(self, source_id: str) -> ResearchSource:
        try:
            return next(source for source in self.sources if source.source_id == source_id)
        except StopIteration as exc:
            raise KeyError(source_id) from exc

    def provisional_sources(self) -> tuple[ResearchSource, ...]:
        return tuple(
            source for source in self.sources if source.status == SourceStatus.PROVISIONAL
        )


def default_research_registry() -> ResearchRegistry:
    sources = (
        ResearchSource(
            source_id="qwen3.8-flash-next",
            title="Qwen3.8-Flash-Next technical report and official repository",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/QwenLM/Qwen3.8-Flash-Next",
            revision="69885871a64393807d988b27b1b5e380e8f28526",
            aliases=("Qwen 3.8 Flash Next", "QA 3.8 Flash Next"),
            notes="Used only through documented architecture translations; no kernels or weights are copied.",
        ),
        ResearchSource(
            source_id="qwen3-asr",
            title="Qwen3-ASR official repository and paper",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/QwenLM/Qwen3-ASR",
            publication="arXiv:2601.21337",
            notes="Independent second-ear ASR and forced-alignment integration.",
        ),
        ResearchSource(
            source_id="faster-whisper",
            title="faster-whisper",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/SYSTRAN/faster-whisper",
            notes="Exact runtime revision is pinned in every benchmark manifest.",
        ),
        ResearchSource(
            source_id="ctranslate2",
            title="CTranslate2",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/OpenNMT/CTranslate2",
            notes="Path scores, N-best generation and quantized CPU/GPU inference.",
        ),
        ResearchSource(
            source_id="whisper-lm",
            title="Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/hitz-zentroa/whisper-lm",
            publication="arXiv:2503.23542",
            notes="N-gram shallow fusion is a baseline; its example neural scorer is not treated as proper sequence likelihood.",
        ),
        ResearchSource(
            source_id="adaptive-ger",
            title="Confident and Adaptive Generative Speech Recognition via Risk Control",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/amitdamritau/adaptive-ger",
            publication="ICLR 2026",
            notes="Finite-sample policy selection and adaptive hypothesis set size.",
        ),
        ResearchSource(
            source_id="mbr-for-asr",
            title="Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/CyberAgentAILab/mbr-for-asr",
            publication="TMLR 2026",
            notes="Existing-candidate MBR is a mandatory non-LLM baseline.",
        ),
        ResearchSource(
            source_id="progres",
            title="ProGRes: Prompted Generative Rescoring on ASR N-best",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/AdaDTur/ProGRes",
            publication="IEEE SLT 2024",
        ),
        ResearchSource(
            source_id="llm-jp-asr",
            title="Whisper encoder with llm-jp decoder sample implementation",
            status=SourceStatus.PINNED_PRIMARY,
            primary_url="https://github.com/tosiyuki/llm-jp-asr",
            notes="Reserved for a later audio-projector experiment after second-pass upper bounds are known.",
        ),
        ResearchSource(
            source_id="glm-5.3",
            title="GLM 5.3 (user-provided model name)",
            status=SourceStatus.PROVISIONAL,
            aliases=("GLM5.3", "GLM 5.3"),
            notes="No architecture claim or implementation attribution is allowed until an official paper/repository and revision are pinned.",
        ),
        ResearchSource(
            source_id="kimi-k3",
            title="Kimi K3 (user-provided model name)",
            status=SourceStatus.PROVISIONAL,
            aliases=("KimiK3", "Kimi K3"),
            notes="No architecture claim or implementation attribution is allowed until an official paper/repository and revision are pinned.",
        ),
    )
    translations = (
        ArchitectureTranslation(
            translation_id="selected-acoustic-memory",
            source_ids=("qwen3.8-flash-next",),
            source_mechanism="Sparse selection of high-value local blocks.",
            semantic_asr_mechanism="Route only contradiction islands to re-listening, second-ear ASR or verification.",
            kind=TranslationKind.ARCHITECTURE_ANALOGY,
            claim_boundary="Scheduling analogy only; no QSA kernel is reproduced.",
            falsification_test="Compare quality, invocation rate and RTF against always-on second-pass inference.",
            implementation_paths=("src/semantic_asr/cascade.py", "src/semantic_asr/planner_v2.py"),
        ),
        ArchitectureTranslation(
            translation_id="constrained-gated-evidence",
            source_ids=("qwen3.8-flash-next",),
            source_mechanism="Dynamically gated branches and residual information flow.",
            semantic_asr_mechanism="Sparse specialist routing with an explicit minimum acoustic contribution.",
            kind=TranslationKind.ARCHITECTURE_ANALOGY,
            claim_boundary="Decision-fusion translation, not a transformer residual reproduction.",
            falsification_test="Ablate sparse experts and acoustic floor on held-out risk/quality/cost metrics.",
            implementation_paths=("src/semantic_asr/training_v2.py",),
        ),
        ArchitectureTranslation(
            translation_id="ngram-local-memory",
            source_ids=("qwen3.8-flash-next", "whisper-lm"),
            source_mechanism="Local n-gram/context augmentation and external LM fusion.",
            semantic_asr_mechanism="Character, mora and subword n-gram scorers with typed provenance and calibration.",
            kind=TranslationKind.DIRECT_IMPLEMENTATION,
            claim_boundary="Uses independently trained language models; no external model table is copied.",
            falsification_test="Compare each n-gram granularity with ASR-only and compact-reranker baselines.",
            implementation_paths=("src/semantic_asr/ngram.py",),
        ),
        ArchitectureTranslation(
            translation_id="adaptive-risk-controlled-k",
            source_ids=("adaptive-ger",),
            source_mechanism="Learn-Then-Test selection of an adaptive hypothesis-set policy.",
            semantic_asr_mechanism="Finite-sample upper-risk filtering over candidate count and stage policies.",
            kind=TranslationKind.DIRECT_IMPLEMENTATION,
            claim_boundary="The initial bound is conservative Hoeffding-Bonferroni, not a claim of reproducing every paper result.",
            falsification_test="Compare fixed K, heuristic adaptive K and risk-controlled K on locked calibration/test splits.",
            implementation_paths=("src/semantic_asr/risk_control.py",),
        ),
        ArchitectureTranslation(
            translation_id="safe-existing-candidate-mbr",
            source_ids=("mbr-for-asr",),
            source_mechanism="Minimum Bayes Risk candidate selection.",
            semantic_asr_mechanism="Character/mora/critical-semantic expected loss over existing surface candidates.",
            kind=TranslationKind.DIRECT_IMPLEMENTATION,
            claim_boundary="Consensus text outside the candidate pool remains an unverified proposal.",
            falsification_test="Report MBR against maximum posterior, oracle K and learned reranker baselines.",
            implementation_paths=("src/semantic_asr/mbr.py",),
        ),
        ArchitectureTranslation(
            translation_id="path-mass-preservation",
            source_ids=("ctranslate2", "faster-whisper"),
            source_mechanism="Beam paths with length-normalized decoder scores.",
            semantic_asr_mechanism="Recover cumulative path scores and aggregate duplicate surfaces with logsumexp.",
            kind=TranslationKind.DIRECT_IMPLEMENTATION,
            claim_boundary="Relies on the pinned CTranslate2 score definition for each benchmark runtime.",
            falsification_test="Compare best-path deduplication with path-mass aggregation and calibration.",
            implementation_paths=("src/semantic_asr/adapters_v2.py", "src/semantic_asr/candidate_pool.py"),
        ),
    )
    return ResearchRegistry(
        sources=sources,
        translations=translations,
        metadata={
            "policy": "Only revision-pinned primary sources can justify named architecture translations.",
        },
    )
