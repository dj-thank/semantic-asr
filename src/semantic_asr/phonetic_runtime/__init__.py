"""Source-audio-only dual phone/mora CTC research runtime."""

from .artifact import (
    LoadedDualCTCArtifact,
    load_dual_ctc_artifact,
    metadata_runtime_digest,
    read_dual_ctc_metadata,
    save_dual_ctc_artifact,
)
from .audio import LoadedWaveform, load_pcm16_wav, sha256_file
from .calibration import (
    CTCUtilityCalibrationReport,
    PhoneticCalibrationCandidate,
    PhoneticCalibrationExample,
    fit_ctc_utility_calibration,
)
from .contracts import (
    DualCTCArtifactMetadata,
    DualCTCModelConfig,
    LogMelFrontendConfig,
    PhoneticInventory,
    PhoneticRuntimeLimits,
    TensorSpecification,
)
from .evaluation import (
    PhoneticEvaluationReport,
    PhoneticUtteranceEvaluation,
    evaluate_phonetic_runtime,
    greedy_posterior_symbols,
)
from .inference import DualCTCPosteriorRuntime
from .manifest import (
    PhoneticManifestRow,
    PhoneticSplitManifest,
    load_phonetic_manifest,
    validate_split_isolation,
)
from .provider import (
    PhoneMoraPosteriorRuntime,
    PhoneticProposalProviderConfig,
    SourceAudioPhoneticProposalProvider,
    SpanLexiconProvider,
)
from .training import (
    DualCTCTrainingConfig,
    DualCTCTrainingResult,
    TrainingEpochMetrics,
    train_dual_ctc_model,
)

__all__ = [
    "CTCUtilityCalibrationReport",
    "DualCTCArtifactMetadata",
    "DualCTCModelConfig",
    "DualCTCPosteriorRuntime",
    "DualCTCTrainingConfig",
    "DualCTCTrainingResult",
    "LoadedDualCTCArtifact",
    "LoadedWaveform",
    "LogMelFrontendConfig",
    "PhoneMoraPosteriorRuntime",
    "PhoneticCalibrationCandidate",
    "PhoneticCalibrationExample",
    "PhoneticEvaluationReport",
    "PhoneticInventory",
    "PhoneticManifestRow",
    "PhoneticProposalProviderConfig",
    "PhoneticRuntimeLimits",
    "PhoneticSplitManifest",
    "PhoneticUtteranceEvaluation",
    "SourceAudioPhoneticProposalProvider",
    "SpanLexiconProvider",
    "TensorSpecification",
    "TrainingEpochMetrics",
    "evaluate_phonetic_runtime",
    "fit_ctc_utility_calibration",
    "greedy_posterior_symbols",
    "load_dual_ctc_artifact",
    "load_pcm16_wav",
    "load_phonetic_manifest",
    "metadata_runtime_digest",
    "read_dual_ctc_metadata",
    "save_dual_ctc_artifact",
    "sha256_file",
    "train_dual_ctc_model",
    "validate_split_isolation",
]
