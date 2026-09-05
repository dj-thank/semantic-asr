"""Adapter that freezes one fair document-beam candidate set before arm scoring."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contracts import sha256_json
from ..document_deliberation import DocumentBeamConfig, plan_document_deliberation
from ..global_deliberation import DeliberationPolicy
from ..semantic_deliberation import SemanticDeliberationConfig
from .runner import PlanningCaseView


@dataclass(frozen=True, slots=True)
class FrozenDocumentBeamPlanner:
    """Run document candidate generation exactly once with linguistic rescoring disabled.

    The resulting alternatives are subsequently frozen by ``prepare_document_experiment`` and
    reused by every experiment arm. Proposal context is fixed here and cannot vary by arm.
    """

    config: DocumentBeamConfig
    build_config: SemanticDeliberationConfig
    local_policy: DeliberationPolicy
    proposal_provider: object | None = None
    proposal_context_name: str | None = None
    audio_paths: Mapping[str, str | Path] | None = None

    def __post_init__(self) -> None:
        if self.config.require_sequence_scorer:
            raise ValueError("experiment candidate planner must disable sequence scoring")
        if self.config.global_context_weight != 0.0:
            raise ValueError("experiment candidate planner must set global_context_weight=0")
        if self.config.proposal_context_arm != "none":
            raise ValueError("experiment candidate planner must freeze proposal_context_arm='none'")
        object.__setattr__(self, "audio_paths", dict(self.audio_paths or {}))

    @classmethod
    def create(
        cls,
        *,
        config: DocumentBeamConfig | None = None,
        build_config: SemanticDeliberationConfig | None = None,
        local_policy: DeliberationPolicy | None = None,
        proposal_provider: object | None = None,
        proposal_context_name: str | None = None,
        audio_paths: Mapping[str, str | Path] | None = None,
    ) -> FrozenDocumentBeamPlanner:
        base = config or DocumentBeamConfig()
        fair = replace(
            base,
            require_sequence_scorer=False,
            global_context_weight=0.0,
            proposal_context_arm="none",
        )
        return cls(
            config=fair,
            build_config=build_config or SemanticDeliberationConfig(),
            local_policy=local_policy or DeliberationPolicy.conservative_default(),
            proposal_provider=proposal_provider,
            proposal_context_name=proposal_context_name,
            audio_paths=audio_paths,
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "configDigest": self.config.digest,
                "buildConfigDigest": self.build_config.digest,
                "localPolicyDigest": self.local_policy.digest,
                "proposalProvider": (
                    None
                    if self.proposal_provider is None
                    else f"{type(self.proposal_provider).__module__}."
                    f"{type(self.proposal_provider).__qualname__}"
                ),
                "proposalContextName": self.proposal_context_name,
                "audioCaseIds": tuple(sorted((self.audio_paths or {}).keys())),
                "sequenceScorer": None,
            }
        )

    def __call__(self, view: PlanningCaseView) -> object:
        context = view.context(self.proposal_context_name)
        audio_path = (self.audio_paths or {}).get(view.case_id)
        supplied: dict[str, Any] = {
            "config": self.config,
            "build_config": self.build_config,
            "local_policy": self.local_policy,
            "sequence_scorer": None,
            "proposal_provider": self.proposal_provider,
            "declared_context": context,
            "audio_path": audio_path,
        }
        signature = inspect.signature(plan_document_deliberation)
        accepted = {
            name: value
            for name, value in supplied.items()
            if name in signature.parameters and value is not None
        }
        required_contract = {"config", "sequence_scorer"}
        if not required_contract.issubset(signature.parameters):
            raise TypeError("document planner no longer exposes the frozen-candidate contract")
        plan = plan_document_deliberation(view.first_pass, **accepted)
        planner_digest = getattr(plan, "planner_digest", None)
        if planner_digest is not None and planner_digest != self.digest:
            raise ValueError("document plan reports a different planner identity")
        return plan
