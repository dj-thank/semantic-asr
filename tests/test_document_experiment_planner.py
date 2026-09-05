from __future__ import annotations

import pytest

from semantic_asr.document_deliberation import DocumentBeamConfig
from semantic_asr.document_experiment.planner import FrozenDocumentBeamPlanner


def test_planner_factory_disables_arm_specific_linguistic_work() -> None:
    planner = FrozenDocumentBeamPlanner.create(
        config=DocumentBeamConfig(
            require_sequence_scorer=True,
            global_context_weight=1.0,
            proposal_context_arm="bidirectional-offline",
        )
    )

    assert not planner.config.require_sequence_scorer
    assert planner.config.global_context_weight == 0.0
    assert planner.config.proposal_context_arm == "none"
    assert planner.digest


def test_direct_planner_construction_rejects_unfair_config() -> None:
    with pytest.raises(ValueError, match="disable sequence scoring"):
        FrozenDocumentBeamPlanner(
            config=DocumentBeamConfig(require_sequence_scorer=True),
            build_config=None,  # type: ignore[arg-type]
            local_policy=None,  # type: ignore[arg-type]
        )
