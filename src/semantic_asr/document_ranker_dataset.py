"""Offline-only labeling of document-lattice alternatives for ranker training.

References are accepted only by this module and are never part of a runtime scorer interface. The
builder converts already-generated ``DocumentPathCandidate`` alternatives into the exact
``DocumentRankInput`` representation used at inference, then attaches CER and critical-token labels
for pairwise training.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import sha256_json
from .deliberation_lattice import DocumentContext
from .document_deliberation_benchmark import character_error_rate
from .document_joint_deliberation import DocumentDeliberationDecision, DocumentPathCandidate
from .document_ranker import DocumentRankExample, DocumentRankInput


def _critical_error_count(reference: str, hypothesis: str, tokens: Sequence[str]) -> int:
    return sum(reference.count(token) != hypothesis.count(token) for token in tokens)


def rank_input_from_candidate(
    candidate: DocumentPathCandidate,
    *,
    context: DocumentContext,
    retained_selection_digest: str,
) -> DocumentRankInput:
    return DocumentRankInput(
        text=candidate.text,
        left_context=context.left_context,
        right_context=context.right_context,
        topic_summary=context.topic_summary,
        entity_ids=context.entity_ids,
        local_score=candidate.local_score,
        overlap_score=candidate.overlap_score,
        mean_audio_support=candidate.mean_audio_support,
        changed_window_count=len(candidate.changed_window_indexes),
        generated_window_count=len(candidate.generated_window_indexes),
        ambiguous_overlap_count=len(candidate.ambiguous_overlap_indexes),
        window_count=len(candidate.options),
        retained_path=candidate.selection_digest == retained_selection_digest,
        metadata={
            "selectionDigest": candidate.selection_digest,
            "candidateDigest": candidate.digest,
            "optionDigests": tuple(option.digest for option in candidate.options),
            "overlapReceiptDigests": tuple(
                receipt.digest for receipt in candidate.overlap_receipts
            ),
            "contextDigest": context.digest,
        },
    )


@dataclass(frozen=True, slots=True)
class DocumentRankerLabeledGroup:
    group_id: str
    reference_sha256: str
    first_pass_text_sha256: str
    context_digest: str
    critical_tokens: tuple[str, ...]
    examples: tuple[DocumentRankExample, ...]
    decision_digest: str
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if not self.group_id or not self.examples:
            raise ValueError("labeled document group requires group_id and examples")
        if len({example.candidate_id for example in self.examples}) != len(self.examples):
            raise ValueError("labeled document candidate IDs must be unique")
        for digest in (
            self.reference_sha256,
            self.first_pass_text_sha256,
            self.context_digest,
            self.decision_digest,
        ):
            if len(digest) != 64:
                raise ValueError("labeled document group contains an invalid digest")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schemaVersion": self.schema_version,
                "groupId": self.group_id,
                "referenceSha256": self.reference_sha256,
                "firstPassTextSha256": self.first_pass_text_sha256,
                "contextDigest": self.context_digest,
                "criticalTokens": self.critical_tokens,
                "exampleDigests": [example.digest for example in self.examples],
                "decisionDigest": self.decision_digest,
            }
        )


def label_document_decision(
    decision: DocumentDeliberationDecision,
    *,
    group_id: str,
    reference: str,
    first_pass_text: str,
    context: DocumentContext,
    critical_tokens: Sequence[str] = (),
) -> DocumentRankerLabeledGroup:
    if not group_id or not reference:
        raise ValueError("group_id and reference are required")
    if decision.context_digest != context.digest:
        raise ValueError("decision and labeling context digests differ")
    retained_digest = decision.retained.selection_digest
    first_pass_exact = first_pass_text == reference
    examples = []
    for candidate in decision.alternatives:
        rank_input = rank_input_from_candidate(
            candidate,
            context=context,
            retained_selection_digest=retained_digest,
        )
        examples.append(
            DocumentRankExample(
                group_id=group_id,
                candidate_id=candidate.selection_digest,
                rank_input=rank_input,
                character_error_rate=character_error_rate(reference, candidate.text),
                critical_error_count=_critical_error_count(
                    reference,
                    candidate.text,
                    critical_tokens,
                ),
                first_pass_exact=first_pass_exact,
                metadata={
                    "decisionDigest": decision.digest,
                    "documentCandidateDigest": candidate.digest,
                    "referenceUse": "offline-label-only",
                },
            )
        )
    return DocumentRankerLabeledGroup(
        group_id=group_id,
        reference_sha256=sha256_json({"reference": reference}),
        first_pass_text_sha256=sha256_json({"firstPassText": first_pass_text}),
        context_digest=context.digest,
        critical_tokens=tuple(critical_tokens),
        examples=tuple(examples),
        decision_digest=decision.digest,
    )


def example_to_json(example: DocumentRankExample) -> dict[str, object]:
    value = example.rank_input
    return {
        "groupId": example.group_id,
        "candidateId": example.candidate_id,
        "text": value.text,
        "leftContext": value.left_context,
        "rightContext": value.right_context,
        "topicSummary": value.topic_summary,
        "entityIds": value.entity_ids,
        "localScore": value.local_score,
        "overlapScore": value.overlap_score,
        "meanAudioSupport": value.mean_audio_support,
        "changedWindowCount": value.changed_window_count,
        "generatedWindowCount": value.generated_window_count,
        "ambiguousOverlapCount": value.ambiguous_overlap_count,
        "windowCount": value.window_count,
        "retainedPath": value.retained_path,
        "characterErrorRate": example.character_error_rate,
        "criticalErrorCount": example.critical_error_count,
        "firstPassExact": example.first_pass_exact,
        "metadata": value.metadata,
        "exampleMetadata": example.metadata,
        "exampleDigest": example.digest,
    }


def write_labeled_groups(
    groups: Sequence[DocumentRankerLabeledGroup],
    path: str | Path,
) -> Path:
    if not groups:
        raise ValueError("at least one labeled document group is required")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("labeled document group IDs must be unique")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for group in groups:
            for example in group.examples:
                handle.write(
                    json.dumps(example_to_json(example), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
    manifest = {
        "schemaVersion": "1",
        "path": output.name,
        "groupDigests": [group.digest for group in groups],
        "exampleCount": sum(len(group.examples) for group in groups),
        "referenceBoundary": "references-used-offline-only",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
