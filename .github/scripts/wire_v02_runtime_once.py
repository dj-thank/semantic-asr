from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


longform_path = Path("src/semantic_asr/longform.py")
text = longform_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from .cache import CacheKey, EvidenceCache, TeacherCacheEntry\n",
    "from .cache import CacheKey, EvidenceCache, TeacherCacheEntry\n"
    "from .candidate_pool import merge_candidate_pools\n",
    "longform candidate-pool import",
)
text = replace_once(
    text,
    "from .fusion import FusionConfig, evidence_summary, fuse_candidates\n",
    "from .evidence_router import (\n"
    "    QuantileBalancedRouterConfig,\n"
    "    RouterState,\n"
    "    route_evidence_actions,\n"
    ")\n"
    "from .fusion import FusionConfig, evidence_summary, fuse_candidates\n",
    "longform router imports",
)
merge_start = text.find("\ndef _strength(candidate: CandidateEvidence)")
merge_end = text.find("\ndef _lattice_context", merge_start)
if merge_start < 0 or merge_end < 0:
    raise SystemExit(
        f"longform merge boundaries not found: start={merge_start}, end={merge_end}"
    )
text = (
    text[:merge_start]
    + "\ndef merge_candidates(\n"
    + "    primary: Iterable[CandidateEvidence],\n"
    + "    additional: Iterable[CandidateEvidence],\n"
    + ") -> list[CandidateEvidence]:\n"
    + "    return merge_candidate_pools(primary, additional, id_prefix=\"merged\")\n\n"
    + text[merge_end:]
)
text = replace_once(
    text,
    "        evidence_budget: EvidenceBudget | None = None,\n"
    "        evidence_enricher: Callable[[CandidateEvidence], CandidateEvidence] | None = None,\n",
    "        evidence_budget: EvidenceBudget | None = None,\n"
    "        balanced_router: bool = False,\n"
    "        router_state: RouterState | None = None,\n"
    "        router_config: QuantileBalancedRouterConfig | None = None,\n"
    "        evidence_enricher: Callable[[CandidateEvidence], CandidateEvidence] | None = None,\n",
    "longform constructor parameters",
)
text = replace_once(
    text,
    "        self.evidence_budget = evidence_budget or EvidenceBudget()\n"
    "        self.evidence_enricher = evidence_enricher\n",
    "        self.evidence_budget = evidence_budget or EvidenceBudget()\n"
    "        self.balanced_router = bool(balanced_router)\n"
    "        self.router_state = router_state or RouterState()\n"
    "        self.router_config = router_config or QuantileBalancedRouterConfig()\n"
    "        self.evidence_enricher = evidence_enricher\n",
    "longform constructor assignments",
)
text = replace_once(
    text,
    "        )\n\n        additional: list[CandidateEvidence] = []\n",
    "        )\n"
    "        routing_diagnostics: dict[str, Any] = {\"enabled\": False}\n"
    "        if self.balanced_router and (plan.selected or plan.rejected):\n"
    "            routed = route_evidence_actions(\n"
    "                (*plan.selected, *plan.rejected),\n"
    "                budget=self.evidence_budget,\n"
    "                state=self.router_state,\n"
    "                config=self.router_config,\n"
    "            )\n"
    "            plan = routed.plan\n"
    "            routing_diagnostics = {\n"
    "                \"enabled\": True,\n"
    "                \"stateDigest\": routed.state_digest,\n"
    "                \"selected\": [\n"
    "                    {\n"
    "                        \"actionId\": row.action.action_id,\n"
    "                        \"kind\": row.action.kind,\n"
    "                        \"routingScore\": row.routing_score,\n"
    "                        \"loadBalanceBonus\": row.load_balance_bonus,\n"
    "                        \"empiricalRewardBonus\": row.empirical_reward_bonus,\n"
    "                        \"semanticBonus\": row.semantic_bonus,\n"
    "                        \"redundancyPenalty\": row.redundancy_penalty,\n"
    "                    }\n"
    "                    for row in routed.selected\n"
    "                ],\n"
    "                \"rejectedCount\": len(routed.rejected),\n"
    "            }\n\n"
    "        additional: list[CandidateEvidence] = []\n",
    "longform router execution",
)
text = replace_once(
    text,
    "            if teacher_result is not None and not teacher_result.abstained:\n"
    "                candidates = [\n"
    "                    replace(\n"
    "                        candidate,\n"
    "                        teacher=teacher_result.probabilities.get(candidate.candidate_id),\n"
    "                    )\n"
    "                    for candidate in candidates\n"
    "                ]\n"
    "                ranked = fuse_candidates(candidates, self.fusion_config)\n",
    "",
    "longform legacy teacher observed reranking",
)
text = replace_once(
    text,
    "            \"evidenceStoppingReason\": plan.stopping_reason,\n"
    "            \"teacherUsed\": teacher_result is not None,\n",
    "            \"evidenceStoppingReason\": plan.stopping_reason,\n"
    "            \"evidenceRouting\": routing_diagnostics,\n"
    "            \"teacherUsed\": teacher_result is not None,\n"
    "            \"teacherAffectsObserved\": False,\n",
    "longform diagnostics",
)
longform_path.write_text(text, encoding="utf-8")


cli_path = Path("src/semantic_asr/cli_v2.py")
text = cli_path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "from .contracts import CandidateEvidence\n",
    "from .contracts import CandidateEvidence\nfrom .evidence_router import RouterState\n",
    "CLI router import",
)
text = replace_once(
    text,
    "def _hotwords(args: argparse.Namespace) -> tuple[str, ...]:\n",
    "def _load_router_state(path: str | Path | None) -> RouterState:\n"
    "    if path is None:\n"
    "        return RouterState()\n"
    "    payload = json.loads(Path(path).read_text(encoding=\"utf-8\"))\n"
    "    return RouterState(\n"
    "        selection_count={\n"
    "            str(key): int(value)\n"
    "            for key, value in dict(\n"
    "                payload.get(\"selectionCount\", payload.get(\"selection_count\", {}))\n"
    "            ).items()\n"
    "        },\n"
    "        reward_sum={\n"
    "            str(key): float(value)\n"
    "            for key, value in dict(\n"
    "                payload.get(\"rewardSum\", payload.get(\"reward_sum\", {}))\n"
    "            ).items()\n"
    "        },\n"
    "        total_selections=int(\n"
    "            payload.get(\"totalSelections\", payload.get(\"total_selections\", 0))\n"
    "        ),\n"
    "        version=str(payload.get(\"version\", \"1\")),\n"
    "    )\n\n\n"
    "def _hotwords(args: argparse.Namespace) -> tuple[str, ...]:\n",
    "CLI router-state loader",
)
text = replace_once(
    text,
    "            evidence_budget=EvidenceBudget(\n"
    "                total_cost_ms=args.evidence_budget_ms,\n"
    "                max_actions=args.max_evidence_actions,\n"
    "            ),\n"
    "            window_ms=args.window_ms,\n",
    "            evidence_budget=EvidenceBudget(\n"
    "                total_cost_ms=args.evidence_budget_ms,\n"
    "                max_actions=args.max_evidence_actions,\n"
    "            ),\n"
    "            balanced_router=args.evidence_router == \"balanced\",\n"
    "            router_state=_load_router_state(args.router_state),\n"
    "            window_ms=args.window_ms,\n",
    "CLI transcriber routing",
)
text = replace_once(
    text,
    "    transcribe.add_argument(\"--max-evidence-actions\", type=int, default=8)\n",
    "    transcribe.add_argument(\"--max-evidence-actions\", type=int, default=8)\n"
    "    transcribe.add_argument(\n"
    "        \"--evidence-router\",\n"
    "        choices=[\"legacy\", \"balanced\"],\n"
    "        default=\"balanced\",\n"
    "    )\n"
    "    transcribe.add_argument(\"--router-state\")\n",
    "CLI routing arguments",
)
cli_path.write_text(text, encoding="utf-8")
