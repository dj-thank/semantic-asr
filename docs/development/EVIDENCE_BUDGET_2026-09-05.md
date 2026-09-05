# Optional evidence execution contract

Issue #24; baseline `a6a6dc2b9c6f7ae3bc7e8f06de54e9439252aa51`.

## Reproduced defect

Four actual-orchestrator fixtures (0ms/0 actions, 0ms/4 actions, 1000ms/0 actions,
and 1ms/1 action) each invoked the teacher once despite an empty approved plan.
The new `test_unfunded_teacher_never_executes` cases failed before this repair.

## Admission and scope

The query policy can request a teacher action; it cannot authorize a call outside
the planner. There is at most one teacher action per window because it ranks the
whole candidate pool. Teacher, re-listening, second-ear and forced-alignment calls
share the small `EvidenceExecution` admission/receipt component. Re-listening and
second-ear dispatch reuse one implementation without changing their decode controls.
A planned lexicon action with no runtime handler is no longer advertised; existing
evidence enrichment remains in the primary decode path.

This is a **per-window optional-call budget**, not a recording-wide latency cap.
Primary ASR, preprocessing, model provisioning and primary enrichers are outside
its scope. Plans reserve estimated milliseconds, not measured milliseconds.
`evidenceBudgetUsedMs` remains the planned estimate for compatibility and gains an
explicit semantics field. `evidenceExecution` records attempted/completed/cache-hit/
failed/not-executed calls, conservative admitted estimates and actual elapsed time.
`secondEarActionCount` and segment `actions` still describe the plan; use the new
execution receipts for actual invocation counts.

Cache admission consumes one action and its conservative estimate; there is no
unbudgeted cache lookup that might become an unexpected inference miss. Cache hits
are reported separately from uncached completions. A failed call consumes admission
and elapsed time. Backend OSError/RuntimeError/ValueError (including TimeoutError)
retains existing observed evidence, records the exception type and does not export
arbitrary exception messages. Primary decode failures remain errors.

The executor cannot preempt a synchronous backend call. If actual optional-call
time exceeds the budget, subsequent launches are suppressed. It deliberately
reports `hardDeadlineEnforced=false`. This is not hard real-time scheduling.

## Compatibility and evidence

No recognition model, fusion weight, profile or root evidence schema changes.
Underfunded teachers now abstain instead of producing unauthorized normalized text.
Failure fallback is visible, not silently reported as successful evidence. Diagnostic
elapsed time is not inserted into the immutable observed evidence hash.

Validation: 21 new parametrized cases; full local suite 498 passed / 3 pre-existing
strict legacy DecodeRequest xfails. Tests include real orchestration/cache reuse,
competing action limits, teacher timeout/invalid-response failures, empty primary
candidates, malicious plan changes, duplicate execution, elapsed overrun and invalid
budget types. The reference model-free/public-decision tests remain in the full suite.

```bash
python -m pytest -q tests/test_evidence_budget_contract.py tests/test_longform.py tests/test_lattice_planner.py
python -m pytest -q
python scripts/replay_phonetic_decisions.py
```

No acoustic accuracy or deployment-quality claim follows from these contract tests.
