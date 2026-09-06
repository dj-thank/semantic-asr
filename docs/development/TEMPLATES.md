# Evidence templates

Copy only the relevant template into a dated record. Replace every placeholder;
an empty template is not a completed task. Link the record from its Issue and PR.

## Implementation completion

- Issue / purpose / explicitly out of scope:
- Base SHA/tree and final SHA/tree:
- Files read and changed; public API/schema compatibility:
- Minimal reproducer and RED result on the base:
- GREEN command, actual runtime, passed/failed/skipped/xfail counts:
- Clean-wheel command run outside the checkout:
- Artifacts and SHA-256; allowed publication operations:
- Known limitations / blockers / rollback:
- Completion status: engineering-complete / blocked / partial:

## Experiment preregistration and result

### Before any test inspection

- Question, hypothesis and what would falsify it:
- Parent Issue; primary source and exact revision; difference from that source:
- Train/dev/calibration/test/exposed manifest digests and rights:
- Group/speaker/recording lineage, duplication exclusions, unknown fields:
- Baselines and ablations; same input, decoder/model revisions and budget:
- Primary metric, critical slices, false-correction limit, confidence method:
- Selection procedure using dev only; calibration procedure on separate data:
- Seeds, optimizer/config, maximum steps/trials/audio/time/storage/cost:
- Freeze artifact hash and time; test access policy:
- Stopping, OOM/NaN handling and resume plan:

### After execution

- Code/tree/runtime/hardware, exact commands and actual stage statuses:
- Model input fields; evidence that reference text was not an inference input:
- For training: updated tensor names, initial/final hashes and parameter deltas,
  frozen-module checks, optimizer steps, gradient/loss logs:
- Checkpoint location/hash, base model identity, fresh-process reload results:
- All trials and rejected configurations, not only the winning run:
- Same-run paired metrics, uncertainty, improve/harm/tie and all failures:
- Latency percentiles/RTF, peak memory, actual cost and sample exclusions:
- Negative results, spelling/semantic distinctions and unverified assumptions:
- State: experiment-complete / failed / partial / blocked:
- Separate promotion decision against the preregistered gate; no automatic default:

## Architecture decision record

- ADR ID, date, status (proposed / accepted / superseded), owner/review:
- Context and concrete code paths; constraints and invariants:
- Alternatives, including leave-as-is; evidence and trade-offs:
- Decision and consequences; what this decision does not authorize:
- Public API/schema/cache migration and rollback:
- Characterization/negative tests and acceptance evidence:
- Related Issues, experiment artifacts and superseded ADR links:

A discussion is not a decision until its chosen outcome is captured here or in
another versioned ADR. A checksum proves identity/consistency, not authorship or
recognition correctness. Preserve previous records instead of rewriting failures.
