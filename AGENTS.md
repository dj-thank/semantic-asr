# Agent execution contract

Read [the execution index](docs/development/README.md), the assigned GitHub Issue,
its prerequisites, and the current source before editing. Issue #23 is the release
roadmap; #21 tracks unfinished phonetic research. An issue being created is not
an implementation being completed.

## Start and scope

1. Record the actual base commit and tree. Confirm the issue is still open and
   prerequisites have evidence. The JSON plan is a dated snapshot, not live status.
2. Use one branch per task. Post the files you intend to edit; do not concurrently
   rewrite files owned by another task. A role in the plan is not a GitHub assignee.
3. First reproduce a bug or capture behavior with characterization tests. Keep
   file moves/refactoring separate from changes to recognition or score semantics.
4. Reuse existing contracts and experiment/checkpoint utilities. Do not introduce
   a second score type, data-rights registry, or generic agent framework.

## Invariants

- Observed speech and normalized text are separate. Language fluency is not proof
  of acoustic correctness. G2P is a pronunciation proposal, not an observation.
- Phone-derived morae are another view of the same evidence, not independent votes.
- Keep model, tokenizer, label set, audio/window, preprocessing and score-domain
  provenance. Mean-frame and mean-token likelihoods are not interchangeable.
- No raw preference becomes a correctness probability without applicable calibration.
- Preserve actual repetitions, fillers, repairs and uncertainty. A partial-span
  decode cannot silently replace a whole-window observation.
- Train/dev/calibration/test/exposed-regression data have different roles. Already
  inspected test examples must not be relabeled as unseen evaluation.
- Check rights separately for training, evaluation and publication. Never commit
  private recordings, credentials, private paths, unapproved text or model weights.
- Respect tool permissions and safety checks; do not route rejected writes through
  encoded payloads, alternate tools or workflows to bypass the rejection.

## Validation

In a suitable development environment, from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python -m pytest -q
python scripts/replay_phonetic_decisions.py
semantic-asr demo --output runs/demo.json
semantic-asr research-smoke --output runs/research-smoke.json
python -m build --wheel
```

Run optional NumPy and CPU PyTorch tests in their documented environments. Base
imports and help must not need models or network access. Test a wheel from outside
the checkout with no source PYTHONPATH. Never delete assertions, add xfails, relax
thresholds or silently skip a backend to manufacture a passing result. Existing
xfails are defects, not a target count. New test paths in the plan are to be created,
not commands falsely claimed to exist already.

## Real training and stopping

A training task needs real optimizer updates, finite gradients, before/after tensor
identities, unchanged frozen tensors, saved weights, fresh-process reload and a
separate evaluation. Downloads, mock backward passes, coefficient fitting and
checkpoint-shaped random tensors are not newly trained acoustic or LLM weights.

Declare finite steps/trials/audio/time/storage and cost limits before execution.
Log failed, partial and negative experiments. If a required resource or permission
is absent, record the specific blocker and completed evidence. Do not promise
unbounded work after the session or automatically promote a model.

## Completion report

Provide issue/PR, base and final SHA/tree, exact commands/environment, test results
including skipped/xfail cases, artifact locations and SHA-256, behavioral/migration
changes, and limitations. Separate engineering-complete, experiment-complete and
promotion-approved. Templates are in [the evidence templates](docs/development/TEMPLATES.md).
