# Output bundle integrity

`TranscriptResult.write()` and `write_outputs()` publish the selected UTF-8
artifacts, then write `<stem>.semantic-asr.bundle.json` **last**. The return mapping
now includes the additive `manifest` key. Existing artifact names, text, subtitle
rendering, and evidence schemas are unchanged. `formats=set()` still returns `{}`
without creating an output directory.

```python
from semantic_asr.outputs import verify_output_bundle

paths = result.write("transcripts")
receipt = verify_output_bundle(
    paths["manifest"], expected_evidence_sha256=result.evidence_sha256
)
```

The completion manifest has this versioned shape:

```json
{
  "schema": "semantic-asr-output-bundle-v1",
  "status": "complete",
  "evidenceSha256": "<verified longform evidence SHA-256>",
  "artifacts": {
    "observed": {
      "filename": "meeting.observed.txt",
      "sha256": "<SHA-256 of exact UTF-8 file bytes>",
      "sizeBytes": 123
    }
  }
}
```

Only selected artifacts are listed, including the facade JSON and utterance SRT
when requested. Names are local filenames, never absolute source paths. Missing,
changed, duplicate, symlinked or nonregular artifacts and unknown schema/status
are rejected. Duplicate JSON keys and path traversal are rejected. The manifest
does not authenticate its author or prove recognition accuracy; an attacker able
to replace both artifacts and hashes can construct a different consistent bundle.

## Publication and interruption

All serialization and known-destination checks occur before publication. A
nonblocking OS advisory lock at `.semantic-asr-export.lock` covers the **output
directory**, including different source stems with overlapping artifact names.
The checkpoint writer uses the same lock primitive. A competing cooperating
writer or verifier raises `RuntimeError`; callers may retry after the writer
finishes. The one-byte lock file remains in place to preserve its inode, but its
lock is released by the OS when a process exits or is killed. Do not delete it
while writers/readers may be active.

After acquiring the lock, the writer repeats preflight, removes the previous
completion manifest when overwriting, publishes each artifact atomically, checks
every artifact's bytes, and atomically publishes the new manifest. Known conflicts
with `overwrite=False` leave the old export unchanged. Symlink destinations are
rejected even with `overwrite=True`.

This is **not a multi-file filesystem transaction**. Disk failure, process death,
or an interrupted overwrite can leave partial files or mixed generations, without
a valid completion manifest. There is no rollback or automatic promotion. Retry
explicitly with `overwrite=True` or write to a new output directory. Individual
file fsync does not promise filesystem-wide durability after power loss.

Consumers must verify **all listed artifacts**, not merely see a manifest file.
Uncooperating external writers are not locked out; ordinary hash checks detect
changed bytes, but this is not a hostile-filesystem security boundary. Verification
holds the same lock only during its checks. It does not guarantee that subsequent
path reads see the same bytes after another explicit overwrite. Consumers needing
a lasting snapshot must coordinate copying/reading with writers or use a separate
immutable export directory.

## Compatibility and migration

- Legacy exports remain readable as individual files. Without a completion
  manifest they are not accepted by `verify_output_bundle`. Re-export a verified
  result to create a bundle; do not manufacture a completion marker for old files.
- The low-level `publish_output_documents` retains its per-file behavior when no
  `manifest_path` is supplied. Bundled publication also requires `evidence_sha256`
  and unique destinations in the manifest directory.
- When overwriting with fewer formats, old unselected files remain untouched and
  are **not members of the new bundle**. Use the manifest's `artifacts` mapping,
  not a directory glob, to determine the completed set.
- A later explicit overwrite from a different source stem can replace a shared
  filename and invalidate an older bundle. Always reverify before consuming it;
  use separate directories when retaining multiple generations or colliding stems.
- Recognition defaults, confidence calibration, observed/normalized separation,
  evidence hashing, and cache keys do not change. This implements the export
  completion portion of Issue #31, not its full evidence-schema migration.
