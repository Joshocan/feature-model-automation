# — run inventory

The inventory is an offline artifact audit, not semantic or conformance evaluation.
It can run while the evaluation specification is a draft. It does not resolve D01–D05,
change the generation protocol, or invoke model clients. Run after campaign writers
have stopped; concurrent updates can invalidate a snapshot.

From the repository root:

```bash
./.venv/bin/python scripts/inventory_campaign.py \
  --output results/ifs-2027/analysis/inventory-2026-09-26-v1
```

Choose a new output directory name for each audit. Existing directories are never
overwritten. Writes are restricted to new subdirectories of the campaign's analysis
directory. Original run files are read and SHA-256 hashed, not modified.

## Tables (JSON and CSV)

- `runs`: one row per planned matrix run, including missing and failed runs; complete
  generation settings, original IDs, matrix membership, terminal status, failure
  details, checkpoint availability, recovery history and resolved/original paths.
- `recovery_snapshots`: archived states linked to their planned run. They are NOT
  extra repetitions, and are not necessarily independent attempts. Resume snapshots
  can overlap; do not sum their tokens or costs.
- `artifacts`: per-file original/resolved paths, bytes and SHA-256 for selected
  runs, recovery snapshots and manifest-listed pilots, including intermediate/raw
  responses where present.
- `pilots`: observed pilot metadata records kept separate from the main denominator.
  `primary_within_pilot` does not imply main-campaign inclusion. Missing/empty sources
  are flagged, including isolated schema-depth sources. This table does not invent
  missing planned pilot runs or claim pilot cohort completeness from discovered files.
- `issues`: missing or changed source hashes, malformed metadata, identity mismatch,
  inconsistent completion, unreadable artifacts and unplanned directories.
- `summary.json`: counts, contract and implementation hashes, creation timestamp
  and relocation mapping. Conformance and semantics remain explicitly unevaluated.

The process returns nonzero when issues are found, but writes its diagnostic tables.
Malformed matrices or duplicate planned IDs abort rather than change the denominator.
Missing metadata is not equivalent to recorded failure: nonexistent runs are
`not_started`, existing incomplete directories are `partial`, corrupt JSON is
`invalid_metadata`. Original status is always separate from inventory classification.
Completion requires matching identity, all N accepted step records/checkpoints and
the final artifact; this is not proof of XSD validity or correctness.

## Relocated data

An optional `--path-map mapping.json` accepts repository-relative directory-prefix
replacements, for example `{"results/old-name": "results/archive/old-name"}`.
Longest prefix wins. Original identities and paths remain in the inventory; resolved
paths and input hashes document the relocation. Escapes outside the repository are
rejected. This makes relocation explicit rather than silently discovering and
substituting another similarly named run.

## Tests

```bash
./.venv/bin/python -m pytest tests/test_inventory.py tests/test_evaluation_spec.py -q
```

Downstream metric drivers still require the structural, semantic, provenance and aggregation stages. They must use the
matrix-led inventory and must not infer experimental membership from final XML alone.
These files can contain original provider error text or paths from recovery history;
review/redact private information in a separately documented publication export.
