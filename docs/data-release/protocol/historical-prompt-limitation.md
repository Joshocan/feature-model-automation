# Historical prompt coverage and proposed disclosure

Status: documented limitation; author acceptance or further recovery pending.

The recorded audit covers 694 main-campaign runs, ten Astra N=1 extension runs
and 55 pilots. All main and extension runs have the matching prompt snapshot.
Among pilots, 38 use the main prompt and two have an older snapshot recovered
from Git. Three other prompt hashes cover the remaining 15 pilots; their exact
template text has not been recovered. See prompt-recovery.json and the per-group
prompt_coverage in ../release-validation.json for hashes and counts.

The search covered local Git revisions of prompts/fm_prompt_template.txt and the
consolidated pilot snapshot. Other backups or editor history may still contain
the missing bytes. A recovered candidate must hash to the recorded value before
being labelled the original. Do not substitute the current template or reconstruct
an alleged original from memory. Different hashes establish different bytes,
not necessarily substantively different instructions.

## Resolution options

1. Recover and hash-verify all missing snapshots; update the coverage catalogue.
2. Explicitly accept the limitation and retain those runs as historical pilot
   evidence, not fully reconstructible prompt-based replications. Keep them
   separate from the main-campaign comparisons.

## Proposed paper/archive disclosure, subject to author approval

Exact prompt-template snapshots are available for all main-campaign and Astra
extension runs and 40 of 55 pilot runs. For 15 early pilots, three recorded
template hashes could not be matched to recovered template text. Their saved
outputs and metadata are retained as historical evidence, but their exact prompt
templates cannot be reconstructed from this archive. These pilots are not pooled
with the main-campaign repetitions.

Author decision, reviewer and date: pending.
