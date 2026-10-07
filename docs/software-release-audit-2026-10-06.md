# Software release preparation — 2026-10-06

## Changes made

1. Matrix/runner/recovery tests use a temporary synthetic corpus with tracked
   protocol inputs. Statistical-family tests use generated planned rows rather
   than a private wide.csv. Expert packet tests use synthetic blank workbooks.
2. The expert-form default now uses the renamed Expert-evaluation-form.xlsx.
   Current I/J answer columns are recognised. Extra table rows and missing
   descriptive/source fields cause explicit failure rather than silently emitting
   incomplete packets. The tracked historical workbook is unchanged. Expert
   response extraction remains deferred.
3. Direct dependencies were separated from optional PDF ingestion. Removed the
   unused LangChain dependency group and conflicting mandatory SentenceTransformer
   dependency; the primary LocalTransformerEncoder retains recorded transformer
   and torch versions. A fresh macOS arm64 Python 3.12 environment resolved and
   installed successfully; pip check found no broken requirements. Its full
   package versions are in config/requirements-macos-arm64-py312.lock.
   The historical .venv was not changed. The lock is version-based, not wheel-hashed.
4. README/SETUP were rewritten for the current engine and offline evaluation.
   Setup now targets .venv-release and does not create retired pipeline folders
   or download NLTK data. The test wrapper no longer installs dependencies implicitly.
5. restore_research_data.py verifies archive checksums and original artifact hashes,
   restores data/results paths only, rejects unsafe/duplicate paths, and refuses
   differing destination files. Default is dry-run. Historical software/configuration
   snapshots are not copied over the selected software release.
6. Added CITATION.cff, CHANGELOG.md, reproduction instructions, a compatibility
   manifest and GitHub Actions tests. Software licence and release identifiers are
   not invented. Existing data-archive licence templates retain their separate scope.

## Validation performed

- Fresh source-only copy plus fresh environment: **372 passed, 2 skipped**.
  Skips were local processed-corpus integration and actual FeatureIDE/JDK controls.
- Archive restoration: 8,530 original artifacts verified; 8,499 data/results
  targets eligible, 8,486 absent files copied into a temporary checkout. Existing
  identical targets were retained. No original campaign output was modified.
- Evaluation verifier: 694 planned, 500 completed, 500 primary semantic eligible,
  369 strict admissible; reported `denominators_reconciled`.
- All six paper figures regenerated in the temporary checkout. All six figure
  source CSVs matched archived paper-figures-v4 bytes exactly.
- Restore tests cover dry-run, idempotent copy, mismatch refusal, corrupted source
  and unsafe paths. Shell syntax and git diff whitespace checks passed.
- No generation API calls, campaigns, Git commits/tags, pushes or publication.

Compatible candidate data manifest SHA256:
`e10a547b59701b614558bbc1433d0c0b51f9124b1d02aa958a301fddb8614b0d`.

## Not yet certified

The fresh environment test did not recompute every semantic score from XML or
rerun every SAT/FeatureIDE evaluation. It verified the software tests, restored
evidence, denominator consistency and figure derivation from existing metrics.
GitHub-hosted CI has been configured but not executed here. Other platforms and
optional PDF ingestion are not covered by the macOS lock. Dependency/model download
and full scientific reproduction require additional recorded checks.

The latest tracked expert instrument contains an extra source table at rows
25–37; preparing a new blank outgoing packet requires a reviewed copy and the
current citation fields. Do not use returned responses as a replacement fixture.

## Remaining decisions before release

- Software licence resolved by subsequent user instruction: MIT. Added root
  LICENSE and updated CITATION.cff; the data archive remains CC BY 4.0.
- Choose a release version and approve the final change set before commit/tag.
- Assign the final data/software DOI and replace candidate compatibility metadata.
- Complete full scientific recomputation if claiming end-to-end reproduction,
  plus final rights/privacy and manuscript reconciliation reviews.

Do not publish the full working directory. Temporary validation environments and
restored evidence were kept under /tmp, outside the release file set.
