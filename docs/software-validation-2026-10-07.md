# Software validation and release-content review — 2026-10-07

## Scope and status

This is local release-candidate validation, not publication approval or a hosted
CI result. No generation API calls, campaign reruns, commits, tags or pushes were
made. Original results are unchanged. Expert-response extraction remains deferred.

The source commit underlying the uncommitted candidate is
`2d56d25a53887d1e22ebad69d2df34d12ac51653`. The reproduction record includes hashes
of the actual working-tree evaluator sources; the commit alone does not identify
this candidate.

## Completed checks

- Recreated `.venv-release` from `config/requirements-macos-arm64-py312.lock`.
  Installation succeeded and `pip check` reported no broken requirements.
- Offline test suite: **373 passed, 1 skipped**. This includes real pinned
  FeatureIDE/JDK integration controls. The skipped legacy integration requires
  its optional processed-corpus input.
- Two additional data-independent CSV-comparison tests passed separately,
  checking numeric tolerance, missingness and schema differences.
- Fresh inventory, structural/SAT and provenance evaluation completed.
- FeatureIDE 3.10.0 checked all 500 completed outputs: **452 accepted, 48 rejected**.
  Its valid, malformed, duplicate-name and reference-model controls passed.
- All ten CSV files produced by those four stages match the archived versions
  **byte for byte**. This includes structural metrics, citation records and the
  complete FeatureIDE result table, not just their headline counts.

Full semantic recomputation and dependent stages are running separately; do not
claim their completion from the checks above. The driver writes a final
`comparison.json` only after all stages complete.

## Reproduction evidence

Fresh outputs, stage logs, exact commands and environment/source identities:

`results/ifs-2027/analysis/reproduction-2026-10-07/`

Release-content audit:

`results/ifs-2027/analysis/release-review-2026-10-07.json`

The offline driver is `scripts/validate_reproduction.py`. It refuses existing
output directories and never calls a generation provider. CSV comparisons retain
row ordering and missing values; numeric comparison uses absolute and relative
tolerances of `1e-6`. Differences require inspection, not automatic acceptance.
PDF byte equality is not the scientific acceptance criterion: figure source data
are compared instead. The audit and reproduction evidence live under ignored
`results/`; include selected reports deliberately in the final release evidence.

## Release-content review

The audit examines tracked and nonignored untracked candidate files, plus Git
history. No credential-pattern matches were found in the candidate or the
**358 historical Git blobs** scanned. This is a heuristic check, not a guarantee
that all private or copyrighted information has been cleared.

Remaining content decisions:

1. **Expert blinding:** `docs/expert-evaluation-strategy.md` contains the explicit
   blinded-code-to-model mapping. Keep this information nonpublic until expert
   responses are locked, or prepare a public protocol without the key. Removing
   a current file does not remove older copies from Git history.
2. **Workbook:** `data/Expert-evaluation-form.xlsx` has public-author metadata
   (Joshua Tetteh Ocansey) and one instruction comment by “Unknown Author”; that
   comment is not a returned expert response. Source-paper excerpts still need
   the researcher's redistribution review. The historical extra table rows
   previously documented are not silently removed by this audit.
3. **Rights:** confirm permission/licensing for third-party excerpts and reference
   artefacts. The software's MIT licence does not grant rights to third-party data.
4. **Portability:** the historical experiment document includes a developer-local
   working-directory example. Code/test matches for `/Users/` also include
   deliberate sanitisation rules and synthetic examples, not credentials.

Neither original returned expert forms, local API-key directories nor campaign
result directories should be added to the software release. Publish an approved
source snapshot, not a ZIP of the entire working directory or `.git` directory.

## GitHub and human gates

- The GitHub Actions workflow exists locally but is not yet committed. Hosted CI
  cannot certify this candidate until the approved changes are committed/pushed.
- Final version, release date, compatible data DOI/version and software DOI must
  be set consistently; a draft data record must not be described as published.
- Human rights/privacy/blinding approval and final commit selection remain
  necessary. This audit does not substitute for them.

## Suggested commit groups

Review each diff and stage explicit paths, not `git add .`. These are proposed
messages, not commits already made. Some files contain earlier work as well.

| Group | Suggested message | Files to review together |
|---|---|---|
| 1 | `test: make release tests independent of private campaign data` | `tests/conftest.py`, `tests/test_analysis_followup.py`, `tests/test_expert_v3.py`, `scripts/expert_artifacts_v3.py` |
| 2 | `build: lock release dependencies and modernise setup scripts` | `config/requirements.txt`, `config/requirements-ingestion.txt`, `config/requirements-macos-arm64-py312.lock`, `scripts/initial_setup.sh`, `scripts/install_requirements.sh`, `scripts/run_tests.sh`, `.gitignore` |
| 3 | `feat: restore and verify compatible research-data artifacts` | `scripts/restore_research_data.py`, `tests/test_restore_research_data.py`, `config/compatible-data.json` |
| 4 | `docs: document current replication and expert workflows` | `README.md`, `SETUP.md`, `docs/reproduce.md`, `docs/expert-evaluation-strategy.md` (subject to blinding review) |
| 5 | `chore: add MIT software license and citation metadata` | `LICENSE`, `CITATION.cff`, `CHANGELOG.md` (finalise approved release identifiers) |
| 6 | `ci: add offline tests and scientific reproduction validation` | `.github/workflows/tests.yml`, `scripts/validate_reproduction.py`, `scripts/audit_release_contents.py`, `tests/test_release_validation.py`, `docs/software-release-audit-2026-10-06.md`, `docs/software-release-checklist.md`, this report |
| 7 | `feat: prepare separately licensed Zenodo data releases` | `scripts/package_zenodo_data.py`, `scripts/update_data_release_documents.py`, changes under `docs/data-release/` |

Keep group 7 distinct from the MIT software metadata: those documents describe
the separately licensed data archive and its human approval requirements.
