# Software release checklist

- [x] Software licence approved by the user: MIT (separate from CC BY 4.0 data licence).
- [ ] Add final CITATION.cff version/date/DOI when assigned.
- [x] Fresh macOS arm64/Python 3.12 installation passes pip check; resolved versions locked.
- [x] Self-contained source-copy tests: 372 passed, 2 skipped.
- [x] Optional skips: processed-corpus integration and pinned FeatureIDE/JDK integration.
- [x] Additional 2026-10-07 local suite: 373 passed, 1 skipped, including real FeatureIDE controls; two new CSV-comparison tests also passed separately.
- [x] Restore the compatible data archive with verified checksums (temporary checkout).
- [x] Fresh inventory, structural/SAT, provenance and all 500 FeatureIDE evaluations match the archived CSVs byte for byte (2026-10-07).
- [ ] Recompute archived results and compare tables/figure data; retain logs.
- [x] Automated candidate/history credential-pattern review completed; see software-validation-2026-10-07.md.
- [ ] Review tracked files and history for private material and third-party rights.
- [ ] Review all outstanding changes; approve final commit/tag and release version.
- [ ] Archive exactly the approved source release, not the working directory.

Expert-response extraction/analysis is deferred and not promised by this release.
Data archive publication/privacy gates remain separate.
