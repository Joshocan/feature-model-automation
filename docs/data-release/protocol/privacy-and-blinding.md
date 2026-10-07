# Privacy and expert-blinding release review

Status: PENDING author review; no publication approval is implied.

Expert anonymity and model blinding are separate. Expert anonymity protects the
people giving ratings. Blinding prevents them knowing which system produced a
model. Public XML, run metadata and matching model names can reveal the latter
even without a selection key and even when all experts are anonymous.

## Checks before public release

1. Confirm all expert submissions are collected and locked before releasing
   identifiable model outputs, or document an explicit alternative release plan
   and its impact on blinding. A reserved DOI is not permission to expose files.
2. Keep response forms, emails, signatures, participant keys and identifying
   comments out of this non-expert release. Check document properties, hidden
   sheets, filenames, comments and Git history if publishing a Git repository.
3. Review protocol/privacy-scan.json. The recorded scan found no credential-pattern
   hits and flagged personal absolute paths in ten files. That is not a complete
   privacy or security audit. Decide whether each path can remain; otherwise
   prepare documented public derivatives while preserving private evidence.
4. Complete the rights review in LICENSES/README.md, including generated quotations.
5. Recheck the exact final bundle members, not only the working directory.

No participant results are being released now. Any later participant-level data
release needs its own review of what participants agreed could be shared;
removing names alone is not proof that free text cannot identify someone.

## Decision record to complete

- Reviewer and review date: pending.
- Expert submissions locked on / alternative plan: pending.
- Exact release manifest hash reviewed: pending.
- Personal-path decisions and rights evidence: pending.
- Release approval or remaining exclusions: pending.
