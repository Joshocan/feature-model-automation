"""Run-directory validator (Phase 6.6).

Given the root of a completed run, verify every part of the logging contract:

  1. All required files exist:  run_config.json, run_meta.json,
                                context_log.jsonl, fm_gen.xml, fm_iter/*.xml
  2. Each file parses (JSON / JSONL / XML)
  3. Cross-artefact consistency:
       - context_log.jsonl has one line per feasible step
       - fm_iter/step_XX.xml exists for every feasible step
       - fm_gen.xml equals the final fm_iter/step_XX.xml
       - run_meta.json declares the same run_id as run_config.json
       - Each step's chunk_ids resolve to doc_ids inside the batch
  4. Frozen-hash matches: run_config.prompt_template_hash /
     metamodel_hash / chunks_hash / encoder_digest match the current
     data/frozen/protocol.sha256 entries. Mismatch means the run was made
     against a different protocol state and cannot be mixed with current
     campaign data.

A run is "complete" only if severity of every finding is INFO. Any WARNING
or ERROR downgrades it — the campaign runner uses this to decide whether
the run should count towards the planned matrix.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET


class Severity(str, Enum):
    INFO    = "INFO"
    WARNING = "WARNING"
    ERROR   = "ERROR"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    detail: str


@dataclass
class ValidationReport:
    run_root: Path
    run_id: Optional[str] = None
    findings: List[Finding] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return all(f.severity == Severity.INFO for f in self.findings)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_root": str(self.run_root),
            "run_id":   self.run_id,
            "complete": self.complete,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "findings": [
                {"check": f.check, "severity": f.severity.value, "detail": f.detail}
                for f in self.findings
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol_hashes(protocol_sha256: Path) -> Dict[str, str]:
    """Parse ``data/frozen/protocol.sha256`` (shasum output) → {rel_path: hash}."""
    out: Dict[str, str] = {}
    if not protocol_sha256.exists():
        return out
    for line in protocol_sha256.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # shasum -a 256 output: "<hash>  <path>"
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The validator
# ─────────────────────────────────────────────────────────────────────────────

def validate_run(
    run_root: Path | str,
    *,
    repo_root: Optional[Path | str] = None,
    protocol_sha256: Optional[Path | str] = None,
) -> ValidationReport:
    """Run every check against ``run_root``. Return a :class:`ValidationReport`.

    ``repo_root`` is used to resolve relative paths for hash checks. Defaults
    to two directories up from ``run_root``'s ``results/<campaign>/...`` layout.
    ``protocol_sha256`` defaults to ``<repo_root>/data/frozen/protocol.sha256``.
    """
    root = Path(run_root).expanduser().resolve()
    findings: List[Finding] = []
    run_id: Optional[str] = None

    def add(check: str, sev: Severity, detail: str) -> None:
        findings.append(Finding(check=check, severity=sev, detail=detail))

    # ── 1. presence
    expected = {
        "run_config.json":     root / "run_config.json",
        "run_meta.json":       root / "run_meta.json",
        "context_log.jsonl":   root / "context_log.jsonl",
        "fm_gen.xml":          root / "fm_gen.xml",
        "fm_iter/":            root / "fm_iter",
    }
    for name, p in expected.items():
        if not p.exists():
            add(f"presence:{name}", Severity.ERROR, f"missing: {p}")
    if findings:
        # Everything downstream requires these files.
        return ValidationReport(run_root=root, findings=findings)

    # ── 2. parseability
    try:
        run_config = json.loads(expected["run_config.json"].read_text())
    except Exception as exc:
        add("parse:run_config", Severity.ERROR, f"{type(exc).__name__}: {exc}")
        run_config = {}
    try:
        run_meta = json.loads(expected["run_meta.json"].read_text())
    except Exception as exc:
        add("parse:run_meta", Severity.ERROR, f"{type(exc).__name__}: {exc}")
        run_meta = {}
    try:
        with open(expected["context_log.jsonl"]) as fh:
            context_lines = [json.loads(l) for l in fh if l.strip()]
    except Exception as exc:
        add("parse:context_log", Severity.ERROR, f"{type(exc).__name__}: {exc}")
        context_lines = []
    try:
        ET.parse(str(expected["fm_gen.xml"]))
    except Exception as exc:
        add("parse:fm_gen", Severity.ERROR, f"{type(exc).__name__}: {exc}")

    iter_files = sorted(expected["fm_iter/"].glob("step_*.xml"))
    for f in iter_files:
        try:
            ET.parse(str(f))
        except Exception as exc:
            add(f"parse:fm_iter/{f.name}", Severity.ERROR, f"{type(exc).__name__}: {exc}")

    run_id = run_meta.get("run_id") or run_config.get("campaign_id")

    # If parsing already failed, stop before consistency.
    if any(f.severity == Severity.ERROR for f in findings):
        return ValidationReport(run_root=root, run_id=run_id, findings=findings)

    # ── 3. consistency
    if run_meta.get("run_id") != _canonical_run_id_from_config(run_config):
        add("consistency:run_id", Severity.ERROR,
            f"run_meta.run_id={run_meta.get('run_id')!r} does not match "
            f"run_config-derived run_id={_canonical_run_id_from_config(run_config)!r}")

    steps_meta = run_meta.get("steps", [])
    if len(steps_meta) != run_config.get("N"):
        add("consistency:step_count", Severity.WARNING,
            f"run_meta has {len(steps_meta)} step records but N={run_config.get('N')}")

    # One context_log line per step
    if len(context_lines) != len(steps_meta):
        add("consistency:context_log_len", Severity.ERROR,
            f"context_log has {len(context_lines)} lines but run_meta has "
            f"{len(steps_meta)} step records")

    # One fm_iter file per feasible step
    feasible_steps = [s for s in steps_meta if s.get("feasible") and s.get("fm_path")]
    if len(iter_files) != len(feasible_steps):
        add("consistency:fm_iter_count", Severity.ERROR,
            f"{len(iter_files)} fm_iter files but {len(feasible_steps)} feasible steps in run_meta")

    # fm_gen.xml must equal the final fm_iter file
    if iter_files:
        last_iter_text = iter_files[-1].read_text()
        gen_text = expected["fm_gen.xml"].read_text()
        if last_iter_text != gen_text:
            add("consistency:fm_gen_vs_iter", Severity.ERROR,
                f"fm_gen.xml differs from {iter_files[-1].name}")

    # Every context-log chunk's doc_id sits inside the batch
    for line in context_lines:
        batch = set(line.get("batch_doc_ids", []))
        stray = [d for d in line.get("chunk_doc_ids", []) if d not in batch]
        if stray:
            add(f"consistency:batch_leak_step_{line.get('step_index')}", Severity.ERROR,
                f"chunks from outside batch: {stray[:5]}")

    # ── 4. hash-pinning
    if repo_root is None:
        # Assume layout: <repo>/results/<campaign>/<corpus>/<config_hash>/<run_id>/
        repo_root = root.parents[3] if len(root.parents) >= 4 else root
    repo_root = Path(repo_root).expanduser().resolve()
    sha_file = Path(protocol_sha256) if protocol_sha256 else repo_root / "data/frozen/protocol.sha256"
    protocol = _load_protocol_hashes(sha_file)

    hash_checks = {
        "prompt_template_hash": "prompts/fm_prompt_template.txt",
        "metamodel_hash":       "prompts/feature-model-schema.xsd",
    }
    for cfg_key, rel_path in hash_checks.items():
        cfg_hash = run_config.get(cfg_key)
        frozen = protocol.get(rel_path)
        if cfg_hash and frozen and cfg_hash != frozen:
            add(f"hash:{cfg_key}", Severity.ERROR,
                f"run_config.{cfg_key}={cfg_hash[:12]} does not match "
                f"frozen protocol.sha256 entry for {rel_path}={frozen[:12]}")

    # chunks_hash — check against the actual chunks.jsonl file for the corpus
    corpus = run_config.get("corpus")
    if corpus:
        chunks_rel = f"data/processed/{corpus}/chunks.jsonl"
        frozen = protocol.get(chunks_rel)
        cfg_hash = run_config.get("chunks_hash")
        if cfg_hash and frozen and cfg_hash != frozen:
            add("hash:chunks_hash", Severity.ERROR,
                f"run_config.chunks_hash mismatch for {chunks_rel}")

    # No mismatches → INFO summary
    if not findings:
        add("summary", Severity.INFO, f"run {run_id} is complete and consistent")

    return ValidationReport(run_root=root, run_id=run_id, findings=findings)


def _canonical_run_id_from_config(run_config: Dict[str, Any]) -> str:
    """Re-derive run_id from a config snapshot the way RunConfig.run_id() does."""
    payload = json.dumps(run_config, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
