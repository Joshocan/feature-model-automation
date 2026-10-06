#!/usr/bin/env python3
"""interactive campaign runner.

Menu-driven runner for a frozen iFS 2027 combined or lane matrix. It groups
planned runs by arm and walks through them with per-arm confirmations and
per-step live progress.

Usage:
  # Generate the two disjoint lane manifests first:
  python scripts/build_run_matrix.py --only-enabled --lane open_weight
  python scripts/build_run_matrix.py --only-enabled --lane astra

  # Terminal 1 — shared Ollama lane (models execute sequentially):
  python scripts/campaign.py \
    --matrix results/ifs-2027/run_matrix_enabled_open_weight.json \
    --lane open_weight

  # Terminal 2 — independent OpenAI lane, safe to run at the same time:
  python scripts/campaign.py \
    --matrix results/ifs-2027/run_matrix_enabled_astra.json \
    --lane astra

  # Only one arm, still confirm before starting:
  python scripts/campaign.py --matrix ... --arm guided_baseline

  # Skip per-run validator (validate later with scripts/validate_run.py):
  python scripts/campaign.py --matrix ... --no-validate

Design
------
* **Arms** are grouped from ``run.extra.arm``.
* **Order for "run all"**: guided_headline → guided_baseline → guided_curve
  → ablation → order_sensitivity → k_doc_sweep_kN.
* **Outcome-safe**: every persisted terminal outcome—including truncation,
  malformed XML, and infeasibility—is retained in the denominator and skipped
  on restart. Interrupted non-terminal artefacts require explicit restart.
* **Parallel-safe**: advisory per-run locks prevent two local processes from
  executing the same matrix row.
* **Per-step callback** prints one line per LLM call.
* **Cost estimate** uses pilot medians if available, else DEFAULT_* fallback
  from estimate_cost.py's price table.

Cost display is a *running estimate* — actual OpenAI billing is authoritative.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import shutil
import socket
import statistics as st
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.generation import (             # noqa: E402
    GenerationLLM,
    RunAlreadyExists,
    RunConfig,
    make_client,
    run_generation,
)
from fame.generation.loop import StepRecord  # noqa: E402
from fame.generation.persistence import (  # noqa: E402
    RunPaths,
    atomic_write_json,
    atomic_write_text,
)
from fame.generation.token_budget import (  # noqa: E402
    OllamaTokenCounter,
    TiktokenCounter,
    UniversalCounter,
)
from fame.retrieval import RetrievalService  # noqa: E402
from fame.validation import validate_run   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────────────────

# USD per 1M tokens (input, output). Keep in sync with estimate_cost.py.
PRICE_TABLE: Dict[str, tuple[float, float]] = {
    "gpt-6-astra":              (10.00, 40.00),
    "minimax-m3:cloud":          (0.60,  2.40),
    "deepseek-v4.1-flash:cloud": (0.00,  0.00),
    "gpt-oss:120b-cloud":        (0.15,  0.60),
    "glm-5.3-flash:cloud":       (0.15,  0.50),
    "deepseek-v4-pro:cloud":     (0.66,  1.98),
}
# Fallback if we can't derive tokens from pilot data.
DEFAULT_PROMPT_TOKENS   = {"rag": 25_000, "nonrag": 100_000}
DEFAULT_COMPLETION_TOKENS = 4_000
DEFAULT_WALL_SECONDS    = {"rag": 12.0, "nonrag": 30.0}


ARM_ORDER = [
    "guided_headline",
    "guided_baseline",
    "guided_curve",
    "ablation",
    "order_sensitivity",
    "k_doc_sweep_k3",
    "k_doc_sweep_k5",
    "k_doc_sweep_k10",
    "k_doc_sweep_k15",
    "astra_cross_corpus",
]


class RunClaimed(RuntimeError):
    """Raised when another campaign process currently owns a run."""


class RunClaim:
    """Advisory per-run lock, automatically released on process exit/crash."""

    def __init__(self, path: Path, *, blocking: bool = False) -> None:
        self.path = path
        self.blocking = blocking
        self._fh = None

    def __enter__(self) -> "RunClaim":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+", encoding="utf-8")
        try:
            flags = fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB)
            fcntl.flock(self._fh.fileno(), flags)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            raise RunClaimed(f"run is claimed by another process: {self.path.stem}") from exc
        self._fh.seek(0)
        self._fh.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "claimed_at": time.time()}, self._fh)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


def _run_config_from_row(run: Dict[str, Any]) -> RunConfig:
    """Build RunConfig while excluding runner-only annotations."""
    payload = {k: v for k, v in run.items()
               if k != "extra" and not k.startswith("_")}
    return RunConfig(**payload, extra=run.get("extra", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Ctrl-C handling
# ─────────────────────────────────────────────────────────────────────────────

_interrupt_requested = False


def _install_sigint():
    def _handler(sig, frame):
        global _interrupt_requested
        _interrupt_requested = True
        print("\n\n[!] Ctrl-C received — will exit cleanly after the current step.\n")
    signal.signal(signal.SIGINT, _handler)


# ─────────────────────────────────────────────────────────────────────────────
# Estimator (uses pilot medians if provided)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Estimator:
    prompt_tokens_median:     Optional[int]   = None
    completion_tokens_median: Optional[int]   = None
    wall_seconds_median:      Optional[float] = None

    def per_step_tokens(self, grounding: str) -> tuple[int, int]:
        if self.prompt_tokens_median and self.completion_tokens_median:
            return self.prompt_tokens_median, self.completion_tokens_median
        return (DEFAULT_PROMPT_TOKENS.get(grounding, 25_000),
                DEFAULT_COMPLETION_TOKENS)

    def per_step_wall(self, grounding: str) -> float:
        if self.wall_seconds_median:
            return float(self.wall_seconds_median)
        return DEFAULT_WALL_SECONDS.get(grounding, 12.0)

    def cost_of(self, model_id: str, prompt_tok: int, comp_tok: int) -> float:
        in_p, out_p = PRICE_TABLE.get(model_id, (0.0, 0.0))
        return (prompt_tok * in_p + comp_tok * out_p) / 1_000_000

    @classmethod
    def from_pilot_dir(cls, pilot_dir: Optional[Path]) -> "Estimator":
        est = cls()
        if not pilot_dir or not pilot_dir.exists():
            return est
        meta_paths = list(pilot_dir.rglob("run_meta.json"))
        p_toks, c_toks, walls = [], [], []
        for mp in meta_paths:
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                continue
            for s in meta.get("steps", []):
                if s.get("feasible") and s.get("finish_reason") == "stop":
                    if s.get("prompt_tokens"):     p_toks.append(s["prompt_tokens"])
                    if s.get("completion_tokens"): c_toks.append(s["completion_tokens"])
                    if s.get("wall_seconds"):      walls.append(s["wall_seconds"])
        if p_toks: est.prompt_tokens_median = int(st.median(p_toks))
        if c_toks: est.completion_tokens_median = int(st.median(c_toks))
        if walls:  est.wall_seconds_median = round(st.median(walls), 2)
        return est


# ─────────────────────────────────────────────────────────────────────────────
# Menu helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(msg: str, default: str = "") -> str:
    try:
        raw = input(msg).strip()
    except EOFError:
        raw = ""
    return raw or default


def _confirm(msg: str, default_yes: bool = False) -> bool:
    prompt = f"{msg} [{'Y/n' if default_yes else 'y/N'}] > "
    ans = _prompt(prompt).lower()
    if not ans:
        return default_yes
    return ans[0] == "y"


def _print_arm_summary(arm: str, runs: List[Dict[str, Any]], done: int,
                       estimator: Estimator) -> tuple[int, float, float]:
    """Return (total_calls, est_cost_usd, est_wall_hours) for this arm."""
    remaining_runs = [r for r in runs if not r.get("_completed")]
    total_calls = sum(int(r["N"]) - int(r.get("_resume_step", 0))
                      for r in remaining_runs)
    cost = 0.0
    wall = 0.0
    for r in remaining_runs:
        calls = int(r["N"]) - int(r.get("_resume_step", 0))
        pt, ct = estimator.per_step_tokens(r["grounding"])
        cost += estimator.cost_of(r["model_id"], pt * calls, ct * calls)
        wall += estimator.per_step_wall(r["grounding"]) * calls
    return total_calls, cost, wall / 3600.0


def _describe_run(run: Dict[str, Any]) -> str:
    return (f"{run['provider']}:{run['model_id']} "
            f"corpus={run['corpus']} N={run['N']} {run['grounding']} "
            f"seed={run['seed']} rep={run['repetition']}")


# ─────────────────────────────────────────────────────────────────────────────
# LLM/counter/retrieval builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm(run: Dict[str, Any]) -> GenerationLLM:
    kwargs: Dict[str, Any] = {}
    prov = run["provider"]
    if prov == "ollama_cloud":
        kwargs["api_key_env"]  = "OLLAMA_API_KEY"
        kwargs["api_key_file"] = "api_keys/ollama_key.txt"
        kwargs["host"]         = "https://ollama.com"
    if prov == "openai":
        kwargs["api_key_env"]  = "OPENAI_API_KEY"
        kwargs["api_key_file"] = "api_keys/openai_key.txt"
    return make_client(provider=prov, model_id=run["model_id"], **kwargs)


def _make_counter(run: Dict[str, Any]):
    prov = run["provider"]
    if prov == "openai":
        try:    return TiktokenCounter(model_id=run["model_id"])
        except: return UniversalCounter()
    if prov == "ollama_cloud":
        return OllamaTokenCounter(model_id=run["model_id"])
    return UniversalCounter()


# ─────────────────────────────────────────────────────────────────────────────
# The arm runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0
    runs_completed: int = 0
    runs_admissible: int = 0
    runs_terminal_failed: int = 0
    runs_validation_failed: int = 0
    runs_skipped: int = 0


def _run_arm(
    arm: str,
    runs: List[Dict[str, Any]],
    *,
    campaign_id: str,
    results_root: Path,
    prompt_template_path: Path,
    metamodel_xsd_text: str,
    orderings_json: Path,
    estimator: Estimator,
    do_validate: bool = True,
    force: bool = False,
) -> RunTotals:
    """Execute every run in ``runs`` for one arm. Verbose per-step output."""
    totals = RunTotals()
    n_runs = len(runs)
    print(f"\n═════════════════════════════════════════════════════════")
    print(f"  Executing arm: {arm}  ({n_runs} runs)")
    print(f"═════════════════════════════════════════════════════════")

    for i, run in enumerate(runs, start=1):
        if _interrupt_requested:
            print("[!] interrupt requested — stopping before next run.")
            totals.runs_skipped += (n_runs - i + 1)
            break

        header = f"[{i:>3d}/{n_runs}] {_describe_run(run)}"
        print(f"\n{header}")

        run_cfg = _run_config_from_row(run)

        # Progress callbacks: one line per step, one line at end.
        step_start_wall = {"t": 0.0}

        def _on_start(cfg: RunConfig, step_idx: int, N: int, doc_ids: List[str]):
            step_start_wall["t"] = time.time()
            print(f"    step {step_idx+1:>2d}/{N} start "
                  f"docs={','.join(doc_ids[:3])}{'…' if len(doc_ids)>3 else ''} ",
                  end="", flush=True)

        def _on_done(cfg: RunConfig, step_idx: int, rec: StepRecord):
            wall = rec.wall_seconds
            if rec.error:
                print(f"→ ERROR ({wall:.1f}s): {rec.error}")
            elif not rec.feasible:
                print(f"→ infeasible ({wall:.1f}s)  assembled_tok={rec.assembled_tokens}")
            else:
                tokens = (rec.prompt_tokens or 0) + (rec.completion_tokens or 0)
                print(f"→ ok ({wall:.1f}s)  chunks={rec.n_chunks} "
                      f"tok={rec.prompt_tokens}/{rec.completion_tokens} "
                      f"finish={rec.finish_reason}")

        # Instantiate services
        try:
            llm = _make_llm(run)
        except Exception as exc:
            print(f"    LLM init error: {exc}")
            totals.runs_terminal_failed += 1
            continue
        counter = _make_counter(run)
        retrieval_service = None
        if run["grounding"] == "rag":
            retrieval_service = RetrievalService.from_config(
                corpus=run["corpus"],
                chroma_root=REPO / "data/chroma",
                config_path=REPO / "config/experiment.yaml",
            )

        run_t0 = time.time()
        claim_path = (results_root / ".campaign_locks" / campaign_id
                      / f"{run_cfg.run_id()}.lock")
        provider_claim_path = (results_root / ".provider_locks"
                               / f"{run_cfg.provider}.lock")
        try:
            with RunClaim(claim_path):
                # Provider-wide blocking lock enforces the frozen concurrency
                # of one request stream per account, even if users launch
                # additional model-filtered processes by accident.
                with RunClaim(provider_claim_path, blocking=True):
                    if run.get("_recover_provider_error"):
                        _archive_provider_error(run, results_root)
                    if run.get("_resume_provider_failure"):
                        _archive_resume_checkpoint(run, results_root)
                    result = run_generation(
                        config=run_cfg,
                        llm=llm,
                        orderings_json=orderings_json,
                        chunks_jsonl=REPO / f"data/processed/{run['corpus']}/chunks.jsonl",
                        prompt_template_path=prompt_template_path,
                        metamodel_xsd_text=metamodel_xsd_text,
                        retrieval_service=retrieval_service,
                        results_root=results_root,
                        token_counter=counter,
                        force=force,
                        resume_provider_failure=bool(run.get("_resume_provider_failure")),
                        on_step_start=_on_start,
                        on_step_done=_on_done,
                    )
        except RunClaimed as exc:
            print(f"    claimed elsewhere — skipping: {exc}")
            totals.runs_skipped += 1
            continue
        except RunAlreadyExists:
            print("    existing incomplete artefacts — skipping; use --force-runs "
                  "to replace this exact run")
            totals.runs_skipped += 1
            continue
        except Exception as exc:
            print(f"    RUN FAILED: {type(exc).__name__}: {exc}")
            totals.runs_terminal_failed += 1
            continue

        # Run-level accounting
        invocation_steps = result.steps[result.invocation_start_step:]
        run_prompt = sum(s.prompt_tokens or 0 for s in invocation_steps)
        run_comp   = sum(s.completion_tokens or 0 for s in invocation_steps)
        run_wall   = round(time.time() - run_t0, 1)
        run_cost   = estimator.cost_of(run["model_id"], run_prompt, run_comp)
        n_ok_steps = sum(1 for s in result.steps if s.carry_forward)

        totals.prompt_tokens     += run_prompt
        totals.completion_tokens += run_comp
        totals.wall_seconds      += run_wall
        totals.cost_usd          += run_cost
        if result.completed:
            totals.runs_completed += 1
        else:
            totals.runs_terminal_failed += 1

        print(f"    run done: steps_ok={n_ok_steps}/{run_cfg.N}  completed={result.completed}  "
              f"tok_in={run_prompt}  tok_out={run_comp}  wall={run_wall}s  "
              f"cost=${run_cost:.4f}  (running: ${totals.cost_usd:.2f})")

        # Per-run validation
        if do_validate:
            rep = validate_run(result.paths.root, repo_root=REPO)
            if rep.admissible:
                totals.runs_admissible += 1
            else:
                print(f"    [validator] {rep.n_errors} errors, {rep.n_warnings} warnings — see run dir")
                totals.runs_validation_failed += 1

    return totals


def _print_totals(arm: str, totals: RunTotals) -> None:
    print(f"\n─── arm complete: {arm} ───")
    print(f"  runs:      completed={totals.runs_completed}  "
          f"admissible={totals.runs_admissible}  "
          f"terminal_failed={totals.runs_terminal_failed}  "
          f"validation_failed={totals.runs_validation_failed}  "
          f"skipped={totals.runs_skipped}")
    print(f"  wall:      {totals.wall_seconds/60.0:.1f} min")
    print(f"  tokens:    in={totals.prompt_tokens:,}  out={totals.completion_tokens:,}")
    print(f"  cost:      ${totals.cost_usd:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Menu
# ─────────────────────────────────────────────────────────────────────────────

def _group_runs_by_arm(matrix: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in matrix["runs"]:
        arm = r.get("extra", {}).get("arm", "unknown")
        by_arm.setdefault(arm, []).append(r)
    return by_arm


def _recoverable_provider_error(meta: Dict[str, Any]) -> bool:
    if meta.get("completed") or meta.get("terminal_status") != "provider_error":
        return False
    steps = meta.get("steps") or []
    error = str(steps[-1].get("error", "")) if steps else ""
    # Intentionally exclude quota/auth errors and model-output failures.
    # The loop persists transport exceptions as "ExceptionClass: message".
    # Match the type, not generic words such as "connection" in an error body.
    return (error.startswith("ChunkedEncodingError:")
            or any(f"HTTP {code}:" in error for code in (408, 500, 502, 503, 504)))


def _archive_provider_error(run: Dict[str, Any], results_root: Path) -> Path:
    """Called under both run/provider locks; preserve the full original attempt."""
    cfg = _run_config_from_row(run)
    paths = RunPaths.for_run(
        results_root=results_root, campaign_id=run["campaign_id"],
        corpus=run["corpus"], config_hash=cfg.config_hash(), run_id=cfg.run_id(),
    )
    meta = json.loads(paths.run_meta.read_text())
    if not _recoverable_provider_error(meta):
        raise RunAlreadyExists("recovery target changed; refusing to overwrite")
    archive = (results_root / "recovery_archive" / run["campaign_id"]
               / run["corpus"] / cfg.run_id() / uuid.uuid4().hex)
    archive.parent.mkdir(parents=True, exist_ok=True)
    paths.root.rename(archive)
    print(f"    original provider-error attempt archived: {archive}", flush=True)
    return archive


def _archive_resume_checkpoint(run: Dict[str, Any], results_root: Path) -> Path:
    """Copy a run before continuing it in place from a provider failure."""
    cfg = _run_config_from_row(run)
    paths = RunPaths.for_run(
        results_root=results_root, campaign_id=run["campaign_id"],
        corpus=run["corpus"], config_hash=cfg.config_hash(), run_id=cfg.run_id(),
    )
    archive = (results_root / "recovery_archive" / run["campaign_id"]
               / run["corpus"] / cfg.run_id() / uuid.uuid4().hex)
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(paths.root, archive)
    print(f"    pre-resume checkpoint archived: {archive}", flush=True)
    return archive


def _mark_completed(runs: List[Dict[str, Any]], results_root: Path,
                    *, retry_provider_errors: bool = False) -> None:
    """Mark runs with a persisted terminal outcome, successful or otherwise.

    A malformed, truncated, infeasible, or provider-failed repetition remains
    part of the experimental denominator and must not be silently rerun until
    it succeeds. Artefacts from an interrupted process without terminal
    metadata remain unresolved and require an explicit ``--force-runs`` restart.
    """
    for r in runs:
        run_cfg = _run_config_from_row(r)
        paths = RunPaths.for_run(
            results_root=results_root,
            campaign_id=r["campaign_id"],
            corpus=r["corpus"],
            config_hash=run_cfg.config_hash(),
            run_id=run_cfg.run_id(),
        )
        terminal_observed = False
        meta = {}
        if paths.run_meta.exists():
            try:
                meta = json.loads(paths.run_meta.read_text())
                terminal_observed = bool(
                    meta.get("completed") is True
                    or meta.get("terminal_status")
                    or meta.get("failed_step") is not None
                )
            except (OSError, json.JSONDecodeError):
                terminal_observed = False
        r["_completed"] = terminal_observed
        if retry_provider_errors:
            selected = _recoverable_provider_error(meta)
            r["_recover_provider_error"] = selected
            r["_completed"] = not selected


def _menu(matrix: Dict[str, Any], by_arm: Dict[str, List[Dict[str, Any]]],
          estimator: Estimator) -> Optional[str]:
    """Show the menu and return the selected arm name, 'ALL', or None to quit."""
    total_runs = sum(len(v) for v in by_arm.values())
    total_done = sum(1 for r in matrix["runs"] if r.get("_completed"))

    print("\n═════════════════════════════════════════════════════════")
    print(f"  iFS 2027 Campaign — Interactive")
    print(f"  matrix: {matrix.get('campaign_id')}  "
          f"({'enabled-only' if matrix.get('only_enabled') else 'full plan'})")
    print("═════════════════════════════════════════════════════════")
    print(f"  runs total:     {total_runs}")
    print(f"  runs completed: {total_done}")
    print(f"  runs remaining: {total_runs - total_done}")
    print()
    arm_list = [a for a in ARM_ORDER if a in by_arm] + \
               [a for a in by_arm if a not in ARM_ORDER]
    for idx, arm in enumerate(arm_list, start=1):
        runs = by_arm[arm]
        done = sum(1 for r in runs if r.get("_completed"))
        calls, cost, wall_h = _print_arm_summary(arm, runs, done, estimator)
        print(f"  {idx:>2d}) {arm:<26s} "
              f"remaining_calls={calls:>5d}  "
              f"done={done}/{len(runs)}  "
              f"est_cost=${cost:>7.2f}  "
              f"est_wall={wall_h:.1f}h")
    print()
    print(f"  A) run all in order")
    print(f"  Q) quit")
    print()
    ans = _prompt("Select > ").upper()
    if ans in ("Q", ""):    return None
    if ans == "A":          return "ALL"
    try:
        i = int(ans)
        if 1 <= i <= len(arm_list):
            return arm_list[i - 1]
    except ValueError:
        pass
    print(f"invalid selection: {ans!r}")
    return _menu(matrix, by_arm, estimator)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=Path, required=True,
                   help="Path to a run_matrix.json (usually the _enabled variant).")
    p.add_argument("--pilot-dir", type=Path, default=REPO / "results/pilot-2026-09-22",
                   help="Pilot run directory whose medians drive cost/wall estimates.")
    p.add_argument("--results-root", type=Path, default=REPO / "results")
    p.add_argument("--arm", default=None,
                   help="If set, run just this arm and exit (still prompts confirmation unless --yes).")
    p.add_argument("--lane", choices=("all", "open_weight", "astra"), default="all",
                   help="Run only rows assigned to this disjoint execution lane.")
    p.add_argument("--model-key", action="append", default=[],
                   help="Further restrict to a model key; repeat for multiple models.")
    p.add_argument("--run-all", action="store_true",
                   help="Non-interactive: iterate every arm in order.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip confirmation prompts. Use with care.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the per-run validator.")
    p.add_argument("--force-runs", action="store_true",
                   help="Overwrite any existing run outputs. Otherwise resume-skip.")
    p.add_argument("--retry-provider-errors", action="store_true",
                   help="Only recover recorded HTTP 408/500/502/503/504 or "
                        "ChunkedEncodingError provider failures; "
                        "archive originals and restart from step 1. Requires --arm.")
    p.add_argument("--resume-run",
                   help="Resume one run_id at its terminal provider-failure step; "
                        "requires --arm and preserves prior successful steps.")
    return p.parse_args()


def _filter_matrix_rows(matrix: Dict[str, Any], *, lane: str,
                        model_keys: List[str]) -> None:
    known_models = set(matrix.get("models", {}))
    unknown = set(model_keys) - known_models
    if unknown:
        raise ValueError(f"unknown --model-key values: {sorted(unknown)}")
    rows = matrix.get("runs", [])
    if lane != "all":
        rows = [r for r in rows if r.get("extra", {}).get("lane") == lane]
    if model_keys:
        selected = set(model_keys)
        rows = [r for r in rows if r.get("extra", {}).get("model_key") in selected]
    matrix["runs"] = rows
    matrix["execution_filter"] = {"lane": lane, "model_keys": sorted(model_keys)}
    matrix["total_runs"] = len(rows)
    matrix["total_calls"] = sum(int(r["N"]) for r in rows)
    matrix["per_arm_calls"] = {}
    matrix["per_model_calls"] = {}
    matrix["per_lane_calls"] = {}
    matrix["per_lane_runs"] = {}
    for row in rows:
        arm = row.get("extra", {}).get("arm", "unknown")
        model = row.get("extra", {}).get("model_key", "unknown")
        row_lane = row.get("extra", {}).get("lane", "unknown")
        calls = int(row["N"])
        matrix["per_arm_calls"][arm] = matrix["per_arm_calls"].get(arm, 0) + calls
        matrix["per_model_calls"][model] = matrix["per_model_calls"].get(model, 0) + calls
        matrix["per_lane_calls"][row_lane] = matrix["per_lane_calls"].get(row_lane, 0) + calls
        matrix["per_lane_runs"][row_lane] = matrix["per_lane_runs"].get(row_lane, 0) + 1


def _write_campaign_snapshots(*, matrix: Dict[str, Any], matrix_path: Path,
                              results_root: Path, lane: str) -> None:
    """Persist immutable inputs needed to reconstruct each execution lane."""
    campaign_id = str(matrix.get("campaign_id", "unknown"))
    snapshot_dir = results_root / campaign_id / "protocol"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = yaml.safe_load((REPO / "config/experiment.yaml").read_text())
    cfg_path = snapshot_dir / "resolved_experiment.json"
    if cfg_path.exists():
        previous = json.loads(cfg_path.read_text())
        if previous != resolved_cfg:
            raise RuntimeError(
                f"campaign protocol snapshot differs from current config: {cfg_path}"
            )
    else:
        atomic_write_json(cfg_path, resolved_cfg)
    atomic_write_json(snapshot_dir / f"run_matrix_{lane}.json", matrix)
    atomic_write_text(snapshot_dir / "protocol.sha256",
                      (REPO / "data/frozen/protocol.sha256").read_text())
    atomic_write_text(snapshot_dir / f"matrix_source_{lane}.txt",
                      str(matrix_path.expanduser().resolve()) + "\n")


def _verify_matrix_hashes(matrix: Dict[str, Any]) -> None:
    """Reject a manifest generated from a different frozen protocol state."""
    expected = matrix.get("hashes", {})
    files = {
        "experiment_config_hash": REPO / "config/experiment.yaml",
        "prompt_template_hash": REPO / "prompts/fm_prompt_template.txt",
        "metamodel_hash": REPO / "prompts/feature-model-schema.xsd",
        "chunks_federation": REPO / "data/processed/federation/chunks.jsonl",
        "chunks_repair": REPO / "data/processed/repair/chunks.jsonl",
    }
    for key, path in files.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(key) != actual:
            raise ValueError(
                f"stale matrix: {key}={expected.get(key)!r} does not match {path}"
            )


def main() -> int:
    _install_sigint()
    args = _cli()
    if args.retry_provider_errors and (not args.arm or args.force_runs or args.run_all):
        print("--retry-provider-errors requires --arm and cannot use --force-runs/--run-all")
        return 2
    if args.resume_run and (not args.arm or args.force_runs
                            or args.retry_provider_errors or args.run_all):
        print("--resume-run requires --arm and cannot use --force-runs, "
              "--retry-provider-errors, or --run-all")
        return 2

    matrix = json.loads(args.matrix.read_text())
    try:
        _verify_matrix_hashes(matrix)
    except ValueError as exc:
        print(exc)
        return 2
    try:
        _filter_matrix_rows(matrix, lane=args.lane, model_keys=args.model_key)
    except ValueError as exc:
        print(f"matrix filter error: {exc}")
        return 2
    if not matrix["runs"]:
        print("matrix filter selected zero runs")
        return 2
    campaign_id = matrix.get("campaign_id", "unknown")
    selected_models = sorted({r.get("extra", {}).get("model_key") for r in matrix["runs"]})
    print(f"loaded matrix: {args.matrix}  campaign_id={campaign_id}  "
          f"lane={args.lane}  models={','.join(selected_models)}")
    try:
        _write_campaign_snapshots(matrix=matrix, matrix_path=args.matrix,
                                  results_root=args.results_root, lane=args.lane)
    except Exception as exc:
        print(f"protocol snapshot error: {exc}")
        return 2
    _mark_completed(matrix["runs"], args.results_root,
                    retry_provider_errors=args.retry_provider_errors)
    if args.resume_run:
        matches = [r for r in matrix["runs"]
                   if _run_config_from_row(r).run_id() == args.resume_run
                   and r.get("extra", {}).get("arm") == args.arm]
        if len(matches) != 1:
            print(f"--resume-run matched {len(matches)} rows; expected exactly one")
            return 2
        target = matches[0]
        target_cfg = _run_config_from_row(target)
        target_paths = RunPaths.for_run(
            results_root=args.results_root,
            campaign_id=target["campaign_id"], corpus=target["corpus"],
            config_hash=target_cfg.config_hash(), run_id=target_cfg.run_id(),
        )
        try:
            target_meta = json.loads(target_paths.run_meta.read_text())
            target["_resume_step"] = int(target_meta["failed_step"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"--resume-run cannot read its checkpoint metadata: {exc}")
            return 2
        target["_completed"] = False
        target["_resume_provider_failure"] = True
        matrix["runs"] = [target]
    estimator = Estimator.from_pilot_dir(args.pilot_dir)
    if estimator.prompt_tokens_median:
        print(f"cost model:  pilot medians  "
              f"(prompt_tok={estimator.prompt_tokens_median}, "
              f"comp_tok={estimator.completion_tokens_median}, "
              f"wall={estimator.wall_seconds_median}s)")
    else:
        print(f"cost model:  DEFAULT_* fallback (no pilot data found under {args.pilot_dir})")

    by_arm = _group_runs_by_arm(matrix)
    prompt_template_path = REPO / "prompts/fm_prompt_template.txt"
    metamodel_xsd_text   = (REPO / "prompts/feature-model-schema.xsd").read_text()
    orderings_json       = REPO / "data/orderings.json"

    def _run(arm: str, force_confirm: bool = True) -> Optional[RunTotals]:
        runs = by_arm[arm]
        remaining = [r for r in runs if not r.get("_completed")]
        calls, cost, wall_h = _print_arm_summary(arm, runs, 0, estimator)
        print(f"\n─── {arm} ───")
        print(f"  runs remaining: {len(remaining)}/{len(runs)}")
        print(f"  planned calls:  {calls}")
        print(f"  est. cost:      ${cost:.2f}")
        print(f"  est. wall:      {wall_h:.1f} h")
        if force_confirm and not args.yes:
            if not _confirm("Proceed?", default_yes=False):
                print("  → skipped.")
                return None
        totals = _run_arm(
            arm, remaining,
            campaign_id=campaign_id,
            results_root=args.results_root,
            prompt_template_path=prompt_template_path,
            metamodel_xsd_text=metamodel_xsd_text,
            orderings_json=orderings_json,
            estimator=estimator,
            do_validate=not args.no_validate,
            force=args.force_runs,
        )
        _print_totals(arm, totals)
        return totals

    def _exit_code(totals: Optional[RunTotals]) -> int:
        if totals is None:
            return 0
        return 1 if (totals.runs_terminal_failed
                     or totals.runs_validation_failed
                     or totals.runs_skipped) else 0

    if args.arm:
        if args.arm not in by_arm:
            print(f"arm {args.arm!r} not present in matrix. Available: {list(by_arm)}")
            return 2
        return _exit_code(_run(args.arm))

    if args.run_all:
        exit_code = 0
        for arm in ARM_ORDER:
            if arm in by_arm:
                exit_code = max(
                    exit_code,
                    _exit_code(_run(arm, force_confirm=False if args.yes else True)),
                )
                if _interrupt_requested:
                    exit_code = 130
                    break
        return exit_code

    while True:
        pick = _menu(matrix, by_arm, estimator)
        if pick is None:
            print("bye.")
            return 0
        if pick == "ALL":
            for arm in ARM_ORDER:
                if arm in by_arm:
                    _run(arm)
                    if _interrupt_requested: break
                    if not _confirm(f"\nContinue to next arm?", default_yes=True):
                        break
            _mark_completed(matrix["runs"], args.results_root)
        else:
            _run(pick)
            _mark_completed(matrix["runs"], args.results_root)


if __name__ == "__main__":
    raise SystemExit(main())
