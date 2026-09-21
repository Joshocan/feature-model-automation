#!/usr/bin/env python3
"""Phase 7.3 — extrapolate campaign cost + wall time from pilot data.

Reads:
  - results/<campaign_id>/run_matrix.json   (Phase 7.5 output)
  - results/pilot-<date>/model_probe.json   (Phase 7.2 output; optional)
  - results/pilot-<date>/<run_id>/run_meta.json (Phase 7.1 output; optional)

Combines with a static per-provider price table (see PRICE_TABLE below) to
produce:

  - Total prompt + completion tokens per model
  - Estimated USD per model
  - Total campaign cost
  - Wall-clock estimate (sum of per-step median wall_seconds × steps)

Usage:
  python scripts/estimate_cost.py --matrix results/ifs-2027/run_matrix.json
  python scripts/estimate_cost.py --matrix ... --pilot results/pilot-.../run_id/
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]


# Per-1M-token USD prices — placeholder table; update with real pricing.
# Format: {model_id: (input_$_per_1M, output_$_per_1M)}
PRICE_TABLE: Dict[str, tuple[float, float]] = {
    # OpenAI reasoning tier (representative; adjust when gpt-6-astra pricing is known)
    "gpt-6-astra":        (10.00, 40.00),
    # Ollama Pro cloud — subscription; effective marginal cost near zero.
    "glm-5.3-flash":       (0.00,  0.00),
    "deepseek-v4.1-flash": (0.00,  0.00),
    # Placeholders for common substitutes
    "gpt-oss:120b-cloud":  (0.00,  0.00),
}


# Default assumptions when no pilot data is available.
DEFAULT_PROMPT_TOKENS_PER_STEP = {
    "rag":    25_000,     # ~k_step chunks + previous_model + invariant blocks
    "nonrag": 100_000,    # entire batch's chunks + invariant blocks
}
DEFAULT_COMPLETION_TOKENS_PER_STEP = 4_000

# Universal fallback wall time if we can't derive from pilot.
DEFAULT_WALL_SECONDS_PER_STEP = {
    "rag":    5.0,
    "nonrag": 12.0,
}


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True, type=Path)
    p.add_argument("--pilot", type=Path, default=None,
                   help="Optional: pilot run directory with run_meta.json.")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_pilot(pilot_dir: Path) -> Dict[str, Any] | None:
    if not pilot_dir:
        return None
    meta_path = pilot_dir / "run_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    steps = [s for s in meta.get("steps", []) if s.get("feasible")]
    if not steps:
        return None
    prompt_toks = [s["prompt_tokens"] for s in steps if s.get("prompt_tokens")]
    comp_toks   = [s["completion_tokens"] for s in steps if s.get("completion_tokens")]
    walls       = [s["wall_seconds"] for s in steps if s.get("wall_seconds")]
    return {
        "prompt_tokens_median":     int(st.median(prompt_toks))   if prompt_toks else None,
        "completion_tokens_median": int(st.median(comp_toks))     if comp_toks   else None,
        "wall_seconds_median":      round(st.median(walls), 2)     if walls       else None,
        "n_steps":                  len(steps),
        "config":                   meta.get("config", {}),
    }


def _pick_step_tokens(run: Dict[str, Any], pilot: Dict[str, Any] | None) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) for one step of ``run``."""
    if pilot and pilot.get("prompt_tokens_median") and pilot.get("completion_tokens_median"):
        return pilot["prompt_tokens_median"], pilot["completion_tokens_median"]
    grounding = run.get("grounding", "rag")
    return (
        DEFAULT_PROMPT_TOKENS_PER_STEP.get(grounding, 25_000),
        DEFAULT_COMPLETION_TOKENS_PER_STEP,
    )


def _pick_wall(run: Dict[str, Any], pilot: Dict[str, Any] | None) -> float:
    if pilot and pilot.get("wall_seconds_median"):
        return float(pilot["wall_seconds_median"])
    return DEFAULT_WALL_SECONDS_PER_STEP.get(run.get("grounding", "rag"), 5.0)


def estimate(matrix: Dict[str, Any], pilot: Dict[str, Any] | None) -> Dict[str, Any]:
    per_model: Dict[str, Dict[str, float]] = {}
    per_arm:   Dict[str, Dict[str, float]] = {}
    grand_prompt = 0
    grand_completion = 0
    grand_wall = 0.0
    grand_cost = 0.0

    for run in matrix["runs"]:
        model_id = run["model_id"]
        arm      = run.get("extra", {}).get("arm", "unknown")
        N        = int(run["N"])

        p_tok, c_tok = _pick_step_tokens(run, pilot)
        wall = _pick_wall(run, pilot)

        run_prompt     = p_tok * N
        run_completion = c_tok * N
        run_wall       = wall * N

        in_price, out_price = PRICE_TABLE.get(model_id, (0.0, 0.0))
        run_cost = (run_prompt * in_price + run_completion * out_price) / 1_000_000

        # accumulate per model
        m = per_model.setdefault(model_id, {"prompt_tokens": 0, "completion_tokens": 0,
                                             "wall_seconds": 0.0, "cost_usd": 0.0,
                                             "runs": 0, "steps": 0})
        m["prompt_tokens"]     += run_prompt
        m["completion_tokens"] += run_completion
        m["wall_seconds"]      += run_wall
        m["cost_usd"]          += run_cost
        m["runs"]              += 1
        m["steps"]             += N

        # accumulate per arm
        a = per_arm.setdefault(arm, {"runs": 0, "steps": 0, "cost_usd": 0.0})
        a["runs"]      += 1
        a["steps"]     += N
        a["cost_usd"]  += run_cost

        grand_prompt     += run_prompt
        grand_completion += run_completion
        grand_wall       += run_wall
        grand_cost       += run_cost

    return {
        "pilot_used":    pilot is not None,
        "total_runs":    len(matrix["runs"]),
        "total_steps":   grand_prompt // (pilot["prompt_tokens_median"] if pilot and pilot.get("prompt_tokens_median")
                                          else DEFAULT_PROMPT_TOKENS_PER_STEP["rag"]),
        "total_prompt_tokens":     grand_prompt,
        "total_completion_tokens": grand_completion,
        "total_wall_hours":        round(grand_wall / 3600.0, 2),
        "total_cost_usd":          round(grand_cost, 2),
        "per_model": {k: {**v, "cost_usd": round(v["cost_usd"], 2),
                          "wall_seconds": round(v["wall_seconds"], 1)}
                      for k, v in per_model.items()},
        "per_arm":   {k: {**v, "cost_usd": round(v["cost_usd"], 2)}
                      for k, v in per_arm.items()},
    }


def main() -> int:
    args = _cli()
    matrix = json.loads(args.matrix.read_text())
    pilot = load_pilot(args.pilot) if args.pilot else None
    est = estimate(matrix, pilot)

    print(f"===== campaign cost estimate =====")
    print(f"  matrix:              {args.matrix}")
    print(f"  pilot data:          {'YES — using pilot medians' if pilot else 'NO — using DEFAULT_*'}")
    print(f"  total runs:          {est['total_runs']}")
    print(f"  total prompt tokens: {est['total_prompt_tokens']:,}")
    print(f"  total completion:    {est['total_completion_tokens']:,}")
    print(f"  wall time:           {est['total_wall_hours']:.1f} h  (sum, single-threaded)")
    print(f"  total cost:          ${est['total_cost_usd']:,.2f}")
    print()
    print("  per model:")
    for m, v in sorted(est["per_model"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        print(f"    {m:<24s} runs={v['runs']:>4d}  cost=${v['cost_usd']:>10,.2f}  "
              f"wall={v['wall_seconds']/3600:.1f}h")
    print()
    print("  per arm:")
    for a, v in sorted(est["per_arm"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        print(f"    {a:<28s} runs={v['runs']:>4d}  cost=${v['cost_usd']:>10,.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(est, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
