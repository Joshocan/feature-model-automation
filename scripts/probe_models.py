#!/usr/bin/env python3
"""Phase 7.2 + 7.4 — probe each configured campaign model.

For every model in ``config/experiment.yaml.generation_models``:

  1. One tiny generation call (~50 input tokens, request 32 output tokens)
  2. Report: reachable/not, wall_seconds, prompt_tokens, completion_tokens,
     finish_reason, and whether reported prompt_tokens ~ matches our
     universal char/4 estimate for this text
  3. Repeat a short burst (default 5 calls in quick succession) to probe
     rate-limit / backoff behaviour (Phase 6.V4)

Emits ``results/pilot-<date>/model_probe.json`` with per-model records.
Exits non-zero if any model is unreachable.

Usage:
  python scripts/probe_models.py
  python scripts/probe_models.py --burst 10           # more aggressive backoff probe
  python scripts/probe_models.py --no-burst           # skip 7.4
  python scripts/probe_models.py --only glm_5_3_flash
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.generation import GenerationRequest, make_client  # noqa: E402
from fame.generation.token_budget import universal_estimate  # noqa: E402


PROBE_PROMPT = (
    "Reply with exactly the string 'OK-<n>' where <n> is the integer 1, "
    "then stop."
)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--burst", type=int, default=5,
                   help="Backoff probe: N quick-fire calls to trigger 429/backoff.")
    p.add_argument("--no-burst", action="store_true")
    p.add_argument("--only", default=None,
                   help="Only probe this generation_models key (e.g. 'glm_5_3_flash').")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: results/pilot-<date>/model_probe.json)")
    return p.parse_args()


def _client_kwargs(cfg: dict) -> dict:
    provider = cfg["provider"]
    kwargs: dict = {}
    if provider == "ollama_cloud":
        kwargs["api_key_env"] = "OLLAMA_API_KEY"
        kwargs["api_key_file"] = "api_keys/ollama_key.txt"
        endpoint = cfg.get("serving_endpoint")
        if endpoint:
            kwargs["host"] = endpoint
    if provider == "openai":
        kwargs["api_key_env"] = "OPENAI_API_KEY"
        kwargs["api_key_file"] = "api_keys/openai_key.txt"
    return kwargs


def probe_one(name: str, cfg: dict, *, burst: int = 0) -> dict:
    """Return a probe record for one model."""
    record: dict = {"name": name, **cfg, "reachable": False, "error": None,
                    "single_call": None, "burst": []}
    try:
        llm = make_client(provider=cfg["provider"], model_id=cfg["model_id"],
                          **_client_kwargs(cfg))
    except Exception as exc:
        record["error"] = f"client_init: {type(exc).__name__}: {exc}"
        return record

    # 1. Single small call
    est_prompt_tokens = universal_estimate(PROBE_PROMPT)
    t0 = time.time()
    try:
        resp = llm.generate(GenerationRequest(
            prompt=PROBE_PROMPT,
            max_output_tokens=32,
            temperature=0.0,
            reasoning_effort=cfg.get("reasoning_effort"),
        ))
        record["reachable"] = True
        record["single_call"] = {
            "wall_seconds":     resp.wall_seconds,
            "prompt_tokens":    resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "finish_reason":    resp.finish_reason,
            "text_preview":     resp.text[:80],
            "tokenizer_delta":  (resp.prompt_tokens - est_prompt_tokens)
                                if resp.prompt_tokens is not None else None,
        }
    except Exception as exc:
        record["error"] = f"single_call: {type(exc).__name__}: {exc}"
        return record

    # 2. Burst — quick-fire calls; watch retry/backoff kick in
    if burst > 0:
        for i in range(burst):
            t0 = time.time()
            try:
                r = llm.generate(GenerationRequest(prompt=PROBE_PROMPT,
                                                    max_output_tokens=16,
                                                    temperature=0.0))
                record["burst"].append({
                    "i":              i,
                    "wall_seconds":   r.wall_seconds,
                    "finish_reason":  r.finish_reason,
                    "ok":             True,
                })
            except Exception as exc:
                record["burst"].append({
                    "i":       i,
                    "ok":      False,
                    "error":   f"{type(exc).__name__}: {exc}",
                    "wall_seconds": round(time.time() - t0, 2),
                })
    return record


def main() -> int:
    args = _cli()
    cfg = yaml.safe_load((REPO / "config/experiment.yaml").read_text())
    models = cfg["generation_models"]
    if args.only:
        if args.only not in models:
            print(f"unknown model key: {args.only!r}. Available: {list(models)}")
            return 2
        models = {args.only: models[args.only]}

    out_path = Path(args.out) if args.out else (
        REPO / f"results/pilot-{date.today().isoformat()}/model_probe.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"===== model probe =====")
    records: list[dict] = []
    burst = 0 if args.no_burst else args.burst
    any_unreachable = False
    for name, mcfg in models.items():
        print(f"\n[{name}] provider={mcfg['provider']}  model_id={mcfg['model_id']}")
        rec = probe_one(name, mcfg, burst=burst)
        records.append(rec)
        if not rec["reachable"]:
            print(f"  UNREACHABLE — {rec.get('error')}")
            any_unreachable = True
            continue
        s = rec["single_call"]
        print(f"  ✓ reachable  wall={s['wall_seconds']}s  "
              f"prompt_tokens={s['prompt_tokens']}  "
              f"completion_tokens={s['completion_tokens']}  "
              f"finish={s['finish_reason']}")
        if s.get("tokenizer_delta") is not None:
            print(f"  tokenizer_delta = provider - char/4  =  {s['tokenizer_delta']}")
        print(f"  text: {s['text_preview']!r}")
        if burst > 0:
            wall = [b["wall_seconds"] for b in rec["burst"]]
            ok = sum(1 for b in rec["burst"] if b.get("ok"))
            print(f"  burst({burst}): {ok}/{burst} ok, wall_range={min(wall):.2f}s..{max(wall):.2f}s")

    with open(out_path, "w") as fh:
        json.dump({"date": date.today().isoformat(),
                   "records": records}, fh, indent=2)
    print(f"\nWrote {out_path}")
    return 1 if any_unreachable else 0


if __name__ == "__main__":
    raise SystemExit(main())
