#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fame.evaluation import evaluate_groundedness


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="LLM-judged groundedness/relevance metrics using TruLens.")
    ap.add_argument("--answers", required=True, help="Text file: one answer per line.")
    ap.add_argument("--contexts", required=True, help="Text file: one context block per line (matching answers).")
    ap.add_argument("--prompts", required=True, help="Text file: one prompt/question per line (matching answers).")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic", "gemini"], help="LLM provider for judging.")
    ap.add_argument("--model", default="gpt-4.1", help="Judge model name.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Judge temperature (default 0.0).")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Judge max tokens.")
    ap.add_argument("--api-key-env", default="", help="Override API key env var (otherwise inferred per provider).")
    ap.add_argument("--base-url", default="", help="Optional OpenAI-compatible base URL.")
    ap.add_argument("--out", default="", help="Optional output JSON path; stdout if omitted.")
    return ap.parse_args()


def _read_lines(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def main() -> None:
    args = parse_args()
    answers = _read_lines(args.answers)
    contexts = _read_lines(args.contexts)
    prompts = _read_lines(args.prompts)

    metrics = evaluate_groundedness(
        answers,
        contexts,
        prompts,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key_env=args.api_key_env or None,
        base_url=args.base_url or None,
    )

    payload = metrics.__dict__
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote groundedness metrics to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
