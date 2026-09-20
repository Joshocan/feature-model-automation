#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Sequence


PIPELINE_DIRS = {
    "ss-rag": ("results/rag/ss-rgfm/fm", "results/rag/ss-rgfm"),
    "is-rag": ("results/rag/is-rgfm/fm", "results/rag/is-rgfm"),
    "ss-nonrag": ("results/non_rag/ss-nonrag/fm", "results/non_rag/ss-nonrag"),
    "is-nonrag": ("results/non_rag/is-nonrag/fm", "results/non_rag/is-nonrag"),
}

PIPELINE_DEFAULT_PATTERNS = {
    "ss-rag": ["ss-rgfm_response_*.xml"],
    "is-rag": ["is_rgfm_response_*.final.xml"],
    "ss-nonrag": ["ss_nonrag_response_*.xml"],
    "is-nonrag": ["is_nonrag_response_*.final.xml"],
}


def _collect_candidates(fm_dir: Path, patterns: List[str]) -> List[dict]:
    seen = set()
    candidates = []
    for pattern in patterns:
        for path in sorted(fm_dir.glob(pattern)):
            resolved = path.expanduser().resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            candidates.append(
                {
                    "run_id": resolved.stem,
                    "fm_xml": str(resolved),
                    "meta": "",
                }
            )
    return candidates


def _prompt_choice(title: str, options: List[str]) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    while True:
        raw = input("Select option: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice. Try again.")


def _rank_one(
    *,
    fm_dir: Path,
    outdir: Path,
    gt_path: Path,
    xsd_path: Path | None,
    top_n: int,
    patterns: Sequence[str],
    coverage_cfg: Any,
    TopFMConfig: Any,
    rank_top_fms: Any,
    require_sat: bool,
) -> dict | None:
    outdir.mkdir(parents=True, exist_ok=True)
    candidates = _collect_candidates(fm_dir, list(patterns))
    if not candidates:
        raise FileNotFoundError(f"No FM XML files matched in {fm_dir} for patterns: {list(patterns)}")

    print(f"[rank_top_fm] Pipeline root: {outdir}")
    print(f"[rank_top_fm] FM dir       : {fm_dir}")
    print(f"[rank_top_fm] Candidates   : {len(candidates)}")
    print(f"[rank_top_fm] Patterns     : {list(patterns)}")
    print(f"[rank_top_fm] Top-N        : {top_n}")
    print(f"[rank_top_fm] Require SAT  : {require_sat}")

    manifest = rank_top_fms(
        candidates=candidates,
        pipeline_root=outdir,
        cfg=TopFMConfig(
            top_n=max(0, int(top_n)),
            gt_xml=gt_path,
            xsd_path=xsd_path,
            coverage=coverage_cfg,
            output_subdir=f"top_{max(0, int(top_n))}",
            require_sat=require_sat,
        ),
    )
    if manifest:
        print(f"Ranked {len(candidates)} feature models for {fm_dir.name}.")
        print(f"Patterns used: {list(patterns)}")
        print(f"Top-FM output: {outdir / 'top_fm' / f'top_{max(0, int(top_n))}'}")
    return manifest


def main() -> None:
    print("[rank_top_fm] Starting ranker...")
    print("[rank_top_fm] Loading configuration and evaluation modules. This can take time because sentence-transformers/torch may initialize here.")

    from fame.config.load import load_config
    from fame.evaluation.coverage import CoverageConfig
    from fame.evaluation.top_fm import TopFMConfig, rank_top_fms
    from fame.utils.dirs import build_paths

    cfg_yaml = load_config()
    paths = build_paths()
    default_xsd = paths.specifications / "feature_model_featureide.xsd"
    ap = argparse.ArgumentParser(description="Rank top feature models from an existing FM collection.")
    ap.add_argument("--fm-dir", default="", help="Directory containing FM XML files to evaluate")
    ap.add_argument("--outdir", default="", help="Output root directory; defaults to --fm-dir parent")
    ap.add_argument("--all-pipelines", action="store_true", help="Rank all four standard pipeline FM directories")
    ap.add_argument("--gt-path", default=str(cfg_yaml.evaluation.ground_truth_xml or ""), help="Ground-truth XML")
    ap.add_argument("--xsd-path", default=str(default_xsd), help="FeatureIDE XSD path")
    ap.add_argument("--top-fm", type=int, default=cfg_yaml.outputs.top_fm, help="Top valid FMs to copy")
    ap.add_argument("--glob", action="append", default=[], help="Glob pattern(s) inside --fm-dir")
    ap.add_argument("--coverage-model-name", default=cfg_yaml.evaluation.coverage.model_name, help="SentenceTransformer model for coverage/semantic evaluation")
    ap.add_argument("--coverage-threshold", type=float, default=cfg_yaml.evaluation.coverage.similarity_threshold, help="Coverage similarity threshold")
    ap.add_argument("--coverage-top-k", type=int, default=cfg_yaml.evaluation.coverage.top_k, help="Coverage top-k matches")
    ap.add_argument("--coverage-feature-weight", type=float, default=cfg_yaml.evaluation.coverage.feature_weight, help="Coverage feature-name weight")
    ap.add_argument("--coverage-parent-weight", type=float, default=cfg_yaml.evaluation.coverage.parent_weight, help="Coverage parent-context weight")
    ap.add_argument("--require-sat", action="store_true", help="Require SAT satisfiability for eligibility")
    ap.add_argument("--interactive", action="store_true", help="Choose the FM file filter interactively")
    args = ap.parse_args()

    if not args.all_pipelines and not args.fm_dir:
        raise ValueError("Provide --fm-dir for single-pipeline ranking, or use --all-pipelines.")

    gt_path = Path(args.gt_path).expanduser().resolve() if args.gt_path else None
    if not gt_path or not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth XML not found: {args.gt_path}")

    xsd_path = Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None
    coverage_cfg = CoverageConfig(
        model_name=str(args.coverage_model_name),
        similarity_threshold=float(args.coverage_threshold),
        top_k=int(args.coverage_top_k),
        feature_weight=float(args.coverage_feature_weight),
        parent_weight=float(args.coverage_parent_weight),
    )

    if args.all_pipelines:
        results = {}
        for pipeline_name, (fm_dir_str, outdir_str) in PIPELINE_DIRS.items():
            fm_dir = Path(fm_dir_str).expanduser().resolve()
            outdir = Path(outdir_str).expanduser().resolve()
            if not fm_dir.exists():
                print(f"Skipping {pipeline_name}: FM directory not found: {fm_dir}")
                continue
            print(f"\n[rank_top_fm] ===== Pipeline: {pipeline_name} =====")
            patterns = list(args.glob) if args.glob else PIPELINE_DEFAULT_PATTERNS[pipeline_name]
            manifest = _rank_one(
                fm_dir=fm_dir,
                outdir=outdir,
                gt_path=gt_path,
                xsd_path=xsd_path,
                top_n=args.top_fm,
                patterns=patterns,
                coverage_cfg=coverage_cfg,
                TopFMConfig=TopFMConfig,
                rank_top_fms=rank_top_fms,
                require_sat=bool(args.require_sat),
            )
            results[pipeline_name] = manifest
        print(json.dumps(results, indent=2))
        return

    fm_dir = Path(args.fm_dir).expanduser().resolve()
    if not fm_dir.exists():
        raise FileNotFoundError(f"FM directory not found: {fm_dir}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else fm_dir.parent
    patterns = list(args.glob) if args.glob else ["*.xml"]
    if args.interactive:
        inferred = sorted(
            {
                p.name.split("_response_", 1)[1].rsplit("_20", 1)[0]
                for p in fm_dir.glob("*.xml")
                if "_response_" in p.name
            }
        )
        if inferred:
            choice = _prompt_choice("Select model family to rank", inferred + ["Custom glob"])
            if choice != "Custom glob":
                patterns = [f"*{choice}*.xml"]
            else:
                custom = input(f"Glob pattern [{patterns[0]}]: ").strip()
                patterns = [custom or patterns[0]]

    manifest = _rank_one(
        fm_dir=fm_dir,
        outdir=outdir,
        gt_path=gt_path,
        xsd_path=xsd_path,
        top_n=args.top_fm,
        patterns=patterns,
        coverage_cfg=coverage_cfg,
        TopFMConfig=TopFMConfig,
        rank_top_fms=rank_top_fms,
        require_sat=bool(args.require_sat),
    )
    if not manifest:
        print("No ranking output produced.")
        return
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
