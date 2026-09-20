from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .coverage import CoverageConfig, CoverageEvaluator
from .feature_list import extract_feature_list
from .quality_sat import analyze_sat_quality
from .semantic import semantic_prf
from .wellformed import validate_feature_model


@dataclass(frozen=True)
class TopFMConfig:
    top_n: int
    gt_xml: Path
    xsd_path: Optional[Path]
    coverage: CoverageConfig
    output_subdir: str = ""
    require_sat: bool = False


def _overall_score(semantic_f1: Optional[float], coverage_score: Optional[float]) -> Optional[float]:
    if semantic_f1 is None or coverage_score is None:
        return None
    return 0.6 * float(semantic_f1) + 0.4 * float(coverage_score)


def _duplicate_feature_names(xml_path: Path) -> List[str]:
    counts = Counter(rec.feature_name for rec in extract_feature_list(xml_path))
    return sorted(name for name, count in counts.items() if count > 1)


def _summary_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        1 if row.get("eligible_ok") else 0,
        float(row.get("overall_score") or -1),
        float(row.get("semantic_f1") or -1),
        float(row.get("coverage_score") or -1),
        -int(row.get("candidate_index") or 0),
    )


def _metric_sort_key(metric: str, row: Dict[str, Any]) -> tuple:
    if metric == "overall":
        return (
            float(row.get("overall_score") or -1),
            float(row.get("semantic_f1") or -1),
            float(row.get("coverage_score") or -1),
            -int(row.get("candidate_index") or 0),
        )
    if metric == "semantic_f1":
        return (
            1 if row.get("eligible_ok") else 0,
            float(row.get("semantic_f1") or -1),
            float(row.get("coverage_score") or -1),
            -int(row.get("candidate_index") or 0),
        )
    if metric == "coverage_score":
        return (
            1 if row.get("eligible_ok") else 0,
            float(row.get("coverage_score") or -1),
            float(row.get("semantic_f1") or -1),
            -int(row.get("candidate_index") or 0),
        )
    return (
        1 if row.get("eligible_ok") else 0,
        float(row.get("semantic_f1") or -1),
        float(row.get("coverage_score") or -1),
        -int(row.get("duplicate_feature_count") or 0),
        -int(row.get("wellformed_error_count") or 0),
        -int(row.get("candidate_index") or 0),
    )


def build_ranked_rows(
    *,
    candidates: Sequence[Dict[str, Any]],
    cfg: TopFMConfig,
) -> List[Dict[str, Any]]:
    if not cfg.gt_xml.exists():
        raise FileNotFoundError(f"Ground-truth XML not found: {cfg.gt_xml}")

    evaluator = CoverageEvaluator(cfg.coverage)
    ranked_rows: List[Dict[str, Any]] = []

    total = len(candidates)
    print(f"[top_fm] Evaluating {total} candidate FM files...")

    for idx, candidate in enumerate(candidates, start=1):
        fm_xml = Path(str(candidate.get("fm_xml") or candidate.get("final_xml") or "")).expanduser().resolve()
        run_id = str(candidate.get("run_id") or fm_xml.stem)
        print(f"[top_fm] [{idx}/{total}] Checking {run_id}")
        if not fm_xml.exists():
            print(f"[top_fm] [{idx}/{total}] Skipping missing file: {fm_xml}")
            continue

        wf = validate_feature_model(fm_xml, xsd_path=cfg.xsd_path)
        duplicate_feature_names: List[str] = []
        duplicate_feature_error = ""
        try:
            duplicate_feature_names = _duplicate_feature_names(fm_xml)
        except Exception as exc:
            duplicate_feature_error = str(exc)
        has_duplicate_features = len(duplicate_feature_names) > 0
        satisfiable: Optional[bool] = None
        sat_error = ""
        if bool(wf.ok) and not has_duplicate_features and not duplicate_feature_error:
            try:
                sat = analyze_sat_quality(fm_xml, compute_products=False)
                satisfiable = sat.satisfiable
            except Exception as exc:
                sat_error = str(exc)

        eligible_ok = bool(wf.ok) and not has_duplicate_features and not duplicate_feature_error
        if cfg.require_sat:
            eligible_ok = eligible_ok and bool(satisfiable) and not sat_error
        coverage_score = None
        semantic_f1 = None
        semantic_precision = None
        semantic_recall = None

        if eligible_ok:
            print(f"[top_fm] [{idx}/{total}] Valid FM. Computing coverage and semantic scores...")
            coverage_score = evaluator.score(cfg.gt_xml, fm_xml, verbose=False)
            sem = semantic_prf(cfg.gt_xml, fm_xml, model=evaluator.model, threshold=cfg.coverage.similarity_threshold)
            semantic_precision = sem.get("semantic_precision")
            semantic_recall = sem.get("semantic_recall")
            semantic_f1 = sem.get("semantic_f1")
            print(
                f"[top_fm] [{idx}/{total}] Done: semantic_f1={semantic_f1}, coverage_score={coverage_score}"
            )
        else:
            reasons: List[str] = []
            if not wf.ok:
                reasons.append(f"xsd_or_wellformed_errors={len(wf.errors)}")
            if has_duplicate_features:
                reasons.append(f"duplicate_features={len(duplicate_feature_names)}")
            if duplicate_feature_error:
                reasons.append("duplicate_check_failed")
            if cfg.require_sat and sat_error:
                reasons.append("sat_check_failed")
            if cfg.require_sat and satisfiable is False:
                reasons.append("unsatisfiable")
            print(f"[top_fm] [{idx}/{total}] Ineligible: {', '.join(reasons) if reasons else 'unknown reason'}")

        ranked_rows.append(
            {
                "candidate_index": idx,
                "run_id": candidate.get("run_id"),
                "fm_xml": str(fm_xml),
                "meta": candidate.get("meta"),
                "wellformed_ok": wf.ok,
                "wellformed_error_count": len(wf.errors),
                "has_duplicate_features": has_duplicate_features,
                "duplicate_feature_count": len(duplicate_feature_names),
                "duplicate_feature_names": duplicate_feature_names,
                "duplicate_feature_error": duplicate_feature_error,
                "satisfiable": satisfiable,
                "sat_error": sat_error,
                "eligible_ok": eligible_ok,
                "coverage_score": coverage_score,
                "semantic_precision": semantic_precision,
                "semantic_recall": semantic_recall,
                "semantic_f1": semantic_f1,
                "overall_score": _overall_score(semantic_f1, coverage_score),
            }
        )

    return ranked_rows


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in name)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def rank_top_fms(
    *,
    candidates: Sequence[Dict[str, Any]],
    pipeline_root: Path,
    cfg: TopFMConfig,
) -> Optional[Dict[str, Any]]:
    if cfg.top_n <= 0 or not candidates:
        return None
    print(f"[top_fm] Ranking into: {pipeline_root}")
    ranked_rows = build_ranked_rows(candidates=candidates, cfg=cfg)

    top_root = pipeline_root / "top_fm"
    if cfg.output_subdir:
        top_root = top_root / cfg.output_subdir
    top_root.mkdir(parents=True, exist_ok=True)

    summary_rows = sorted(ranked_rows, key=_summary_sort_key, reverse=True)

    metric_specs = ("overall", "semantic_f1", "coverage_score", "wellformed_ok")

    manifest: Dict[str, Any] = {
        "pipeline_root": str(pipeline_root),
        "ground_truth_xml": str(cfg.gt_xml),
        "top_n": cfg.top_n,
        "summary_table": str(top_root / "top_fm_scores.csv"),
        "overall_ranking_rule": (
            "eligible if wellformed_ok=1 and duplicate_feature_count=0"
            + (" and satisfiable=1" if cfg.require_sat else "")
            + ", then overall_score = 0.6 * semantic_f1 + 0.4 * coverage_score"
        ),
        "metrics": {},
    }

    for metric in metric_specs:
        metric_dir = top_root / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        eligible = [r for r in ranked_rows if r.get("eligible_ok")]
        print(f"[top_fm] Selecting top {cfg.top_n} for metric '{metric}' from {len(eligible)} eligible candidates")
        ordered = sorted(eligible, key=lambda r, m=metric: _metric_sort_key(m, r), reverse=True)[: cfg.top_n]
        copied: List[Dict[str, Any]] = []
        for rank, row in enumerate(ordered, start=1):
            src = Path(str(row["fm_xml"]))
            dst = metric_dir / f"rank{rank:02d}_{_safe_name(src.name)}"
            shutil.copy2(src, dst)
            row_copy = dict(row)
            row_copy["rank"] = rank
            row_copy["copied_xml"] = str(dst)
            copied.append(row_copy)
        metric_table = metric_dir / f"top_{metric}.csv"
        _write_csv(
            metric_table,
            copied,
            [
                "rank",
                "run_id",
                "fm_xml",
                "copied_xml",
                "meta",
                "wellformed_ok",
                "wellformed_error_count",
                "eligible_ok",
                "has_duplicate_features",
                "duplicate_feature_count",
                "duplicate_feature_names",
                "satisfiable",
                "sat_error",
                "overall_score",
                "semantic_f1",
                "coverage_score",
                "semantic_precision",
                "semantic_recall",
            ],
        )
        manifest["metrics"][metric] = {
            "table": str(metric_table),
            "items": copied,
        }

    _write_csv(
        top_root / "top_fm_scores.csv",
        summary_rows,
        [
            "candidate_index",
            "run_id",
            "fm_xml",
            "meta",
            "wellformed_ok",
            "wellformed_error_count",
            "eligible_ok",
            "has_duplicate_features",
            "duplicate_feature_count",
            "duplicate_feature_names",
            "duplicate_feature_error",
            "satisfiable",
            "sat_error",
            "overall_score",
            "semantic_f1",
            "coverage_score",
            "semantic_precision",
            "semantic_recall",
        ],
    )
    (top_root / "top_fm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
