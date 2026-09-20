#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from fame.evaluation.feature_list import extract_feature_list
from fame.evaluation.constraints import extract_constraints
from fame.evaluation.coverage import extract_nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Export ground-truth FeatureIDE FM to JSON (features, constraints, hierarchy)")
    ap.add_argument("--gt", default="data/ground_truth/federation.xml", help="Ground-truth FeatureIDE XML")
    ap.add_argument("--out-dir", default="results/ground_truth/export", help="Output directory for JSON files")
    args = ap.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth XML not found: {gt_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Features
    feats = extract_feature_list(gt_path)
    feats_json = [f.to_dict() for f in feats]
    (out_dir / "gt_features.json").write_text(json.dumps(feats_json, indent=2), encoding="utf-8")

    # Constraints
    cons = extract_constraints(gt_path)
    cons_json = [c.to_dict() for c in cons]
    (out_dir / "gt_constraints.json").write_text(json.dumps(cons_json, indent=2), encoding="utf-8")

    # Hierarchy (parent -> child edges with depth; root_feature depth=0)
    nodes = extract_nodes(gt_path)  # list of (name, parent_name)
    # Build depth map by traversing from root-feature nodes (parent None)
    depths = {}
    queue = []
    for name, parent in nodes:
        if parent is None:
            depths[name] = 0
            queue.append(name)
    # Propagate depths in breadth-first manner
    while queue:
        cur = queue.pop(0)
        cur_depth = depths.get(cur, 0)
        for child, parent in nodes:
            if parent == cur and child not in depths:
                depths[child] = cur_depth + 1
                queue.append(child)

    edges: List[Dict[str, str | None | int]] = []
    for name, parent in nodes:
        edges.append(
            {
                "parent": parent,
                "child": name,
                "depth": depths.get(name),
            }
        )
    (out_dir / "gt_hierarchy.json").write_text(json.dumps(edges, indent=2), encoding="utf-8")

    print("GT export completed")
    print(f"Features     : {len(feats_json)} -> {out_dir / 'gt_features.json'}")
    print(f"Constraints  : {len(cons_json)} -> {out_dir / 'gt_constraints.json'}")
    print(f"Hierarchy edges: {len(edges)} -> {out_dir / 'gt_hierarchy.json'}")


if __name__ == "__main__":
    main()
