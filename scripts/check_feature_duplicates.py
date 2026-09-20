#!/usr/bin/env python
"""Check duplicate feature/group names in a FeatureIDE XML FM."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from lxml import etree

from fame.utils.dirs import build_paths, ensure_for_stage

NODE_TAGS = {"feature", "and", "or", "alt"}
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
UNDERSCORE_RE = re.compile(r"_+")


def _normalize_name(name: str, mode: str) -> str:
    if mode == "none":
        return name
    n = (name or "").strip().lower()
    n = NON_ALNUM_RE.sub("_", n)
    n = UNDERSCORE_RE.sub("_", n).strip("_")
    return n


def _is_node(elem) -> bool:
    return isinstance(elem.tag, str) and elem.tag in NODE_TAGS


def _collect_occurrences(xml_path: Path) -> List[Dict[str, object]]:
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    if root.tag != "featureModel":
        raise ValueError(f"Root tag is '{root.tag}', expected 'featureModel'")
    struct = root.find("struct")
    if struct is None:
        raise ValueError("<struct> not found in feature model")

    rows: List[Dict[str, object]] = []

    def walk(node, *, path: str, depth: int, parent_name: str) -> None:
        child_nodes = [ch for ch in node if _is_node(ch)]
        for idx, ch in enumerate(child_nodes, start=1):
            child_path = f"{path}/{ch.tag}[{idx}]"
            name = (ch.attrib.get("name") or "").strip()
            next_parent = parent_name

            if name:
                rows.append(
                    {
                        "name": name,
                        "tag": ch.tag,
                        "line": int(ch.sourceline) if getattr(ch, "sourceline", None) else None,
                        "depth": depth,
                        "parent_name": parent_name or None,
                        "path": child_path,
                    }
                )
                next_parent = name

            walk(ch, path=child_path, depth=(depth + 1 if name else depth), parent_name=next_parent)

    walk(struct, path="/featureModel/struct", depth=0, parent_name="")
    return rows


def _duplicate_groups(rows: List[Dict[str, object]], *, mode: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        key = _normalize_name(str(r["name"]), mode)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, occ in groups.items():
        if len(occ) < 2:
            continue
        raw_names = sorted({str(x["name"]) for x in occ})
        out.append(
            {
                "group_key": key,
                "count": len(occ),
                "unique_raw_names": raw_names,
                "occurrences": occ,
            }
        )
    out.sort(key=lambda x: (-int(x["count"]), str(x["group_key"])))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check duplicate feature/group names in FeatureIDE XML")
    ap.add_argument("--xml", required=True, help="Path to FeatureIDE XML")
    ap.add_argument(
        "--normalization",
        choices=("basic", "none"),
        default="basic",
        help="Name normalization used for near-duplicate detection (default: basic).",
    )
    ap.add_argument("--out", default="", help="Optional output JSON path")
    ap.add_argument("--quiet", action="store_true", help="Only print JSON payload")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths()
    ensure_for_stage("evaluation", paths)

    xml_path = Path(args.xml).expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    rows = _collect_occurrences(xml_path)
    exact_groups = _duplicate_groups(rows, mode="none")
    normalized_groups = _duplicate_groups(rows, mode=args.normalization)

    # Keep only true "near-duplicates" for normalized view when using basic mode:
    # same normalized key but with more than one raw form.
    if args.normalization == "basic":
        normalized_groups = [g for g in normalized_groups if len(g["unique_raw_names"]) > 1 or g["count"] > 1]

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = (
        Path(args.out).expanduser().resolve()
        if str(args.out).strip()
        else paths.evaluation_root / f"feature_duplicates_{xml_path.stem}_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "xml": str(xml_path),
        "timestamp_utc": ts,
        "normalization": args.normalization,
        "total_named_nodes": len(rows),
        "unique_exact_names": len({str(r["name"]) for r in rows}),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_nodes": sum(int(g["count"]) - 1 for g in exact_groups),
        "normalized_duplicate_groups": len(normalized_groups),
        "normalized_duplicate_nodes": sum(int(g["count"]) - 1 for g in normalized_groups),
        "duplicates_exact": exact_groups,
        "duplicates_normalized": normalized_groups,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.quiet:
        print(json.dumps(payload))
        return

    status = "SUCCESS:" if payload["exact_duplicate_groups"] == 0 else "WARN:"
    print(f"{status} Duplicate scan completed")
    print(f"XML                  : {xml_path}")
    print(f"Named nodes          : {payload['total_named_nodes']}")
    print(f"Unique names         : {payload['unique_exact_names']}")
    print(f"Exact duplicate sets : {payload['exact_duplicate_groups']}")
    print(f"Exact duplicate nodes: {payload['exact_duplicate_nodes']}")
    print(f"Near-duplicate sets  : {payload['normalized_duplicate_groups']}")
    print(f"Near-duplicate nodes : {payload['normalized_duplicate_nodes']}")
    print(f"Saved                : {out_path}")

    if exact_groups:
        print("\nTop exact duplicates:")
        for g in exact_groups[:10]:
            print(f"- {g['group_key']} (count={g['count']})")


if __name__ == "__main__":
    main()
