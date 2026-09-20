#!/usr/bin/env python
"""Check well-formedness/validation of a FeatureIDE XML FM."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from fame.evaluation import validate_feature_model
from fame.utils.dirs import build_paths, ensure_for_stage


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Well-formedness/XSD check for FeatureIDE XML")
    ap.add_argument("--xml", required=True, help="Path to FeatureIDE XML")
    ap.add_argument("--xsd", help="Optional XSD schema path for validation")
    ap.add_argument("--quiet", action="store_true", help="Only print JSON result")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths()
    ensure_for_stage("evaluation", paths)

    res = validate_feature_model(args.xml, args.xsd)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    xml_path = Path(args.xml)
    xsd_path = Path(args.xsd) if args.xsd else None
    out_name = f"wellformed_{xml_path.stem}_{timestamp}.json"
    out_file = paths.evaluation_root / out_name

    payload = {
        "ok": res.ok,
        "errors": res.errors,
        "xml": str(xml_path),
        "xsd": str(xsd_path) if xsd_path else None,
        "timestamp_utc": timestamp,
    }
    out_file.write_text(json.dumps(payload, indent=2))

    if args.quiet:
        print(json.dumps(payload))
    else:
        status = "SUCCESS:" if res.ok else "ERROR:"
        print(f"{status} Well-formed: {res.ok}")
        if res.errors:
            print("Errors:")
            for e in res.errors:
                print(f"- {e}")
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
