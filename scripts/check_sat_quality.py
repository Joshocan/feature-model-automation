#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def _collect_xmls(fm_xml: str | None, fm_dir: str | None, patterns: list[str]) -> list[Path]:
    if bool(fm_xml) == bool(fm_dir):
        raise ValueError('Provide exactly one of: fm_xml or --fm-dir')

    if fm_xml:
        path = Path(fm_xml).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f'FM XML not found: {path}')
        return [path]

    root = Path(fm_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f'FM directory not found: {root}')

    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in (patterns or ['*.xml']):
        for path in sorted(root.glob(pattern)):
            resolved = path.resolve()
            if resolved.is_file() and resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    if not files:
        raise FileNotFoundError(f'No FM XML files matched in {root} for patterns: {patterns or ["*.xml"]}')
    return files


def _payload(xml_path: Path, result=None, error: Exception | None = None) -> dict:
    if error is not None:
        return {
            'fm_xml': str(xml_path),
            'valid_fm': False,
            'satisfiable': None,
            'dead_features': None,
            'dead_features_count': None,
            'core_features': None,
            'core_features_count': None,
            'products_count': None,
            'unsat_core_labels': None,
            'unsat_reasons': None,
            'error': f'{type(error).__name__}: {error}',
        }
    return {
        'fm_xml': str(xml_path),
        'valid_fm': True,
        'satisfiable': result.satisfiable,
        'dead_features': result.dead_features,
        'dead_features_count': len(result.dead_features or []),
        'core_features': result.core_features,
        'core_features_count': len(result.core_features or []),
        'products_count': result.products_count,
        'unsat_core_labels': result.unsat_core_labels,
        'unsat_reasons': result.unsat_reasons,
        'error': None,
    }


def _print_single(payload: dict) -> None:
    print(f'FM XML             : {payload["fm_xml"]}')
    print(f'Valid FM           : {payload["valid_fm"]}')
    print(f'Satisfiable        : {payload["satisfiable"]}')
    print(f'Dead features      : {payload["dead_features"]}')
    print(f'Dead features count: {payload["dead_features_count"]}')
    print(f'Core features      : {payload["core_features"]}')
    print(f'Core features count: {payload["core_features_count"]}')
    print(f'Products count     : {payload["products_count"]}')
    print(f'Unsat core         : {payload["unsat_core_labels"]}')
    print(f'Unsat reasons      : {payload["unsat_reasons"]}')
    print(f'Error              : {payload["error"]}')


def _print_batch(rows: Iterable[dict]) -> None:
    rows = list(rows)
    total = len(rows)
    valid_count = sum(1 for row in rows if row['valid_fm'] is True)
    invalid_count = total - valid_count
    sat_count = sum(1 for row in rows if row['satisfiable'] is True)
    unsat_count = sum(1 for row in rows if row['satisfiable'] is False)
    error_count = sum(1 for row in rows if row['error'])
    unknown_count = total - sat_count - unsat_count - error_count

    print(f'Total FMs    : {total}')
    print(f'Valid FMs    : {valid_count}')
    print(f'Invalid FMs  : {invalid_count}')
    print(f'Satisfiable  : {sat_count}')
    print(f'Unsatisfiable: {unsat_count}')
    print(f'Errors       : {error_count}')
    print(f'Unknown      : {unknown_count}')
    print('')
    for row in rows:
        if row['error']:
            status = 'INVALID'
        elif row['satisfiable'] is True:
            status = 'SAT'
        elif row['satisfiable'] is False:
            status = 'UNSAT'
        else:
            status = 'VALID'
        print(f'[{status}] {row["fm_xml"]}')
        if row['unsat_reasons']:
            for reason in row['unsat_reasons']:
                print(f'  - {reason}')
        if row['error']:
            print(f'  - {row["error"]}')


def _write_out(path: Path, rows: list[dict]) -> None:
    suffix = path.suffix.lower()
    if suffix == '.json':
        path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        return
    if suffix == '.jsonl':
        with path.open('w', encoding='utf-8') as fh:
            for row in rows:
                fh.write(json.dumps(row) + '\n')
        return
    if suffix == '.csv':
        fieldnames = [
            'fm_xml', 'valid_fm', 'satisfiable', 'dead_features_count', 'core_features_count', 'products_count',
            'dead_features', 'core_features', 'unsat_core_labels', 'unsat_reasons', 'error',
        ]
        with path.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    **row,
                    'dead_features': json.dumps(row['dead_features']),
                    'core_features': json.dumps(row['core_features']),
                    'unsat_core_labels': json.dumps(row['unsat_core_labels']),
                    'unsat_reasons': json.dumps(row['unsat_reasons']),
                })
        return
    raise ValueError(f'Unsupported output format for {path}; use .json, .jsonl, or .csv')


def main() -> None:
    from fame.evaluation.quality_sat import analyze_sat_quality

    ap = argparse.ArgumentParser(description='Run SAT quality analysis on one FM XML file or a directory of FM XML files.')
    ap.add_argument('fm_xml', nargs='?', help='Path to one FM XML file')
    ap.add_argument('--fm-dir', help='Directory containing FM XML files')
    ap.add_argument('--glob', action='append', default=[], help='Glob pattern(s) to use with --fm-dir')
    ap.add_argument('--products', action='store_true', help='Count valid configurations (can be slow)')
    ap.add_argument('--json', action='store_true', help='Print JSON to stdout for single-file mode')
    ap.add_argument('--out', help='Optional output path (.json, .jsonl, or .csv)')
    args = ap.parse_args()

    xml_paths = _collect_xmls(args.fm_xml, args.fm_dir, args.glob)
    rows = []
    for xml_path in xml_paths:
        try:
            result = analyze_sat_quality(xml_path, compute_products=bool(args.products))
            rows.append(_payload(xml_path, result=result))
        except Exception as exc:
            rows.append(_payload(xml_path, error=exc))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_out(out_path, rows)
        print(f'Wrote SAT quality results to {out_path}')

    if len(rows) == 1:
        if args.json:
            print(json.dumps(rows[0], indent=2))
            return
        _print_single(rows[0])
        return

    if args.json:
        print(json.dumps(rows, indent=2))
        return
    _print_batch(rows)


if __name__ == '__main__':
    main()
