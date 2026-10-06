#!/usr/bin/env python3
"""Derive numeric dead-feature counts from saved evaluation lists; no SAT rerun."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = 'structural__dead_features'
COUNT = 'structural__dead_feature_count'
STATUSES = {'ok', 'not_applicable', 'ineligible', 'missing_artifact',
            'unsupported', 'evaluator_error'}


def derive_counts(rows: list[dict]) -> list[dict]:
    """Empty JSON lists mean zero; non-ok metrics retain null and their reason."""
    out, seen = [], set()
    for original in rows:
        row = dict(original)
        rid = row.get('run_id')
        if not rid or rid in seen:
            raise ValueError(f'Missing or duplicate run_id: {rid!r}')
        seen.add(rid)
        status = row.get(SOURCE + '__status')
        if status not in STATUSES:
            raise ValueError(f'{rid}: missing or unrecognised dead-feature status {status!r}')
        count = None
        if status == 'ok':
            try:
                names = json.loads(row[SOURCE])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{rid}: expected JSON list for an ok dead-feature metric') from exc
            if (not isinstance(names, list) or
                    any(not isinstance(n, str) or not n.strip() for n in names) or
                    len(names) != len(set(names))):
                raise ValueError(f'{rid}: dead features must be a list of unique nonempty names')
            count = len(names)
        row[COUNT] = count
        row[COUNT + '__status'] = status
        row[COUNT + '__reason'] = row.get(SOURCE + '__reason', '')
        out.append(row)
    return out


def digest(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wide', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = REPO / 'results/ifs-2027/analysis'
    if output == allowed or not output.is_relative_to(allowed) or output.exists():
        parser.error('Choose a fresh subdirectory under results/ifs-2027/analysis')
    with args.wide.open(newline='', encoding='utf-8-sig') as f:
        rows = derive_counts(list(csv.DictReader(f)))
    if not rows:
        raise ValueError('Input table contains no runs')
    counts = [r[COUNT] for r in rows if r[COUNT + '__status'] == 'ok']
    summary = dict(
        n_runs=len(rows), n_evaluated=len(counts), n_unavailable=len(rows)-len(counts),
        n_zero=sum(n == 0 for n in counts), n_positive=sum(n > 0 for n in counts),
        total_dead_feature_occurrences=sum(counts), maximum_count=max(counts, default=None),
        status_counts=dict(Counter(r[COUNT + '__status'] for r in rows)),
        input_path=str(args.wide.resolve()), input_sha256=digest(args.wide),
        implementation_sha256=digest(Path(__file__)),
        definition='Length of saved non-root dead-feature name list when status=ok; otherwise null.',
        note='Derived export only. No new SAT solving or model calls. Unsatisfiable outputs retain not_applicable.',
    )
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'wide.csv', rows, list(rows[0]))
    compact = ['run_id', 'model_id', 'corpus', 'arm', 'N', 'grounding', 'seed',
               'repetition', 'inventory_status', 'structural__satisfiable',
               'structural__satisfiable__status', SOURCE, SOURCE+'__status',
               SOURCE+'__reason', COUNT, COUNT+'__status', COUNT+'__reason',
               'structural__dead_feature_ratio', 'structural__dead_feature_ratio__status']
    write_csv(output / 'dead_feature_counts.csv', rows, compact)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
