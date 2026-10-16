#!/usr/bin/env python3
"""Read-only paired run bootstrap for the independent-max negative control.

Print CSV to stdout; diagnostics to stderr. Never changes scores or archives.
All four paper-mean checks must pass before any bootstrap samples are drawn.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/ifs-2027/analysis'
TAUS = (.4, .6)
EXPECTED = {('repair', .4): .228, ('repair', .6): .176,
            ('federation', .4): .053, ('federation', .6): .011}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_differences(scores_path, inventory_path, policy='independent_max', taus=TAUS):
    inventory = json.loads(inventory_path.read_text())
    completed = {r['run_id']: r['corpus'] for r in inventory if r['completed'] is True}
    if len(completed) != 500 or sum(r['completed'] is True for r in inventory) != 500:
        raise ValueError('Expected exactly 500 distinct completed inventory runs')
    if set(completed.values()) != {'repair', 'federation'}:
        raise ValueError('Unexpected corpus in completed population')
    cells = {(corpus, tau): {} for corpus in ('repair', 'federation') for tau in taus}
    with scores_path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['matching_policy'] != policy:
                continue
            tau = float(row['tau'])
            if tau not in taus:
                continue
            rid, corpus = row['run_id'], row['corpus']
            if rid not in completed or completed[rid] != corpus or row['status'] != 'ok':
                raise ValueError(f'Unknown, wrong-corpus or invalid score: {rid}')
            if row['own_reference_corpus'] != corpus or row['cross_reference_corpus'] != ('federation' if corpus == 'repair' else 'repair'):
                raise ValueError(f'Incorrect own/cross reference labels: {rid}')
            cell = cells[corpus, tau]
            if rid in cell:
                raise ValueError(f'Duplicate score: {rid}, tau={tau}')
            own, cross = float(row['f1_total_own']), float(row['f1_total_cross'])
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (own, cross)):
                raise ValueError(f'Invalid F1: {rid}')
            delta = own-cross
            if not math.isclose(delta, float(row['f1_total_own_minus_cross']), abs_tol=1e-12, rel_tol=0):
                raise ValueError(f'Saved difference inconsistent with own minus cross: {rid}')
            cell[rid] = delta
    for (corpus, tau), cell in cells.items():
        expected_ids = {rid for rid, c in completed.items() if c == corpus}
        if set(cell) != expected_ids:
            raise ValueError(f'Incomplete population: {corpus}, tau={tau}')
    return cells


def check_means(cells, expected=EXPECTED):
    failures = []
    for key, target in expected.items():
        actual = statistics.fmean(cells[key].values())
        print(f'Sanity: {key[0]} tau={key[1]:.1f} n={len(cells[key])} '
              f'mean={actual:.9f} expected={target:.3f}', file=sys.stderr)
        if abs(actual-target) > .001:
            failures.append(f'{key}: mean {actual:.9f} differs from {target:.3f} by {abs(actual-target):.9f}')
    if failures:
        raise ValueError('STOP: paper-mean check failed; no intervals computed.\n' + '\n'.join(failures))


def bootstrap(cells, seed):
    """Sample whole runs; share draws across taus within each corpus.

    Stable sorted run IDs; independent SeedSequence child streams per corpus.
    NumPy linear empirical percentiles at 2.5 and 97.5, 5,000 draws each.
    """
    rows = []
    for corpus, stream in zip(('repair', 'federation'), np.random.SeedSequence(seed).spawn(2)):
        ids = sorted(cells[corpus, .4])
        values = np.array([[cells[corpus, tau][rid] for tau in TAUS] for rid in ids])
        rng = np.random.default_rng(stream)
        draws = np.empty((5000, len(TAUS)))
        for b in range(5000):
            draws[b] = values[rng.integers(0, len(ids), size=len(ids))].mean(axis=0)
        limits = np.percentile(draws, [2.5, 97.5], axis=0, method='linear')
        for j, tau in enumerate(TAUS):
            rows.append((corpus, tau, len(ids), statistics.fmean(values[:, j]), limits[0,j], limits[1,j]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, default=BASE/'cross-reference-v1/scores.csv')
    parser.add_argument('--inventory', type=Path, default=BASE/'inventory-current-v1/runs.json')
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--federation-maximum', action='store_true',
                        help='Print the maximum Federation mean difference across both policies and tau 0.3–0.8; no bootstrap')
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('Seed must be nonnegative')
    try:
        cells = load_differences(args.scores, args.inventory)
        check_means(cells)  # Gate ALL cells before the first draw.
        if args.federation_maximum:
            candidates = []
            for policy in ('independent_max', 'one_to_one'):
                sweep = load_differences(args.scores, args.inventory, policy,
                                         (.3, .4, .5, .6, .7, .8))
                for (corpus, tau), values in sweep.items():
                    if corpus == 'federation':
                        candidates.append((statistics.fmean(values.values()), tau, policy, len(values)))
            maximum = max(row[0] for row in candidates)
            writer = csv.writer(sys.stdout, lineterminator='\n')
            writer.writerow(('corpus', 'tau', 'matching_policy', 'n', 'mean_difference'))
            for mean, tau, policy, n in candidates:
                if mean == maximum:
                    writer.writerow(('Federation', f'{tau:.3f}', policy, n, f'{mean:.3f}'))
            return 0
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f'Bootstrap: 5000 resamples, 95% percentile CI, seed={args.seed}, '
          f'NumPy={np.__version__}, linear percentiles, sorted run IDs.', file=sys.stderr)
    print(f'scores_sha256={digest(args.scores)}\ninventory_sha256={digest(args.inventory)}', file=sys.stderr)
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(('corpus', 'tau', 'n', 'mean_difference', 'ci_lower', 'ci_upper'))
    for corpus, tau, n, mean, lower, upper in bootstrap(cells, args.seed):
        writer.writerow((corpus.title(), f'{tau:.3f}', n, f'{mean:.3f}', f'{lower:.3f}', f'{upper:.3f}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
