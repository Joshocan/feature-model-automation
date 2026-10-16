#!/usr/bin/env python3
"""Analyse two independently completed audit templates without altering originals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fame.evaluation.inventory import sha256
from scripts.prepare_pair_audit import LABELS, PUBLIC, read_csv, write_csv


def validate_ratings(rows, key):
    result = {}
    for row in rows:
        item = row.get('item_id')
        if item not in key or item in result:
            raise ValueError(f'Unknown or duplicate item: {item}')
        if any(row.get(k) != key[item][k] for k in PUBLIC):
            raise ValueError(f'Changed frozen feature text or corpus: {item}')
        label = row.get('label', '').strip()
        if label not in LABELS:
            raise ValueError(f'{item}: fill label with one of {LABELS}; blanks are not cannot_judge')
        result[item] = label
    if set(result) != set(key):
        raise ValueError('Missing rating items')
    return result


def agreement(a, b):
    pairs = [(a[k], b[k]) for k in a if a[k] != 'cannot_judge' and b[k] != 'cannot_judge']
    n = len(pairs)
    if not n:
        return dict(n=0, raw_agreement=None, expected_agreement=None, kappa=None, status='no_jointly_judged_items')
    left, right = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = sum(x == y for x, y in pairs)/n
    expected = sum(left[c]*right[c] for c in LABELS[:3])/(n*n)
    return dict(n=n, raw_agreement=observed, expected_agreement=expected,
                kappa=(observed-expected)/(1-expected) if expected < 1 else None,
                status='ok' if expected < 1 else 'undefined_constant_marginals')


def analyse(key, counts, a, b):
    rows = []
    for sampling in counts:
        band = sampling['band']
        ids = [k for k, r in key.items() if r['band'] == band]
        for name, ratings in (('R01', a), ('R02', b)):
            labels = Counter(ratings[k] for k in ids)
            n, judged = len(ids), len(ids)-labels['cannot_judge']
            rows.append(dict(band=band, rater=name, population=int(sampling['population']),
                sampled=n, judged=judged, **{c: labels[c] for c in LABELS},
                same_share_all_sampled=labels['same_concept']/n if n else None,
                same_share_judged=labels['same_concept']/judged if judged else None))
    weighted = []
    for name in ('R01', 'R02'):
        rs = [r for r in rows if r['rater'] == name and r['population']]
        total = sum(r['population'] for r in rs)
        valid = total > 0 and all(r['sampled'] for r in rs)
        same = sum(r['population']*r['same_concept']/r['sampled'] for r in rs) / total if valid else None
        cj = sum(r['population']*r['cannot_judge']/r['sampled'] for r in rs) / total if valid else None
        weighted.append(dict(rater=name, eligible_population=total,
            estimated_same_share_all=same, estimated_cannot_judge_share=cj,
            estimated_same_share_judged=same/(1-cj) if valid and cj < 1 else None))
    return rows, weighted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--rater1', type=Path, required=True)
    parser.add_argument('--rater2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Use a fresh output directory')
    if args.rater1.resolve() == args.rater2.resolve():
        parser.error('Provide two separately completed rating files')
    private = args.audit_dir / 'AUTHOR_ONLY'
    manifest = json.loads((private / 'manifest.json').read_text())
    for relative, expected in manifest['frozen_files'].items():
        path = (args.audit_dir / relative).resolve()
        if not path.is_relative_to(args.audit_dir.resolve()) or sha256(path) != expected:
            raise ValueError(f'Frozen audit input changed: {relative}; rate copies, not originals')
    key_rows = read_csv(private / 'sample_key.csv')
    key = {r['item_id']: r for r in key_rows}
    if len(key) != len(key_rows):
        raise ValueError('Duplicate frozen item IDs')
    a = validate_ratings(read_csv(args.rater1), key)
    b = validate_ratings(read_csv(args.rater2), key)
    counts = read_csv(private / 'sampling_counts.csv')
    bands, weighted = analyse(key, counts, a, b)
    agreement_rows = [dict(scope='all', **agreement(a, b))]
    for c in counts:
        ids = [k for k, r in key.items() if r['band'] == c['band']]
        agreement_rows.append(dict(scope=c['band'], **agreement({k:a[k] for k in ids}, {k:b[k] for k in ids})))
    confusion = Counter((a[k], b[k]) for k in key)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'band_summary.csv', bands, list(bands[0]))
    write_csv(args.output / 'weighted_summary.csv', weighted, list(weighted[0]))
    write_csv(args.output / 'agreement.csv', agreement_rows, list(agreement_rows[0]))
    write_csv(args.output / 'confusion.csv', [dict(rater1=x, rater2=y, count=confusion[x,y]) for x in LABELS for y in LABELS], ['rater1','rater2','count'])
    summary = dict(items=len(key), agreement=agreement_rows[0],
        excluded_from_kappa=sum(a[k]=='cannot_judge' or b[k]=='cannot_judge' for k in key),
        kappa_definition='unweighted nominal Cohen kappa; jointly judged pairs only',
        interpretation='Descriptive agreement on a band-balanced sample, not population-weighted kappa. Repeated occurrences and runs are dependent; no independence-based intervals or significance tests.',
        weighting='Band population weights; cannot_judge reported separately, not treated as unrelated. Conditional same-share is a ratio estimate.',
        inputs={str(p.resolve()):sha256(p) for p in (args.rater1,args.rater2,private/'manifest.json')},
        implementation_sha256=sha256(Path(__file__)))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['agreement'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
