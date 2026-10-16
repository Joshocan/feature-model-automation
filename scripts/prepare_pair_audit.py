#!/usr/bin/env python3
"""Freeze a blinded, generated-side best-match audit from saved own-reference pairs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fame.evaluation.inventory import sha256
from scripts.tau_rescore import _iter_run_pairs

BANDS = ('[0.4,0.5)', '[0.5,0.6)', '[0.6,0.7)', '[0.7,0.8)', '[0.8,1]')
LABELS = ('same_concept', 'related', 'unrelated', 'cannot_judge')
PUBLIC = ('item_id', 'corpus', 'generated_feature', 'reference_feature')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def band_for(value):
    if value < .4:
        return None
    for index, upper in enumerate((.5, .6, .7, .8)):
        if value < upper:
            return BANDS[index]
    return BANDS[-1]


def best_pairs(pairs):
    """Validate the full Cartesian matrix; use lowest reference index on exact ties."""
    seen, gens, refs, best = set(), {}, {}, {}
    for p in pairs:
        g, r = p['gen_index'], p['ref_index']
        if g < 0 or r < 0 or (g, r) in seen:
            raise ValueError('Duplicate or negative occurrence index')
        seen.add((g, r))
        for mapping, index, name in ((gens, g, p['gen_name']), (refs, r, p['ref_name'])):
            if index in mapping and mapping[index] != name:
                raise ValueError('Inconsistent occurrence name')
            mapping[index] = name
        old = best.get(g)
        if old is None or (-p['similarity'], r) < (-old['similarity'], old['ref_index']):
            best[g] = p
    if len(seen) != len(gens)*len(refs):
        raise ValueError('Incomplete Cartesian similarity matrix')
    return [best[g] for g in sorted(best) if band_for(best[g]['similarity']) is not None]


def sample_pairs(candidates, per_band, seed):
    pools = {b: [] for b in BANDS}
    for p in candidates:
        pools[band_for(p['similarity'])].append(p)
    rng, selected, counts = random.Random(seed), [], []
    for band, pool in pools.items():
        pool.sort(key=lambda p: (p['run_id'], p['gen_index']))
        n = min(per_band, len(pool))
        selected.extend({**p, 'band': band} for p in rng.sample(pool, n))
        counts.append(dict(band=band, population=len(pool), requested=per_band,
                           sampled=n, shortage=per_band-n))
    rng.shuffle(selected)  # Item IDs and visible order must not encode the band.
    return [{**p, 'item_id': f'P{i:03d}'} for i, p in enumerate(selected, 1)], counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--per-band', type=int, default=20)
    args = parser.parse_args(argv)
    if args.output.exists() or args.per_band < 1:
        parser.error('Use a fresh output directory and positive --per-band')
    runs_path = args.inventory / 'runs.json'
    runs = json.loads(runs_path.read_text())
    metadata = {r['run_id']: r for r in runs}
    if len(metadata) != len(runs):
        raise ValueError('Duplicate inventory IDs')
    summary_path = args.pairs.parent / 'summary.json'
    semantic = json.loads(summary_path.read_text())
    if semantic['inventory_sha256'] != sha256(runs_path):
        raise ValueError('Pairs summary and inventory identities differ')
    candidates, seen = [], set()
    for pairs in _iter_run_pairs(args.pairs):
        rid = pairs[0]['run_id']
        run = metadata.get(rid)
        if run is None or run['completed'] is not True:
            raise ValueError(f'Unknown or noncompleted run: {rid}')
        if any(p['corpus'] != run['corpus'] for p in pairs):
            raise ValueError('Pair corpus differs from inventory')
        seen.add(rid)
        candidates.extend({**p, 'model_id': run['model_id']} for p in best_pairs(pairs))
    if len(seen) != semantic['semantic_scored']:
        raise ValueError('Scored-run count differs from semantic summary')
    completed = {r['run_id'] for r in runs if r['completed'] is True}
    omitted = sorted(completed-seen)
    if omitted:
        raise ValueError('Completed runs missing pairs; resolve eligibility explicitly before sampling')
    selected, counts = sample_pairs(candidates, args.per_band, args.seed)
    if not selected:
        raise ValueError('No eligible best matches')
    args.output.mkdir(parents=True)
    private = args.output / 'AUTHOR_ONLY'
    private.mkdir()
    public = [{k: p[k] for k in PUBLIC} for p in (
        dict(item_id=r['item_id'], corpus=r['corpus'], generated_feature=r['gen_name'],
             reference_feature=r['ref_name']) for r in selected)]
    key = [{**p, 'run_id': r['run_id'], 'model_id': r['model_id'],
            'gen_index': r['gen_index'], 'ref_index': r['ref_index'],
            'similarity': r['similarity'], 'band': r['band']} for p, r in zip(public, selected)]
    write_csv(private / 'sample_key.csv', key, list(key[0]))
    write_csv(private / 'sampling_counts.csv', counts, list(counts[0]))
    for rater in ('R01', 'R02'):
        rows = [{**p, 'label': '', 'comment': ''} for p in public]
        write_csv(args.output / f'ratings_{rater}.csv', rows, [*PUBLIC, 'label', 'comment'])
    (args.output / 'RATER_INSTRUCTIONS.md').write_text(
        '# Concept correspondence rating\n\n'
        'Work independently. Compare each generated feature with the reference feature '
        'in the stated domain (model repair or model federation). Judge concepts, not spelling. '
        'Do not search for the producing model or automated scores. This is a names-only audit; '
        'do not add descriptions or hierarchy selectively.\n\n'
        'Fill `label` using exactly one of:\n\n'
        '- `same_concept`: interchangeable concepts in this domain, including synonyms.\n'
        '- `related`: associated, broader/narrower, subtype or component, but not equivalent.\n'
        '- `unrelated`: different concepts with no meaningful relationship here.\n'
        '- `cannot_judge`: names and domain do not provide enough information.\n\n'
        'Optional explanations go in `comment`. Complete every label; do not change IDs '
        'or feature text. Repeated name pairs are intentional occurrences; rate every row. '
        'Do not discuss ratings until both original files have been saved.\n', encoding='utf-8')
    manifest = dict(audit='post_hoc_generated_side_best_match_own_reference', seed=args.seed,
        per_band=args.per_band, sampled=len(selected), eligible_occurrences=len(candidates),
        scored_runs=len(seen), bands=BANDS, counts=counts, labels=LABELS,
        unit='generated node occurrence; repeated name pairs retained',
        tie_break='lowest reference occurrence index for exact similarity ties',
        presentation='feature names and domain only; model/run/score/band hidden',
        inputs={str(p.resolve()): sha256(p) for p in (args.pairs, runs_path, summary_path)},
        implementation_sha256=sha256(Path(__file__)),
        frozen_files={str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()})
    (private / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Sampled {len(selected)} items. Send each rater only their ratings CSV and RATER_INSTRUCTIONS.md.')
    print('Keep AUTHOR_ONLY private. Shortages:', {r['band']: r['shortage'] for r in counts if r['shortage']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
