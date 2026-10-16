#!/usr/bin/env python3
"""Exploratory own/opposite-reference control from saved XML; no generation calls.

Recomputes both conditions with the pinned encoder and existing threshold scorer.
Original evaluations and the primary evaluation contract are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO
sys.path.insert(0, str(REPO))

from fame.evaluation.coverage import extract_nodes
from fame.evaluation.inventory import sha256
from fame.evaluation.semantic import cosine_similarity_matrix
from fame.evaluation.tau_rescore import rescore_run
from scripts.evaluate_semantic import CONTRACT_PATH, GROUND_TRUTH, _encoder_or_die

TAUS = (.3, .4, .5, .6, .7, .8)
POLICIES = ('independent_max', 'one_to_one')
META = ('run_id', 'corpus', 'model_id', 'arm', 'N', 'grounding',
        'ordering_id', 'k_doc', 'metamodel_block', 'seed', 'repetition')
CELL = ('corpus', 'model_id', 'arm', 'N', 'grounding', 'ordering_id',
        'k_doc', 'metamodel_block')
METRICS = ('precision', 'recall_total', 'f1_total')


def yes(value):
    return str(value).lower() == 'true'


def headline(run):
    return (int(run['N']) == 10 and yes(run['metamodel_block'])
            and run['ordering_id'] == 'primary'
            and (run['grounding'] == 'nonrag' or run['k_doc'] == 5)
            and run['arm'] in ('guided_headline', 'astra_cross_corpus'))


def pairs_from_matrix(run_id, generated, reference, matrix):
    """Preserve occurrence indices, including duplicate feature names."""
    if matrix.shape != (len(generated), len(reference)):
        raise ValueError('Similarity matrix shape mismatch')
    rows = []
    for i, name in enumerate(generated):
        for j, target in enumerate(reference):
            value = float(matrix[i, j])
            if not math.isfinite(value) or not -1.00001 <= value <= 1.00001:
                raise ValueError('Invalid cosine similarity')
            rows.append(dict(run_id=run_id, gen_index=i, gen_name=name,
                             ref_index=j, ref_name=target, similarity=value))
    return rows


def score_conditions(run, generated, references, matrices, taus):
    own = run['corpus']
    cross = 'federation' if own == 'repair' else 'repair'
    if not generated or any(not references[c] for c in (own, cross)):
        raise ValueError('Empty feature set is not a measured zero score')
    paired = {condition: pairs_from_matrix(run['run_id'], generated, references[c], matrices[c])
              for condition, c in (('own', own), ('cross', cross))}
    rows = []
    for tau in taus:
        for policy in POLICIES:
            row = {k: run[k] for k in META}
            row.update(tau=tau, matching_policy=policy, status='ok',
                       own_reference_corpus=own, cross_reference_corpus=cross,
                       n_generated=len(generated), n_reference_own=len(references[own]),
                       n_reference_cross=len(references[cross]),
                       size_ratio_own=len(generated)/len(references[own]))
            for condition in ('own', 'cross'):
                result = rescore_run(paired[condition], tau=tau, matching_policy=policy)
                for key in (*METRICS, 'n_generated_matched', 'n_reference_matched', 'n_matched_pairs'):
                    row[f'{key}_{condition}'] = getattr(result, key)
            for key in METRICS:
                row[f'{key}_own_minus_cross'] = row[f'{key}_own'] - row[f'{key}_cross']
            rows.append(row)
    return rows


def summarise(runs, scores, taus, *, headline_only=False):
    """Unweighted run means, own-reference median size, and average-tie ranks.

    Include planned cells with no scores: their means/ranks remain null, not zero.
    Both reference conditions always use the same eligible outputs in a cell.
    """
    keys = ('corpus', 'grounding', 'model_id') if headline_only else CELL
    groups, scored = defaultdict(list), defaultdict(list)
    selected = [r for r in runs if not headline_only or headline(r)]
    ids = {r['run_id'] for r in selected}
    for run in selected:
        groups[tuple(run[k] for k in keys)].append(run)
    for row in scores:
        if row['run_id'] in ids:
            scored[(*[row[k] for k in keys], row['tau'], row['matching_policy'])].append(row)
    result = []
    for key, planned in groups.items():
        for tau in taus:
            for policy in POLICIES:
                cell = scored.get((*key, tau, policy), [])
                row = dict(zip(keys, key))
                row.update(tau=tau, matching_policy=policy, planned=len(planned),
                           completed=sum(yes(r['completed']) for r in planned), scored=len(cell),
                           median_size_ratio_own=statistics.median(r['size_ratio_own'] for r in cell) if cell else None,
                           status='ok' if cell else 'no_eligible_outputs', descriptive_only=True)
                for metric in METRICS:
                    for suffix in ('own', 'cross', 'own_minus_cross'):
                        field = f'{metric}_{suffix}'
                        row[f'mean_{field}'] = statistics.mean(r[field] for r in cell) if cell else None
                result.append(row)
    if headline_only:
        for row in result:
            peers = [p for p in result if p['scored'] and
                     all(p[k] == row[k] for k in ('corpus', 'grounding', 'tau', 'matching_policy'))]
            for condition in ('own', 'cross'):
                value = row[f'mean_f1_total_{condition}']
                values = [p[f'mean_f1_total_{condition}'] for p in peers]
                row[f'rank_f1_{condition}'] = (1 + sum(v > value for v in values)
                    + (sum(v == value for v in values)-1)/2) if value is not None else None
            row['rank_changed'] = (row['rank_f1_own'] != row['rank_f1_cross']) if row['scored'] else None
    return result


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PreparedEncoder:
    """Reuse per-run generated embeddings and the two fixed reference embeddings."""
    def __init__(self, values):
        self.values = values

    def encode(self, texts, **kwargs):
        return self.values[tuple(texts)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tau', type=float, nargs='+', default=list(TAUS))
    parser.add_argument('--limit', type=int, help='Smoke check only; partial results must not be used for Table 2')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    allowed = REPO / 'results/ifs-2027/analysis'
    if out.exists() or out == allowed or not out.is_relative_to(allowed):
        parser.error('Choose a NEW subdirectory under results/ifs-2027/analysis')
    if len(set(args.tau)) != len(args.tau) or any(not math.isfinite(t) or not 0 <= t <= 1 for t in args.tau):
        parser.error('Thresholds must be unique finite numbers between 0 and 1')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    paths = {name: args.inventory / name for name in ('runs.json', 'artifacts.json', 'summary.json')}
    inventory = json.loads(paths['summary.json'].read_text())
    if inventory['contract_sha256'] != sha256(CONTRACT_PATH) or inventory['issues']:
        raise ValueError('Inventory contract mismatch or unresolved integrity issues')
    runs = json.loads(paths['runs.json'].read_text())
    if len(runs) != inventory['planned'] or len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Inventory count or duplicate run IDs')
    if any(r['corpus'] not in GROUND_TRUTH for r in runs):
        raise ValueError('Unknown output corpus')
    artifacts = json.loads(paths['artifacts.json'].read_text())
    indexed = {a['resolved_path']: a for a in artifacts
               if a['population'] == 'main' and a['snapshot'] == 'selected'}
    inputs = {str(p.resolve()): sha256(p) for p in (*paths.values(), CONTRACT_PATH, *GROUND_TRUTH.values())}
    references = {c: [name for name, _ in extract_nodes(p)] for c, p in GROUND_TRUTH.items()}
    if any(not names for names in references.values()):
        raise ValueError('Empty reference')
    eligible, exclusions = [], []
    for run in runs:
        if not yes(run['completed']):
            exclusions.append(dict(run_id=run['run_id'], status='ineligible', reason='No completed final output'))
            continue
        relative = run['resolved_path'] + '/fm_gen.xml'
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or relative not in indexed or not path.is_file() or sha256(path) != indexed[relative]['sha256']:
            raise ValueError(f"Missing or changed final XML: {run['run_id']}")
        inputs[str(path)] = indexed[relative]['sha256']
        try:
            names = [name for name, _ in extract_nodes(path)]
            if not names:
                raise ValueError('No extractable feature nodes')
        except Exception as exc:
            exclusions.append(dict(run_id=run['run_id'], status='unextractable', reason=str(exc)))
            continue
        eligible.append((run, names))
    if not eligible:
        raise ValueError('No eligible completed outputs')
    if args.limit:
        for run, _ in eligible[args.limit:]:
            exclusions.append(dict(run_id=run['run_id'], status='not_evaluated', reason='Smoke-check limit'))
        eligible = eligible[:args.limit]
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    encoder, identity = _encoder_or_die(json.loads(CONTRACT_PATH.read_text()))
    fixed = {tuple(names): encoder.encode(names, normalize_embeddings=True, convert_to_tensor=False)
             for names in references.values()}
    out.mkdir(parents=True)
    scores = []
    for index, (run, names) in enumerate(eligible, 1):
        print(f"[{index}/{len(eligible)}] {run['run_id']} {run['corpus']}", flush=True)
        prepared = PreparedEncoder({**fixed, tuple(names): encoder.encode(names, normalize_embeddings=True, convert_to_tensor=False)})
        matrices = {c: cosine_similarity_matrix(ref, names, model=prepared) for c, ref in references.items()}
        scores.extend(score_conditions(run, names, references, matrices, args.tau))
    write_csv(out / 'scores.csv', scores)
    write_csv(out / 'cell_summary.csv', summarise(runs, scores, args.tau))
    write_csv(out / 'headline_rankings.csv', summarise(runs, scores, args.tau, headline_only=True))
    write_csv(out / 'exclusions.csv', exclusions, ['run_id', 'status', 'reason'])
    manifest = dict(inputs=inputs, implementation={p: sha256(SOURCE_ROOT / p) for p in (
        'scripts/evaluate_cross_reference.py', 'scripts/evaluate_semantic.py',
        'fame/evaluation/semantic.py', 'fame/evaluation/tau_rescore.py',
        'fame/evaluation/coverage.py', 'fame/evaluation/local_encoder.py')})
    (out / 'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    summary = dict(analysis='post_hoc_cross_reference_control', planned=len(runs),
        completed=sum(yes(r['completed']) for r in runs), scored=len(eligible), score_rows=len(scores),
        partial=args.limit is not None, tau=args.tau, policies=POLICIES, encoder_identity=identity,
        reference_counts={c: len(n) for c, n in references.items()},
        one_to_one_objective='maximum cardinality, then maximum total cosine similarity',
        population='completed extractable outputs; no strict-admissibility filter',
        aggregation='unweighted run means; rankings within output corpus and grounding; average exact ties',
        caveat='Related-domain comparison, not a guaranteed unrelated null; unequal reference sizes and survivor counts. All rankings descriptive.',
        outputs={p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Saved {len(scores)} paired score rows in {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
