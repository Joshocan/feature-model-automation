#!/usr/bin/env python3
"""Join matching sensitivity scores to inventory metadata; report headline ranks."""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fame.evaluation.inventory import sha256


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def compare(wide, scores):
    metadata = {r['run_id']: r for r in wide}
    if len(metadata) != len(wide):
        raise ValueError('Duplicate run IDs in wide table')
    by_run = defaultdict(dict)
    for r in scores:
        key = (r['run_id'], float(r['tau']))
        if r['run_id'] not in metadata or r['matching_policy'] in by_run[key]:
            raise ValueError('Unknown run or duplicate matching score')
        by_run[key][r['matching_policy']] = r
    paired = []
    for (rid, tau), policies in sorted(by_run.items()):
        if 'independent_max' not in policies or 'one_to_one' not in policies:
            raise ValueError('Both independent_max and one_to_one are required')
        meta = metadata[rid]
        row = {k: meta[k] for k in ('run_id','model_id','corpus','N','grounding','arm','seed','repetition','strict_admissible')}
        row['tau'] = tau
        for policy, score in policies.items():
            for metric in ('precision','recall_total','f1_total'):
                row[f'{policy}__{metric}'] = float(score[metric])
        base, one = policies['independent_max'], policies['one_to_one']
        row['n_generated'] = int(base['n_generated'])
        row['n_reference'] = int(base['n_reference'])
        row['generated_reference_ratio'] = row['n_generated'] / row['n_reference']
        for metric in ('precision','recall_total','f1_total'):
            row[f'delta_one_minus_independent__{metric}'] = float(one[metric])-float(base[metric])
        row['precision_ratio_independent_to_one'] = (float(base['precision'])/float(one['precision']) if float(one['precision']) else None)
        paired.append(row)
    cells = defaultdict(list)
    for r in paired:
        if r['N'] == '10' and r['arm'] in ('guided_headline','astra_cross_corpus'):
            cells[(r['corpus'],r['grounding'],r['tau'],r['model_id'])].append(r)
    rankings = []
    for (corpus, grounding, tau, model), rs in sorted(cells.items()):
        row = dict(corpus=corpus,grounding=grounding,tau=tau,model_id=model,n=len(rs),
                   median_generated=statistics.median(r['n_generated'] for r in rs),
                   median_generated_reference_ratio=statistics.median(r['generated_reference_ratio'] for r in rs),
                   planned=sum(r['model_id']==model and r['corpus']==corpus and r['grounding']==grounding and str(r['N'])=='10' and r['arm'] in ('guided_headline','astra_cross_corpus') for r in wide))
        ratios = [r['precision_ratio_independent_to_one'] for r in rs if r['precision_ratio_independent_to_one'] is not None]
        row['median_precision_ratio_independent_to_one'] = statistics.median(ratios) if ratios else None
        for key in rs[0]:
            if any(key.startswith(p+'__') for p in ('independent_max','one_to_one','one_to_one_max_weight')):
                row[key] = statistics.mean(r[key] for r in rs)
        rankings.append(row)
    for row in rankings:
        peers = [r for r in rankings if all(r[k]==row[k] for k in ('corpus','grounding','tau'))]
        for metric in ('precision','recall_total','f1_total'):
            for policy in ('independent_max','one_to_one'):
                key = f'{policy}__{metric}'
                value = row[key]
                row[f'rank__{key}'] = 1 + sum(r[key]>value for r in peers) + (sum(r[key]==value for r in peers)-1)/2
            row[f'rank_changed__{metric}'] = row[f'rank__independent_max__{metric}'] != row[f'rank__one_to_one__{metric}']
    return paired, rankings


def write(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wide',type=Path,required=True);p.add_argument('--tau',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();allowed=REPO/'results/ifs-2027/analysis'
    if out.exists() or out==allowed or not out.is_relative_to(allowed):
        raise SystemExit('Choose a fresh directory under results/ifs-2027/analysis')
    paired,ranks=compare(read(a.wide),read(a.tau));out.mkdir(parents=True)
    write(out/'matching_comparison.csv',paired);write(out/'headline_rankings.csv',ranks)
    (out/'summary.json').write_text(json.dumps(dict(wide_sha256=sha256(a.wide),tau_sha256=sha256(a.tau),
        implementation_sha256=sha256(Path(__file__)),n_run_tau_rows=len(paired),
        note='Descriptive ranks among models with completed outputs; unequal survivor populations. No rank assigned to empty cells.'),indent=2)+'\n')


if __name__=='__main__':
    main()
