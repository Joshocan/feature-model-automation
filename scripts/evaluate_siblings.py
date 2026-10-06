#!/usr/bin/env python3
"""Export sibling P/R/F1 and parent audit from saved similarities, no model calls."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fame.evaluation.inventory import sha256
from fame.evaluation.sibling_agreement import unpack_pairs, correspondence, sibling_metrics
from scripts.tau_rescore import _iter_run_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tau', type=float, default=0.4)
    p.add_argument('--run-id', nargs='+', help='Optional small audit subset; omitted = all saved runs')
    a = p.parse_args()
    allowed = REPO / 'results/ifs-2027/analysis'
    out = a.output.resolve()
    if out.exists() or out == allowed or not out.is_relative_to(allowed):
        p.error('Choose a fresh subdirectory under results/ifs-2027/analysis')
    if not 0 <= a.tau <= 1:
        p.error('tau must be between 0 and 1')
    selected = set(a.run_id or [])
    scores, audits, seen = [], [], set()
    for rows in _iter_run_pairs(a.pairs):
        rid = rows[0]['run_id']
        if selected and rid not in selected:
            continue
        seen.add(rid)
        matrix, generated, reference = unpack_pairs(rows)
        for policy in ('independent_max', 'one_to_one', 'exact_unique'):
            mapping = correspondence(matrix, generated, reference, policy, a.tau)
            meta = dict(run_id=rid, corpus=rows[0]['corpus'], matching_policy=policy,
                        tau=a.tau if policy != 'exact_unique' else None)
            scores.append(dict(meta, **sibling_metrics(generated, reference, mapping)))
            for i, j in mapping.items():
                gp, rp = generated[i]['parent_index'], reference[j]['parent_index']
                mapped_parent = mapping.get(gp)
                eligible = gp is not None and rp is not None and mapped_parent is not None
                audits.append(dict(meta, gen_index=i, gen_name=generated[i]['name'],
                    gen_parent_index=gp, gen_parent_name=generated[gp]['name'] if gp is not None else '',
                    ref_index=j, ref_name=reference[j]['name'],
                    ref_parent_index=rp, ref_parent_name=reference[rp]['name'] if rp is not None else '',
                    similarity=float(matrix[i,j]), mapped_parent_ref_index=mapped_parent,
                    mapped_parent_ref_name=reference[mapped_parent]['name'] if mapped_parent is not None else '',
                    parent_similarity=float(matrix[gp,mapped_parent]) if gp is not None and mapped_parent is not None else None,
                    parent_evaluable=eligible, parent_correct=mapped_parent == rp if eligible else None,
                    literal_parent_name_equal=generated[gp]['name'] == reference[rp]['name'] if gp is not None and rp is not None else None))
        print(f'{rid}: scored three matching policies', flush=True)
    if not seen or selected - seen:
        raise ValueError(f'No scores or requested runs absent: {selected-seen}')
    out.mkdir(parents=True)
    for filename, data in [('sibling_metrics.csv', scores), ('node_alignment_audit.csv', audits)]:
        with (out / filename).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]) if data else [])
            w.writeheader(); w.writerows(data)
    summary = dict(n_runs=len(seen), n_scores=len(scores), tau=a.tau,
        exploratory=True, pairs_sha256=sha256(a.pairs), run_ids=sorted(seen),
        definition='Sibling P/R/F1 conditional on matched non-root generated pairs with distinct reference counterparts; parent labels ignored.',
        policies=['independent_max', 'one_to_one', 'exact_unique'],
        note='Exact unique excludes duplicate names and does not use tau. One-to-one maximises cardinality then weight. No replacement of existing parent-match or inferential families.',
        implementation_sha256={path:sha256(REPO/path) for path in ['scripts/evaluate_siblings.py','scripts/tau_rescore.py','fame/evaluation/sibling_agreement.py']},
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy']},
        outputs_sha256={name:sha256(out/name) for name in ['sibling_metrics.csv','node_alignment_audit.csv']})
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Saved {len(scores)} scores for {len(seen)} runs to {out}')


if __name__ == '__main__':
    main()
