#!/usr/bin/env python3
"""Read-only saved-output reanalysis, in gated order 4, 1, 2, 3.

Writes only new results/reanalysis_2026-10-15* files. No model calls.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fame.evaluation.alignment import extract_indexed_nodes
from fame.evaluation.reachable import load_attribution
from fame.evaluation.quality_sat import _build_cnf, _solve, FEATURE_TAGS
from fame.evaluation.structural import evaluate_structure
from scripts.evaluate_semantic import _partition_and_reach_for

BASE = ROOT / 'results/ifs-2027/analysis'
OUT = ROOT / 'results/reanalysis_2026-10-15'
REPORT = OUT.with_suffix('.md')
SEED = 20261015
INPUTS = {}
SECTIONS = []


def check(ok, message):
    if not ok:
        raise ValueError(message)


def record(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_csv(path):
    with record(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def ci(values, key):
    """Deterministic per-statistic stream, whole-run percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(mean=None, ci_lower=None, ci_upper=None)
    check(np.isfinite(values).all(), f'Nonfinite bootstrap data: {key}')
    salt = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, salt]))
    draws = values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], method='linear')
    return dict(mean=statistics.fmean(values), ci_lower=float(lo), ci_upper=float(hi))


def fmt(v):
    if v is None:
        return 'NA'
    if isinstance(v, (float, np.floating)):
        return f'{v:.3f}'
    return str(v)


def output(name, rows):
    path = OUT / (name + '.csv')
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: fmt(v) for k, v in row.items()} for row in rows)
    return rows


def table(rows):
    fields = list(rows[0])
    return '\n'.join(['| ' + ' | '.join(fields) + ' |',
                      '| ' + ' | '.join('---' for _ in fields) + ' |'] +
                     ['| ' + ' | '.join(fmt(r[k]) for k in fields) + ' |' for r in rows])


def flush(status):
    REPORT.write_text('# Reanalysis — 2026-10-15\n\n' + status + '\n\n'
        'Saved completed outputs only; no generation or embedding calls. '
        'Run means use 5,000 whole-run resamples, seed 20261015, '
        '95% linear percentile intervals. Runs are sorted by ID. '
        'Each statistic has a stable SHA-256-derived random stream. '
        'These pooled intervals do not adjust for dependence between configurations '
        'or survivor selection. NA denotes an empty denominator, never zero.\n\n'
        + '\n\n'.join(SECTIONS) + '\n')
    (OUT / 'input_hashes.json').write_text(json.dumps(INPUTS, indent=2) + '\n')


def select(rows, selector):
    return [r for r in rows if all(str(r.get(k)).lower() == str(v).lower()
                                  for k, v in selector.items())]


def task4(wide):
    result, deltas, pending_ci = [], [], []
    cells = {}
    for family in ('A_grounding', 'C_ablation'):
        path = ROOT / f'config/analysis/families/{family}.json'
        declaration = json.loads(record(path).read_text())
        for comp in declaration['comparisons']:
            endpoint = ('semantic__semantic_f1_total' if family == 'A_grounding'
                        else 'structural__structural_conformance')
            if comp['metric'] != endpoint:
                continue
            pair = []
            for arm in ('arm_a', 'arm_b'):
                subset = select(wide, comp[arm])
                completed = [r for r in subset if r['inventory_status'] == 'completed']
                check(all(r['structural__structural_conformance__status'] == 'ok'
                          for r in completed), 'Missing conformance observation')
                nconf = sum(r['structural__structural_conformance'] == 'True' for r in subset)
                sel = comp[arm]
                label = (sel['grounding'] if family == 'A_grounding' else
                         'guided' if sel['metamodel_block'] else 'ablated')
                row = dict(family=family, corpus=sel['corpus'], model=sel['model_id'],
                           N=sel['N'], condition=label, planned=len(subset),
                           completed=len(completed), conformant=nconf,
                           conformant_completed=nconf/len(completed) if completed else None,
                           conformant_planned=nconf/len(subset))
                cells[family, sel['model_id'], sel['corpus'], sel['N'], label] = row
                for pop, poprows in [('completed', completed), ('planned', subset)]:
                    values = [int(r['structural__structural_conformance'] == 'True') for r in poprows]
                    pending_ci.append((row, pop, values, str((comp['name'], arm, pop))))
                pair.append(row)
                result.append(row)
            a, b = pair
            # Delta-Conf: percentage-point difference conditional on completion.
            delta = 100 * (a['conformant_completed'] - b['conformant_completed']) if a['completed'] and b['completed'] else None
            deltas.append(dict(family=family, corpus=a['corpus'], model=a['model'], N=a['N'],
                               completed_a=a['completed'], completed_b=b['completed'],
                               delta_conf_completed_pp=delta,
                               delta_conf_planned_pp=100*(a['conformant_planned']-b['conformant_planned'])))
    ds, glm, astra = 'deepseek-v4.1-flash:cloud', 'glm-5.3-flash:cloud', 'gpt-6-astra'
    for family, model, corpus, n, a, b, expected in [
        ('A_grounding', ds, 'federation', 10, 'rag', 'nonrag', (12,8)),
        ('A_grounding', glm, 'repair', 10, 'rag', 'nonrag', (18,14)),
        ('C_ablation', astra, 'repair', 10, 'guided', 'ablated', (7,9))]:
        actual = tuple(cells[family,model,corpus,n,label]['completed'] for label in (a,b))
        check(actual == expected, f'Task 4 counts mismatch: {actual} != {expected}')
    for family, model, corpus, n, expected in [
        ('A_grounding',ds,'federation',10,58), ('A_grounding',glm,'federation',10,28),
        ('A_grounding',glm,'repair',10,32), ('C_ablation',glm,'repair',10,31),
        ('C_ablation',glm,'federation',1,27), ('C_ablation',astra,'repair',10,0)]:
        row = next(r for r in deltas if (r['family'],r['model'],r['corpus'],r['N']) == (family,model,corpus,n))
        check(abs(row['delta_conf_completed_pp']-expected) <= .5, f'Task 4 Delta-Conf mismatch: {row}')
    for row, pop, values, key in pending_ci:
        stats = ci(values, key)
        row[pop+'_ci_lower'], row[pop+'_ci_upper'] = stats['ci_lower'], stats['ci_upper']
    output('task4_conformance_denominators', result)
    output('task4_conformance_differences', deltas)
    source = Path(__file__).read_text().splitlines()
    lineno = next(i for i,s in enumerate(source,1) if s.strip().startswith('delta = 100'))
    SECTIONS.append('## Task 4 — conformance denominators\n\nSanity checks passed. '
        'All declared grounding cells and metamodel cells (N=1 and N=10 where planned) are included. '
        'The published examples reproduce **100 × (conformant_A/completed_A − conformant_B/completed_B)**. '
        'A is RAG/guided; B is non-RAG/ablated.\n\n' + table(result) + '\n\n' + table(deltas) +
        '\n\nNo original manuscript-table Delta-Conf calculation was found in the repository. '
        'The saved-analysis code filters metric status to ok (`scripts/analyse_family.py:172`) '
        'and computes `return (sum(values) / len(values)) if values else None` '
        '(`fame/evaluation/statistical_family.py:82`), i.e. the mean of observed Boolean conformance. '
        f'The explicit reconstruction is `scripts/reanalyse_2026_10_15.py:{lineno}`:\n\n'
        '```python\n' + source[lineno-1].strip() + '\n```\n\n'
        'This reproduces the published differences, but is not a claim that this newly added line generated the original paper table. '
        'Conformance differences use completed denominators; completion rates must be reported separately. '
        'For Astra Repair grounding, conditional non-RAG conformance and its difference are undefined (zero completed outputs).')


def task1(runs, wide, nodes, refs, info):
    audits = read_csv(BASE/'sibling-current-v1/node_alignment_audit.csv')
    alignments = defaultdict(dict)
    for r in audits:
        if r['matching_policy'] == 'independent_max':
            rid, i, j = r['run_id'], int(r['gen_index']), int(r['ref_index'])
            check(nodes[rid][i]['name'] == r['gen_name'] and refs[r['corpus']][j]['name'] == r['ref_name'], 'Alignment/XML mismatch')
            check(i not in alignments[rid], 'Duplicate alignment')
            alignments[rid][i] = (j, float(r['similarity']))
    attrs = {c: load_attribution(record(ROOT/f'data/attribution/{c}.csv')) for c in refs}

    def calculate(mode):
        rows = []
        for rid, run in runs.items():
            c = run['corpus']; generated = nodes[rid]; reference = refs[c]
            if mode == 'exact_unique':
                gc = Counter(x['name'] for x in generated); rc = Counter(x['name'] for x in reference)
                lookup = {x['name']: j for j,x in enumerate(reference) if rc[x['name']] == 1}
                mapping = {i: lookup[x['name']] for i,x in enumerate(generated) if gc[x['name']] == 1 and x['name'] in lookup}
            else:
                threshold = .4 if mode == 'independent_max_0.4' else .8
                mapping = {i:j for i,(j,s) in alignments[rid].items() if s >= threshold}
            applicable = agreeing = 0; baseline = 0.
            for i,j in mapping.items():
                docs = attrs[c].get(reference[j]['name'], set())
                if not docs:
                    continue
                check(docs <= set(info[c]['manifest_ids']), 'Attribution contains unknown documents')
                for doc in generated[i]['citations']:
                    applicable += 1; agreeing += int(doc in docs)
                    baseline += len(docs)/len(info[c]['manifest_ids'])
            rows.append(dict(run_id=rid, corpus=c, alignment=mode, applicable=applicable,
                             agreeing=agreeing, baseline_expected=baseline,
                             agreement=agreeing/applicable if applicable else None,
                             baseline=baseline/applicable if applicable else None,
                             difference=(agreeing-baseline)/applicable if applicable else None))
        return rows
    unrestricted = calculate('independent_max_0.4')
    sanity = []
    for c, n, agree, base in [('repair',39465,38.0,32.5),('federation',13473,34.6,34.5)]:
        rr = [r for r in unrestricted if r['corpus']==c]
        total = sum(r['applicable'] for r in rr)
        actual_agree = 100*sum(r['agreeing'] for r in rr)/total
        actual_base = 100*sum(r['baseline_expected'] for r in rr)/total
        sanity.append(dict(corpus=c, applicable=total, agreement_percent=actual_agree, baseline_percent=actual_base))
        if not (total == n and abs(actual_agree-agree)<=.05+1e-10 and abs(actual_base-base)<=.05+1e-10):
            output('task1_failed_sanity',sanity)
            eligible=[r for r in rr if r['applicable']]
            run_baseline=100*statistics.fmean(r['baseline'] for r in eligible)
            feature_baseline=100*statistics.fmean(len(d) for d in attrs[c].values() if d)/len(info[c]['manifest_ids'])
            SECTIONS.append('## Task 1 — stopped at unrestricted sanity check\n\n'+table(sanity)+
                f'\n\nExpected baseline: {base:.3f}%. Recomputed citation-weighted uniform-document baseline: {actual_base:.3f}%. '
                'For each applicable citation, chance agreement is |attribution(aligned feature)| / corpus document count. '
                f'The equal-run mean baseline is {run_baseline:.3f}% across {len(eligible)} contributing runs; '
                f'equal weighting of attested reference features instead yields {feature_baseline:.3f}%. '
                'Neither alternative reproduces the requested published baseline. The repository search found no original RQ4 uniform-baseline calculation. '
                'The discrepancy therefore needs reconciliation with that calculation or its original inputs; it is not a rounding difference. '
                'No restricted attribution intervals or Tasks 2–3 were computed.\n\n'
                '### Implications for paper claims\n\n'
                'Task 4 confirms conditional-on-completion conformance differences, not improvements in conformant yield per planned run. '
                'For Astra Repair metamodel, the conditional difference is 0.000 pp, while the planned-run difference is −20.000 pp. '
                'Do not revise the RQ4 chance-adjusted claim until the baseline discrepancy is resolved; no conclusions from the unrun tasks are available.')
            raise ValueError(f'Task 1 unrestricted sanity failed: {sanity[-1]}; expected {n}, {agree:.3f}%, {base:.3f}% (rounding tolerance 0.050 pp)')
        for r in rr:
            w = wide[r['run_id']]
            check(r['applicable'] == int(w['semantic__n_attribution_applicable']) and r['agreeing'] == int(w['semantic__n_attribution_agreeing']), 'Per-run attribution count mismatch')
    output('task1_sanity',sanity)
    allrows = unrestricted + calculate('independent_max_0.8') + calculate('exact_unique')
    summaries = []
    for mode in ('independent_max_0.4','independent_max_0.8','exact_unique'):
        for c in refs:
            rr = [r for r in allrows if r['corpus']==c and r['alignment']==mode and r['applicable']]
            total = sum(r['applicable'] for r in rr)
            agreement = 100*sum(r['agreeing'] for r in rr)/total if total else None
            baseline = 100*sum(r['baseline_expected'] for r in rr)/total if total else None
            summaries.append(dict(corpus=c, alignment=mode, applicable_citations=total,
                contributing_runs=len(rr), pooled_agreement_percent=agreement, baseline_percent=baseline,
                pooled_difference_pp=agreement-baseline if total else None,
                **{'run_difference_pp_'+k:v for k,v in ci([100*r['difference'] for r in rr], c+mode).items()}))
    output('task1_run_attribution',allrows); output('task1_attribution_summary',summaries)
    SECTIONS.append('## Task 1 — attribution at high-confidence alignments\n\nUnrestricted sanity checks passed, including each run’s saved numerator and denominator. '
        'Uniform-document baseline for each applicable citation is the number of distinct documents attributed to its aligned reference feature divided by corpus document count (54/23). '
        'Pooled rates weight citations; run-level mean differences weight contributing runs equally. Empty-applicability runs are excluded, not scored zero. '
        'Exact unique means literal name equality occurring once in each XML tree, without a similarity threshold.\n\n'+table(summaries))


def task2(runs, wide, info):
    scores = read_csv(BASE/'tau-matching-v2/tau_sweep.csv')
    cross = {(r['run_id'],r['matching_policy']):r for r in read_csv(BASE/'cross-reference-v1/scores.csv') if float(r['tau'])==.4}
    rows=[]; seen=set()
    for c, sizes in [('repair',(58,42,84)),('federation',(36,94,59))]:
        p=info[c]['partition']
        check((len(p.attested),len(p.organising),len(info[c]['reach']))==sizes, 'Task 2 partition sizes mismatch')
    for s in scores:
        if float(s['tau']) != .4 or s['matching_policy'] not in ('independent_max','one_to_one'):
            continue
        rid, policy=s['run_id'],s['matching_policy']; check((rid,policy) not in seen,'Duplicate recall row'); seen.add((rid,policy))
        c=runs[rid]['corpus']; p=info[c]['partition']; matched=set(json.loads(s['matched_reference_ids']))
        full=len(matched)/len(p.full)
        check(matched <= p.full and len(matched)==int(s['n_reference_matched']), 'Invalid matched-reference set')
        check(math.isclose(full,float(s['recall_total']),abs_tol=1e-12,rel_tol=0), 'Task 2 saved recall mismatch')
        check(math.isclose(full,float(cross[rid,policy]['recall_total_own']),abs_tol=1e-12,rel_tol=0), 'Task 2 cross-reference own recall mismatch')
        if policy=='independent_max':
            check(math.isclose(full,float(wide[rid]['semantic__semantic_recall_total']),abs_tol=1e-12,rel_tol=0), 'Task 2 primary recall mismatch')
        rows.append(dict(run_id=rid,corpus=c,policy=policy,recall_full=full,
                         recall_attested=len(matched & p.attested)/len(p.attested),
                         recall_organising=len(matched & p.organising)/len(p.organising),
                         recall_reach=len(matched & info[c]['reach'])/len(info[c]['reach'])))
    check(seen=={(rid,p) for rid in runs for p in ('independent_max','one_to_one')},'Task 2 incomplete population')
    rows.sort(key=lambda r:(r['run_id'],r['policy']))
    summaries=[]
    for c in info:
        for policy in ('independent_max','one_to_one'):
            rr=[r for r in rows if r['corpus']==c and r['policy']==policy]
            for part in ('full','attested','organising','reach'):
                size=len(info[c]['reach']) if part=='reach' else len(getattr(info[c]['partition'],part))
                summaries.append(dict(corpus=c,policy=policy,partition=part,reference_features=size,n=len(rr),
                                      **ci([r['recall_'+part] for r in rr],c+policy+part)))
    output('task2_run_recall',rows);output('task2_recall_summary',summaries)
    SECTIONS.append('## Task 2 — partitioned recall\n\nAll 1,000 run/policy full-reference recalls match saved tau scores and independently saved own-reference control scores (absolute tolerance 1e-12); independent-max also matches primary semantic scores. '
        'Partition sizes match the requested values. Independent-max uses all above-threshold reference hits; one-to-one uses its saved matched reference set. '
        'Reach is ancestor-closed and overlaps both attested and organising partitions; it is not a third disjoint class.\n\n'+table(summaries))


def task3(runs, wide, paths):
    conform=[rid for rid in runs if wide[rid]['structural__structural_conformance']=='True']
    unsat=[rid for rid in conform if wide[rid]['structural__satisfiable']=='False']
    check(len(conform)==381 and sum(wide[r]['structural__satisfiable']=='True' for r in conform)==369
          and len(unsat)==12 and all(runs[r]['model_id']=='glm-5.3-flash:cloud' for r in unsat),
          'Task 3 conformance/SAT/GLM sanity failed')
    rows=[]; diagnoses=[]; wchecks=[]
    xsd=record(ROOT/'prompts/feature-model-schema.xsd')
    for rid,run in runs.items():
        metrics=evaluate_structure(paths[rid],xsd,expected_root=run['root_feature'],run_sat=False)
        for w in ('W1','W3','W5'):
            m=metrics[w]; saved=wide[rid]['structural__'+w]
            check(m['status']==wide[rid]['structural__'+w+'__status'] and str(m['value'])==saved,'Task 3 W-check changed: '+rid+' '+w)
            wchecks.append(dict(run_id=rid,corpus=run['corpus'],check=w,value=m['value'],status=m['status'],reason=m['reason']))
        if rid not in conform:continue
        root=ET.parse(paths[rid]).getroot()
        nctc=len(root.findall('constraints/rule')); labels=set(); mandatory=[]
        for parent in root.find('struct').iter():
            if parent.tag not in ('or','alt'):continue
            for child in parent:
                if child.tag in FEATURE_TAGS and child.get('mandatory','').lower() in ('true','1'):
                    mandatory.append((parent.get('name'),child.get('name'),parent.tag))
                    labels.add(f"Mandatory relation: selecting '{parent.get('name')}' requires child '{child.get('name')}'.")
        enc,_=_build_cnf(paths[rid]); sat=_solve(enc.clauses)
        check(sat==(wide[rid]['structural__satisfiable']=='True'),'Task 3 SAT reproduction mismatch '+rid)
        rows.append(dict(run_id=rid,corpus=run['corpus'],model=run['model_id'],cross_tree_constraints=nctc,
                         mandatory_under_or_alt=len(mandatory),either=bool(nctc or mandatory),satisfiable=sat))
        if not sat:
            no_ctc=_solve([cl for cl,label in zip(enc.clauses,enc.clause_labels) if not label.startswith('Constraint rule ')])
            no_mand=_solve([cl for cl,label in zip(enc.clauses,enc.clause_labels) if label not in labels])
            both=_solve([cl for cl,label in zip(enc.clauses,enc.clause_labels) if label not in labels and not label.startswith('Constraint rule ')])
            check(both,'Unexpected UNSAT after both removals: '+rid)
            diagnosis=('either individual change restores SAT' if no_ctc and no_mand else
                       'cross-tree removal only restores SAT' if no_ctc else
                       'mandatory clearing only restores SAT' if no_mand else 'both changes required')
            diagnoses.append(dict(run_id=rid,corpus=run['corpus'],arm=run['arm'],N=run['N'],grounding=run['grounding'],seed=run['seed'],
                                  n_constraints=nctc,n_mandatory_under_or_alt=len(mandatory),sat_without_constraints=no_ctc,
                                  sat_without_group_mandatory=no_mand,sat_without_both=both,classification=diagnosis,
                                  mandatory_edges=json.dumps(mandatory)))
    summary=[]
    for label,pred in [('all_conformant',lambda r:True),('cross_tree',lambda r:r['cross_tree_constraints']>0),
                       ('group_mandatory',lambda r:r['mandatory_under_or_alt']>0),('either',lambda r:r['either']),('neither',lambda r:not r['either'])]:
        rr=[r for r in rows if pred(r)]
        summary.append(dict(subset=label,n=len(rr),satisfiable=sum(r['satisfiable'] for r in rr),
                            **ci([int(r['satisfiable']) for r in rr],label)))
    ws=[dict(check=w,completed=len(runs),evaluated=sum(r['check']==w and r['status']=='ok' for r in wchecks),
             violated=sum(r['check']==w and r['status']=='ok' and r['value'] is False for r in wchecks)) for w in ('W1','W3','W5')]
    output('task3_sat_runs',rows);output('task3_unsat_diagnoses',diagnoses);output('task3_sat_summary',summary)
    output('task3_w_checks',wchecks);output('task3_w_summary',ws)
    SECTIONS.append('## Task 3 — satisfiability and well-formedness\n\nSanity passed: 381 conformant, 369 satisfiable, 12 unsatisfiable, all GLM. All original SAT values and W1/W3/W5 values were rechecked against XML.\n\n'+table(summary)+'\n\n'+table(ws)+'\n\n'+table([{k:v for k,v in r.items() if k != 'mandatory_edges'} for r in diagnoses])+
        '\n\n`fame/evaluation/quality_sat.py:74–77` adds child ⇒ parent and, for any mandatory child, parent ⇒ child, **regardless of parent group kind**. '
        'OR also requires at least one child; ALT additionally enforces pairwise exclusion. Two mandatory children of a selected ALT parent therefore conflict. '
        'Clearing group mandatory flags here removes exactly those implication clauses; removing constraints removes all constraint-rule clauses. '
        'All interventions occur in memory; no XML is changed. These are sensitivity diagnoses under the implemented translation, not minimal unsatisfiable-core proofs. '
        'The either group is a broad potential-conflict denominator, not a claim that each such model is capable of an actual contradiction (an OR mandatory child alone need not conflict).')


def main():
    check(not OUT.exists() and not REPORT.exists(),'Refusing to overwrite prior reanalysis')
    OUT.mkdir(parents=True)
    try:
        wide_rows=sorted(read_csv(BASE/'aggregate-current-v4/wide.csv'),key=lambda r:r['run_id'])
        wide={r['run_id']:r for r in wide_rows}
        inventory=json.loads(record(BASE/'inventory-current-v1/runs.json').read_text())
        runs={r['run_id']:r for r in sorted(inventory,key=lambda r:r['run_id']) if r['completed']}
        check(len(runs)==500 and len(wide)==694,'Wrong completed/planned population')
        print('Task 4: checking denominators',flush=True)
        task4(wide_rows);flush('Task 4 complete; remaining tasks pending.')
        artifacts=json.loads(record(BASE/'inventory-current-v1/artifacts.json').read_text())
        hashes={r['resolved_path']:r['sha256'] for r in artifacts if r['snapshot']=='selected'}
        paths={rid:ROOT/run['resolved_path']/'fm_gen.xml' for rid,run in runs.items()}
        for path in paths.values():
            record(path);check(INPUTS[str(path.relative_to(ROOT))]==hashes[str(path.relative_to(ROOT))],f'XML changed since inventory: {path}')
        nodes={rid:extract_indexed_nodes(path) for rid,path in paths.items()}
        refs={c:extract_indexed_nodes(record(ROOT/f'data/ground_truth/{c}.xml')) for c in ('repair','federation')}
        info={c:_partition_and_reach_for(c) for c in refs}
        for c in refs:record(ROOT/f'data/feature_partition/{c}.csv')
        print('Task 1: unrestricted attribution sanity then restricted alignments',flush=True)
        task1(runs,wide,nodes,refs,info);flush('Tasks 4 and 1 complete; remaining tasks pending.')
        print('Task 2: partition and per-run recall sanity then summaries',flush=True)
        task2(runs,wide,info);flush('Tasks 4, 1 and 2 complete; Task 3 pending.')
        print('Task 3: conformance/SAT sanity then counterfactual diagnosis',flush=True)
        task3(runs,wide,paths)
        flush('All four tasks complete. No archived outputs or scoring code modified.')
    except Exception as exc:
        flush('**STOPPED:** '+str(exc)+'\n\nLater tasks were not run. Completed sections below remain available.')
        print('STOPPED: '+str(exc),file=sys.stderr)
        return 1
    print(REPORT)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
