#!/usr/bin/env python3
"""Regenerate Table 3 baselines and complete the 2026-10-15 offline reanalysis.

No model calls, scoring edits, archived-output changes, or Task 4 rerun.
Existing Task 4 report is retained; follow-up sections are appended once.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import reanalyse_2026_10_15 as old

MODELS = ['deepseek-v4.1-flash:cloud', 'glm-5.3-flash:cloud', 'gpt-6-astra']
PUBLISHED = {'repair': [(35.1,32.7,2.4),(44.5,32.6,11.8),(38.6,31.9,6.7)],
             'federation': [(35.8,34.8,1.0),(32.9,34.8,-1.9),(35.7,33.5,2.2)]}
TERMS = re.compile(r'chance|uniform|baseline|attr|attribution_agreement|provenance', re.I)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8','replace')


def search_history(reuse=None):
    """Search each reachable text blob once across every locally available ref."""
    if reuse is not None:
        reuse=reuse.resolve()
        for filename in ('stepA_search_hits.csv','stepA_search_scope.json'):
            shutil.copy2(old.record(reuse/filename),old.OUT/filename)
        summary=json.loads((old.OUT/'stepA_search_scope.json').read_text())
        old.SECTIONS.append('## Step A — original calculation search\n\n'
            f'Reused the completed search audit: {summary["reachable_commits"]} reachable commits, '
            f'{summary["historical_text_blobs"]} historical text blobs, {summary["working_tree_text_files"]} current text files; '
            f'{len(summary["notebooks"])} notebooks and {len(summary["parquets"])} parquet files. '
            'All locally available branches/refs and ignored research result files were included. '
            'No original Chance-producing script or saved Chance table was located. '
            'The source alignment evaluator counts applicable/agreeing citations, but no baseline. '
            'Full paths, object hashes and search scope are in the accompanying stepA files. '
            'The numerical reconstruction below identifies the old weighting without claiming to have recovered its original source code.')
        return
    refs=git('for-each-ref','--format=%(refname) %(objectname)').splitlines()
    objects=git('rev-list','--objects','--all').splitlines()
    suffixes={'.py','.md','.ipynb','.json','.csv','.tsv','.tex','.txt','.r','.R','.yaml','.yml'}
    entries=[]
    for line in objects:
        oid,_,name=line.partition(' ')
        if Path(name).suffix in suffixes and not name.startswith(('api_keys/','.venv/')):
            entries.append((oid,name))
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    hits=[]
    for oid,name in entries:
        proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
        header=proc.stdout.readline().decode().split()
        data=proc.stdout.read(int(header[2]));proc.stdout.read(1)
        text=data.decode('utf-8','replace')
        matched=[(i,line) for i,line in enumerate(text.splitlines(),1) if TERMS.search(line)]
        if matched:
            # Preserve source location and short snippets, not entire historic files.
            special=[(i,line) for i,line in matched if re.search(r'chance|uniform|baseline',line,re.I)]
            hits.append(dict(scope='git_blob',path=name,object=oid,matched_lines=len(matched),
                             chance_uniform_baseline_lines=len(special),examples=json.dumps(special[:5])))
    proc.stdin.close();proc.wait()
    # Include ignored outputs and untracked research files, exclude dependencies and secrets.
    count=0; notebooks=[]; parquets=[]; saved_hits=[]
    prune={'.git','node_modules','__pycache__','api_keys','.cache','.pytest_cache','.mypy_cache','.ruff_cache'}
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in prune and not d.startswith(('.venv','venv'))]
        for name in files:
            path=Path(directory)/name; rel=str(path.relative_to(ROOT))
            if rel.startswith('results/reanalysis_2026-10-15') or name in ('reanalyse_2026_10_15.py','reanalyse_provenance_followup.py'):
                continue  # Do not mistake this investigation for an original source.
            if path.suffix=='.parquet':
                parquets.append(rel)
                import pyarrow.parquet as pq
                cols=pq.read_schema(path).names
                saved_hits.append(dict(scope='parquet_schema',path=rel,object='',matched_lines=0,
                                       chance_uniform_baseline_lines=0,examples=json.dumps(cols)))
                continue
            if path.suffix not in suffixes:continue
            count+=1
            if path.suffix=='.ipynb':notebooks.append(rel)
            # CSV fields capture whether a chance/baseline output is present;
            # scan all rows for explicit labels, without treating unrelated decimals as evidence.
            n=0;special=0;examples=[]
            with path.open(errors='replace') as f:
                for i,line in enumerate(f,1):
                    if TERMS.search(line):
                        n+=1
                        if re.search(r'chance|uniform|baseline',line,re.I):
                            special+=1
                            if len(examples)<5:examples.append((i,line.strip()[:500]))
            if n:
                saved_hits.append(dict(scope='working_tree',path=rel,object='',matched_lines=n,
                                       chance_uniform_baseline_lines=special,examples=json.dumps(examples)))
    old.output('stepA_search_hits',hits+saved_hits)
    summary=dict(refs=refs,reachable_commits=int(git('rev-list','--all','--count')),
                 historical_text_blobs=len(entries),working_tree_text_files=count,
                 notebooks=notebooks,parquets=parquets,
                 scope='All locally available refs and reachable text blobs; working tree including ignored research outputs. No remote-only/unreachable history, compressed ZIP contents, dependency directories or credentials.')
    (old.OUT/'stepA_search_scope.json').write_text(json.dumps(summary,indent=2)+'\n')
    old.SECTIONS.append('## Step A — original calculation search\n\n'
        f'Searched {summary["reachable_commits"]} reachable commits, {len(entries)} historical text blobs, '
        f'and {count} current text files, including ignored saved results. Available refs: '+', '.join(refs)+'. '
        f'Notebooks found: {len(notebooks)}; parquet files: {len(parquets)}. '
        'The complete candidate/source index is `stepA_search_hits.csv`. '
        'No original Table 3 Chance-producing script or saved Chance table was located in the initial repository search. '
        'The existing `fame/evaluation/alignment.py` computes agreement by counting each citation on an aligned attested feature; '
        'it does not compute a uniform-document baseline. The pilot analysis also counts applicable citation pairs but supplies no such baseline. '
        'The formulas below therefore distinguish a numerical reconstruction of the published wording from recovery of original source code. '
        'Search scope excludes dependency/cache/credential folders, remote-only or unreachable commits, and compressed archives; no remote refs were fetched.')


def score_mapping(generated, reference, mapping, attribution, n_docs):
    applicable=agreeing=0; expected=0.; feature_probs=[]
    range_attested={reference[j]['name'] for j in mapping.values() if attribution.get(reference[j]['name'])}
    for i,j in mapping.items():
        docs=attribution.get(reference[j]['name'],set())
        citations=generated[i]['citations']
        if not docs or not citations:continue
        prob=len(docs)/n_docs
        feature_probs.append(prob)
        applicable+=len(citations); agreeing+=sum(doc in docs for doc in citations)
        expected+=len(citations)*prob
    # Different, explicit feature-equal definitions; neither repeats a feature per citation.
    range_mean=st.fmean(len(attribution[name])/n_docs for name in sorted(range_attested)) if range_attested else None
    return dict(applicable=applicable,agreeing=agreeing,expected_agreeing=expected,
                applicable_generated_features=len(feature_probs),generated_feature_probability_sum=sum(feature_probs),
                distinct_attested_range_features=len(range_attested),range_feature_mean=range_mean,
                agreement=agreeing/applicable if applicable else None,
                citation_baseline=expected/applicable if applicable else None,
                difference=(agreeing-expected)/applicable if applicable else None)


def attribution_analysis(runs,wide,nodes,refs,info):
    audit=old.read_csv(old.BASE/'sibling-current-v1/node_alignment_audit.csv')
    summary=json.loads(old.record(old.BASE/'sibling-current-v1/summary.json').read_text())
    audit_path=old.BASE/'sibling-current-v1/node_alignment_audit.csv'
    old.check(old.INPUTS[str(audit_path.relative_to(ROOT))]==summary['outputs_sha256'][audit_path.name], 'Saved alignment audit changed')
    old.check(set(summary['run_ids'])==set(runs),'Alignment population mismatch')
    maps=defaultdict(dict)
    for r in audit:
        if r['matching_policy']!='independent_max':continue
        rid=r['run_id'];i=int(r['gen_index']);j=int(r['ref_index']);c=runs[rid]['corpus']
        old.check(float(r['tau'])==.4 and i not in maps[rid],'Duplicate or wrong-tau alignment')
        old.check(nodes[rid][i]['name']==r['gen_name'] and refs[c][j]['name']==r['ref_name'],'Alignment/XML mismatch')
        maps[rid][i]=(j,float(r['similarity']))
    attrs={c:old.load_attribution(old.record(ROOT/f'data/attribution/{c}.csv')) for c in refs}
    for c,expected in [('repair',(58,1011,54)),('federation',(36,281,23))]:
        a={k:v for k,v in attrs[c].items() if v}
        old.check((len(a),sum(map(len,a.values())),len(info[c]['manifest_ids']))==expected,'Attribution/document sizes mismatch')
        old.check(all(d <= set(info[c]['manifest_ids']) for d in a.values()),'Unknown attribution documents')
    threshold_reference_chance={}
    for s in old.read_csv(old.BASE/'tau-matching-v2/tau_sweep.csv'):
        if float(s['tau'])!=.4 or s['matching_policy']!='independent_max':continue
        rid=s['run_id']; c=runs[rid]['corpus']
        names={n for n in json.loads(s['matched_reference_ids']) if attrs[c].get(n)}
        # This is ran(E_tau): all above-threshold reference hits, NOT ran(alpha).
        threshold_reference_chance[rid]=st.fmean(len(attrs[c][n])/len(info[c]['manifest_ids']) for n in sorted(names)) if names else None
    old.check(set(threshold_reference_chance)==set(runs),'Threshold-reference population mismatch')
    def calculate(mode):
        out=[]
        for rid,run in runs.items():
            c=run['corpus'];g=nodes[rid];ref=refs[c]
            if mode=='exact_unique':
                gc=Counter(n['name'] for n in g);rc=Counter(n['name'] for n in ref)
                lookup={n['name']:j for j,n in enumerate(ref) if rc[n['name']]==1}
                mapping={i:lookup[n['name']] for i,n in enumerate(g) if gc[n['name']]==1 and n['name'] in lookup}
            else:
                threshold=.4 if mode=='independent_max_0.4' else .8
                mapping={i:j for i,(j,s) in maps[rid].items() if s>=threshold}
            out.append(dict(run_id=rid,corpus=c,model=run['model_id'],alignment=mode,
                            threshold_reference_feature_equal=threshold_reference_chance[rid] if mode=='independent_max_0.4' else None,
                            **score_mapping(g,ref,mapping,attrs[c],len(info[c]['manifest_ids']))))
        return out
    rows=calculate('independent_max_0.4')
    # Gate unrestricted counts before restricted scoring or any bootstrap.
    for c in refs:
        for model,pub in zip(MODELS,PUBLISHED[c]):
            group=[r for r in rows if r['corpus']==c and r['model']==model]
            agreement=100*sum(r['agreeing'] for r in group)/sum(r['applicable'] for r in group)
            old.check(abs(agreement-pub[0])<=.05+1e-10,'Published model agreement mismatch '+c+' '+model)
            legacy=100*sum(r['applicable']*r['threshold_reference_feature_equal'] for r in group)/sum(r['applicable'] for r in group)
            old.check(abs(legacy-pub[1])<=.05+1e-10 and abs(agreement-legacy-pub[2])<=.05+1e-10,
                      'Published Chance/Diff reconstruction mismatch '+c+' '+model)
    for c,n,agree in [('repair',39465,38.0),('federation',13473,34.6)]:
        rr=[r for r in rows if r['corpus']==c];total=sum(r['applicable'] for r in rr)
        agreement=100*sum(r['agreeing'] for r in rr)/total
        old.check(total==n and abs(agreement-agree)<=.05+1e-10,'Step B unrestricted sanity failed '+c)
        legacy=100*sum(r['applicable']*r['threshold_reference_feature_equal'] for r in rr)/total
        old.check(abs(legacy-({'repair':32.5,'federation':34.5}[c]))<=.05+1e-10,'Aggregate historical baseline mismatch '+c)
        if c=='repair':
            baseline=100*sum(r['expected_agreeing'] for r in rr)/total
            old.check(abs(baseline-28.8)<=.05+1e-10,'Step B Repair baseline sanity failed')
        for r in rr:
            old.check(r['applicable']==int(wide[r['run_id']]['semantic__n_attribution_applicable']) and
                      r['agreeing']==int(wide[r['run_id']]['semantic__n_attribution_agreeing']), 'Saved per-run citation counts mismatch')
    rows+=calculate('independent_max_0.8')+calculate('exact_unique')
    results=[]
    for mode in ('independent_max_0.4','independent_max_0.8','exact_unique'):
        for c in refs:
            for model in MODELS+['ALL']:
                rr=[r for r in rows if r['alignment']==mode and r['corpus']==c and (model=='ALL' or r['model']==model) and r['applicable']]
                total=sum(r['applicable'] for r in rr); nf=sum(r['applicable_generated_features'] for r in rr)
                agreement=100*sum(r['agreeing'] for r in rr)/total if total else None
                baseline=100*sum(r['expected_agreeing'] for r in rr)/total if total else None
                # Reconstruct the supplied methods wording: distinct attested ran(alpha)
                # averaged within run, then run means weighted by citation count.
                range_weighted=100*sum(r['applicable']*r['range_feature_mean'] for r in rr)/total if total else None
                results.append(dict(corpus=c,model=model,alignment=mode,applicable_citations=total,
                    contributing_runs=len(rr),agreement_percent=agreement,citation_baseline_percent=baseline,
                    pooled_difference_pp=agreement-baseline if total else None,
                    generated_feature_equal_baseline_percent=100*sum(r['generated_feature_probability_sum'] for r in rr)/nf if nf else None,
                    range_feature_equal_citation_weighted_percent=range_weighted,
                    threshold_reference_feature_equal_citation_weighted_percent=100*sum(r['applicable']*r['threshold_reference_feature_equal'] for r in rr)/total if total and mode=='independent_max_0.4' else None,
                    **{'run_difference_pp_'+k:v for k,v in old.ci([100*r['difference'] for r in rr],c+model+mode).items()}))
    old.output('attribution_run_scores',rows)
    old.output('table3_consistent', [r for r in results if r['alignment']=='independent_max_0.4'])
    old.output('task1_restricted', [r for r in results if r['alignment']!='independent_max_0.4'])
    comparison=[]
    for c in refs:
        for model,pub in zip(MODELS,PUBLISHED[c]):
            r=next(r for r in results if r['corpus']==c and r['model']==model and r['alignment']=='independent_max_0.4')
            comparison.append(dict(corpus=c,model=model,published_agreement=pub[0],recomputed_agreement=r['agreement_percent'],
                published_chance=pub[1],reconstructed_range_chance=r['range_feature_equal_citation_weighted_percent'],
                reconstructed_threshold_reference_chance=r['threshold_reference_feature_equal_citation_weighted_percent'],
                citation_weighted_chance=r['citation_baseline_percent'],published_diff=pub[2],
                reconstructed_threshold_reference_diff=r['agreement_percent']-r['threshold_reference_feature_equal_citation_weighted_percent'],
                chance_matches_rounding=abs(pub[1]-r['threshold_reference_feature_equal_citation_weighted_percent'])<=.05+1e-10))
    old.output('published_table3_comparison',comparison)
    reproduced=all(r['chance_matches_rounding'] for r in comparison)
    old.SECTIONS.append('## Steps A–B — Table 3 reconstruction and consistent baseline\n\n'
        'Applicable-citation counts, pooled agreements and Repair’s approximately 28.8% citation-weighted baseline pass the supplied checks. '
        'The six original Chance entries are compared below with a reconstruction of the supplied methods wording. '
        f'All six published Chance values reproduced by the **all-threshold-reference-hit** feature-equal reconstruction: **{reproduced}**. '
        'Numerical reproduction does not establish recovery of the original implementation.\n\n'+old.table(comparison)+'\n\n'
        '**Definitions:** Citation baseline = sum over applicable citations of |attr(alpha(f))|/|D| divided by applicable citation count. '
        'Generated-feature-equal baseline gives each cited, aligned generated feature occurrence one vote, instead of one vote per citation. '
        'Range-feature-equal/citation-weighted baseline first averages |attr(t)|/|D| over distinct attested t in ran(alpha) in each run '
        '(including aligned features with no citations), then weights that run mean by its applicable citation count. '
        'Repeated mappings to the same reference feature receive one vote in the within-run range mean. '
        '**The published values instead reproduce a fourth formula:** for each run, average over distinct attested reference features hit by **any** above-threshold edge (ran(E_tau), saved as matched_reference_ids), '
        'then weight those run averages by applicable-citation counts. This includes reference features never selected by the generated-side best-match alpha. '
        'It both uses a different feature population and weights within-run features equally rather than according to their applicable citation counts. '
        'These are different estimands; run-level mean difference below averages each nonempty run’s agreement minus its **citation** baseline equally. '
        'Confidence intervals resample those whole runs, not individual citations.\n\n'+old.table([r for r in results if r['alignment']=='independent_max_0.4']))
    old.SECTIONS.append('## Step C — Task 1 restricted attribution\n\n'
        'Similarity ≥0.8 retains generated-side best matches meeting that cutoff. Exact unique uses literal names occurring once in each tree, independent of similarity. '
        'Contributing runs require at least one applicable citation; the structural Table 9 counts 472/361 instead require eligible aligned children and are not citation denominators. '
        'All models and both corpus aggregates are in the CSV.\n\n'+old.table([r for r in results if r['alignment']!='independent_max_0.4' and r['model']=='ALL']))


def checked_task2(runs, wide, info):
    scores=old.read_csv(old.BASE/'tau-matching-v2/tau_sweep.csv')
    for policy,target in [('independent_max',.457),('one_to_one',.395)]:
        vals=[float(r['recall_total']) for r in scores if float(r['tau'])==.4 and r['matching_policy']==policy]
        value=st.fmean(vals)
        old.check(len(vals)==500 and abs(value-target)<=.0005,'Published pooled recall mismatch '+policy)
        old.SECTIONS.append(f'Pooled full-reference recall precheck: {policy}, n={len(vals)}, mean={value:.3f}; matches {target:.3f}.')
    old.task2(runs,wide,info)


def checked_task3(runs, wide, paths):
    for model, completed, conformant, sat in zip(MODELS,[219,247,34],[166,182,33],[166,170,33]):
        rr=[rid for rid,r in runs.items() if r['model_id']==model]
        actual=(len(rr),sum(wide[r]['structural__structural_conformance']=='True' for r in rr),
                sum(wide[r]['structural__satisfiable']=='True' for r in rr))
        old.check(actual==(completed,conformant,sat),'Published model cascade mismatch '+model+str(actual))
    old.task3(runs,wide,paths)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/reanalysis_2026-10-15-followup')
    p.add_argument('--report',type=Path,default=ROOT/'results/reanalysis_2026-10-15.md')
    p.add_argument('--reuse-search',type=Path,help='Reuse completed Step A search audit directory; hashes recorded')
    p.add_argument('--replace-followup',action='store_true',help='Replace only this script’s prior follow-up report section; preserve Task 4 verbatim')
    args=p.parse_args(); old.OUT=args.output.resolve(); report=args.report.resolve()
    old.check(not old.OUT.exists(),'Choose a new output directory')
    old.check(old.OUT.is_relative_to(ROOT/'results') and report.is_relative_to(ROOT/'results'),'Outputs must be under results')
    old.OUT.mkdir(parents=True)
    prior=report.read_text() if report.exists() else '# Reanalysis — 2026-10-15\n'
    if args.replace_followup:
        prior=prior.split('\n\n## Follow-up — consistent provenance baseline',1)[0]
    old.check('## Follow-up — consistent provenance baseline' not in prior,'Follow-up already appended; choose a fresh report')
    print('Step A: searching local Git history and saved files',flush=True)
    search_history(args.reuse_search)
    wide={r['run_id']:r for r in old.read_csv(old.BASE/'aggregate-current-v4/wide.csv')}
    inv=json.loads(old.record(old.BASE/'inventory-current-v1/runs.json').read_text())
    runs={r['run_id']:r for r in sorted(inv,key=lambda r:r['run_id']) if r['completed']}
    old.check(len(runs)==500,'Expected 500 completed outputs')
    artifacts=json.loads(old.record(old.BASE/'inventory-current-v1/artifacts.json').read_text())
    hashes={r['resolved_path']:r['sha256'] for r in artifacts if r['snapshot']=='selected'}
    paths={rid:ROOT/r['resolved_path']/'fm_gen.xml' for rid,r in runs.items()}
    for path in paths.values():
        old.record(path);rel=str(path.relative_to(ROOT))
        old.check(old.INPUTS[rel]==hashes[rel],'XML differs from inventory '+rel)
    nodes={rid:old.extract_indexed_nodes(path) for rid,path in paths.items()}
    refs={c:old.extract_indexed_nodes(old.record(ROOT/f'data/ground_truth/{c}.xml')) for c in ('repair','federation')}
    info={c:old._partition_and_reach_for(c) for c in refs}
    failures=[]
    for name,func in [('Steps B/C attribution',lambda:attribution_analysis(runs,wide,nodes,refs,info)),
                      ('Task 2',lambda:checked_task2(runs,wide,info)),('Task 3',lambda:checked_task3(runs,wide,paths))]:
        print(name,flush=True)
        try:func()
        except (ValueError,KeyError,FileNotFoundError) as exc:
            failures.append(name+': '+str(exc));old.SECTIONS.append('## '+name+' stopped\n\n'+str(exc))
            print(failures[-1],flush=True)
    for c in refs:
        old.record(ROOT/f'data/feature_partition/{c}.csv')
        old.record(ROOT/('data/raw/repair/manifest_repair.csv' if c=='repair' else 'data/raw/federation/manifest_fed.csv'))
    for path in [Path(__file__),ROOT/'scripts/reanalyse_2026_10_15.py',ROOT/'fame/evaluation/quality_sat.py']:
        old.record(path)
    if not failures:
        old.SECTIONS.append('## Claims affected and recommended reporting\n\n'
            '1. **Correct the provenance definition and Table 3 Chance/Diff columns together.** The old values reproduce a reference-hit-feature-equal baseline using ran(E_tau), not ran(alpha). '
            'For the requested citation-weighted definition, Repair is 38.036% agreement versus 28.771% baseline (+9.265 pp); Federation is 34.617% versus 36.773% (−2.156 pp). '
            'The pooled differences are not the equal-run mean differences: Repair’s run mean is +15.454 pp [14.292, 16.623]; Federation’s is +0.036 pp [−0.901, 1.041]. '
            'Do not attach the latter intervals to the pooled estimates.\n\n'
            '2. **High-confidence attribution changes the picture.** At similarity ≥0.8, run-mean agreement-minus-baseline is positive with intervals excluding zero in both corpora. '
            'Under exact-unique names, Repair remains positive, while Federation’s interval includes zero. These restricted populations differ and do not establish textual entailment.\n\n'
            '3. **Attested-reference recall exceeds organising-reference recall under both matching policies in both corpora.** '
            'Full-reference recall remains consistent with the archived means, 0.457 independent-max and 0.395 one-to-one. Organising-feature matches do not by themselves prove hallucination.\n\n'
            '4. **Qualify the SAT result by its opportunity for contradiction and the implemented translation.** '
            'Of 381 conformant outputs, 227 have neither cross-tree constraints nor mandatory children under OR/ALT groups and all are satisfiable; '
            '142/154 in the broader potential-conflict subset are satisfiable. All 12 UNSAT models remain UNSAT without cross-tree constraints and become SAT when group-mandatory implications are removed. '
            'This implicates the tree/mandatory translation, not cross-tree constraints as a necessary cause. It is not evidence that FeatureIDE applies identical semantics.\n\n'
            '5. **Do not report zero W3/W5 violations:** one completed output violates both; W1 has zero. These are counts among all 500 completed outputs, not just the conformant subset.')
    (old.OUT/'input_hashes.json').write_text(json.dumps(old.INPUTS,indent=2)+'\n')
    (old.OUT/'status.json').write_text(json.dumps(dict(failures=failures,seed=old.SEED,resamples=5000,numpy=old.np.__version__),indent=2)+'\n')
    report.write_text(prior.rstrip()+'\n\n## Follow-up — consistent provenance baseline\n\n'
        'This section supersedes the earlier Task 1 stop. Task 4 was neither rerun nor changed. '
        'Bootstrap: 5,000 whole-run resamples; fixed seed 20261015, deterministic per-statistic streams; 95% percentile intervals. '
        'Run-mean intervals are conditional on applicable observations, pooled across configurations, not stratified by arm.\n\n'
        +'\n\n'.join(old.SECTIONS)+'\n')
    print(report,flush=True)
    return bool(failures)


if __name__=='__main__':
    raise SystemExit(main())
