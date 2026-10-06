#!/usr/bin/env python3
"""One CSV: offline evaluation of the ten Astra N1 RAG expert-extension attempts."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import evaluate_semantic as sem
from scripts import evaluate_featureide as ide
from fame.evaluation.inventory import sha256
from fame.evaluation.structural import evaluate_structure, METRICS
from fame.evaluation.provenance import evaluate_provenance, PROVENANCE_METRICS
from fame.evaluation.tau_rescore import rescore_run
from fame.evaluation.sibling_agreement import unpack_pairs, correspondence, sibling_metrics


def flatten(row, prefix, metrics):
    for name, env in metrics.items():
        key = prefix + '__' + name
        row[key] = env['value']
        row[key+'__status'] = env['status']
        row[key+'__reason'] = env.get('reason', '')


def discover(root):
    runs, cells = [], set()
    for path in sorted(root.rglob('run_config.json')):
        c = json.loads(path.read_text())
        cell = (c['corpus'], c['seed'])
        if cell in cells:
            raise ValueError(f'Duplicate corpus/seed: {cell}')
        if (c['model_id'] != 'gpt-6-astra' or c['N'] != 1 or c['grounding'] != 'rag'
                or c['metamodel_block'] is not True or c['k_doc'] != 5 or c['ordering_id'] != 'primary'):
            raise ValueError(f'Unexpected extension configuration: {path}')
        m = json.loads(path.with_name('run_meta.json').read_text())
        if m['config'] != c or m['run_id'] != path.parent.name:
            raise ValueError(f'Run metadata/configuration identity mismatch: {path}')
        cells.add(cell); runs.append((path.parent, c, m))
    if cells != {(c,s) for c in ('repair','federation') for s in range(5)}:
        raise ValueError('Expected exactly seeds 0–4 in each corpus; do not silently omit attempts')
    return runs


def add_repeat_summaries(rows):
    keys=['structural__n_features', 'structural__max_depth', 'structural__mandatory_ratio',
          'semantic__semantic_precision', 'semantic__semantic_recall_total',
          'semantic__semantic_f1_total', 'one_to_one__f1_total',
          'siblings_one_to_one__sibling_f1']
    for corpus in ('repair','federation'):
        group=[r for r in rows if r['corpus']==corpus]
        for key in keys:
            vals=[float(r[key]) for r in group if r.get(key) is not None
                  and r.get(key+'__status','ok')=='ok']
            for r in group:
                r['corpus_repeats__'+key+'__n']=len(vals)
                r['corpus_repeats__'+key+'__mean']=statistics.mean(vals) if vals else None
                r['corpus_repeats__'+key+'__sd']=statistics.stdev(vals) if len(vals)>1 else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=REPO/'results/expert-astra-n1-rag-v1')
    p.add_argument('--output',type=Path,required=True,help='New CSV file (not a directory)')
    p.add_argument('--jar',type=Path,default=REPO/'tools/featureide/lib/de.ovgu.featureide.lib.fm-v3.10.0.jar')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists; choose a new versioned CSV')
    if a.output.suffix.lower()!='.csv':p.error('--output must name a CSV file')
    runs=discover(a.results.resolve())
    contract=sem._load_contract();tau=contract['semantic']['tau_primary']
    xsd=REPO/'prompts/feature-model-schema.xsd'
    if any(c['metamodel_hash']!=sha256(xsd) for _,c,_ in runs):
        p.error('Run schema hash differs from evaluation XSD')
    if not a.jar.is_file() or sha256(a.jar)!=ide.JAR_SHA256:
        p.error('Pinned FeatureIDE JAR missing or mismatched; see docs/featureide-evaluation.md')
    sources={str(path):sha256(path) for path in
             [sem.CONTRACT_PATH,xsd,sem.RHO_PATH,*sem.GROUND_TRUTH.values(),
              *sem.ATTRIBUTION.values(),*sem.PARTITION.values(),*sem.MANIFESTS.values()]}
    implementations={str(path.relative_to(REPO)):sha256(path) for path in
                    [Path(__file__).resolve(),ide.JAVA_SOURCE,*sorted((REPO/'fame/evaluation').glob('*.py')),
                     Path(sem.__file__).resolve(),Path(ide.__file__).resolve()]}
    info={c:sem._partition_and_reach_for(c) for c in sem.GROUND_TRUTH}
    print('Loading pinned local semantic encoder (no API calls)...',flush=True)
    encoder,identity=sem._encoder_or_die(contract)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='astra-evaluation-') as tmp:
        subprocess.run(['javac','-encoding','UTF-8','-cp',str(a.jar.resolve()),'-d',tmp,str(ide.JAVA_SOURCE)],check=True)
        cp=os.pathsep.join([tmp,str(a.jar.resolve())])
        java_version=subprocess.run(['java','-version'],capture_output=True,text=True,check=True).stderr
        for corpus,ref in sem.GROUND_TRUTH.items():
            test=ide.parse_model(ref,cp)
            if test['status']!='ok' or test['parsed_ok'] is not True:
                raise RuntimeError(f'FeatureIDE reference preflight failed: {test}')
        for index,(root,c,m) in enumerate(runs,1):
            corpus=c['corpus'];ref=info[corpus];xml=root/'fm_gen.xml'
            paths=[root/'run_config.json',root/'run_meta.json',root/'context_log.jsonl',*sorted((root/'fm_iter').glob('*.xml'))]
            if xml.is_file():paths.append(xml)
            hashes={str(path):sha256(path) for path in paths if path.is_file()}
            row={**c,'run_id':m['run_id'],'population':'expert_extension_only',
                 'completed':m['completed'],'terminal_status':m['terminal_status'],
                 'total_wall_seconds':m['total_wall_seconds'],'tau':tau,
                 'input_sha256':hashes,'evaluation_source_sha256':sources,
                 'implementation_sha256':implementations,'encoder_identity':identity,
                 'featureide_jar_sha256':ide.JAR_SHA256,'java_version':java_version}
            print(f'[{index}/10] {corpus} seed={c["seed"]}: evaluating',flush=True)
            if not m['completed']:
                for prefix,keys in [('structural',METRICS),('semantic',sem.SEMANTIC_METRICS),('provenance',PROVENANCE_METRICS)]:
                    flatten(row,prefix,{k:dict(value=None,status='ineligible',reason='No completed final output') for k in keys})
                row['extension_evaluation_status']='ineligible'
                rows.append(row);continue
            if not xml.is_file():raise ValueError(f'Completed run lacks final XML: {root}')
            structural=evaluate_structure(xml,xsd,expected_root=c['root_feature'])
            featureide=ide.parse_model(xml,cp)
            structural['featureide_parse']=dict(value=featureide['parsed_ok'],status=featureide['status'],reason=featureide['reason'])
            row['featureide_diagnostics']=featureide
            dead=structural['dead_features']
            structural['dead_feature_count']=dict(value=len(dead['value']) if dead['status']=='ok' else None,
                                                  status=dead['status'],reason=dead.get('reason',''))
            flatten(row,'structural',structural)
            outcome=sem.evaluate_semantic(xml,sem.GROUND_TRUTH[corpus],encoder=encoder,
                tau_primary=tau,tau_sweep=contract['semantic']['tau_sweep'],
                attested=ref['partition'].attested,organising=ref['partition'].organising,
                reach=ref['reach'],rho=ref['rho'],attribution=sem.load_attribution(sem.ATTRIBUTION[corpus]))
            flatten(row,'semantic',outcome['metrics'])
            pairs=[dict(r,run_id=m['run_id']) for r in outcome['pairs']]
            if pairs:
                matrix,g,r=unpack_pairs(pairs)
                for policy in ('independent_max','one_to_one'):
                    score=rescore_run(pairs,tau=tau,matching_policy=policy)
                    flatten(row,policy,{k:dict(value=getattr(score,k),status='ok') for k in ('precision','recall_total','f1_total','n_matched_pairs')})
                for policy in ('independent_max','one_to_one','exact_unique'):
                    metrics=sibling_metrics(g,r,correspondence(matrix,g,r,policy,tau))
                    row.update({'siblings_'+policy+'__'+k:v for k,v in metrics.items()})
            row['matching_evaluation_status']='ok' if pairs else 'ineligible'
            provenance=evaluate_provenance(xml,root/'fm_iter',root/'context_log.jsonl',
                known_doc_ids=sem._read_manifest_docids(sem.MANIFESTS[corpus]),n_planned_steps=1)
            flatten(row,'provenance',provenance['metrics'])
            errors=[k for k,v in row.items() if k.endswith('__status') and v=='evaluator_error']
            row['extension_evaluation_status']='evaluator_error' if errors else 'evaluated'
            row['evaluator_error_metrics']=errors
            if any(not Path(path).is_file() or sha256(Path(path))!=digest for path,digest in hashes.items()):
                raise ValueError('Input changed during evaluation')
            rows.append(row)
    add_repeat_summaries(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with a.output.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
    print(f'Saved {len(rows)} run rows to {a.output}. Corpus summary columns repeat within each corpus; do not count them as extra observations.')
    return int(any(r['extension_evaluation_status']=='evaluator_error' for r in rows))


if __name__=='__main__':raise SystemExit(main())
