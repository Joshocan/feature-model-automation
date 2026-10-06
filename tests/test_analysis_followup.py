import csv
import json
from pathlib import Path

import pytest

from fame.evaluation.tau_rescore import rescore_run
from fame.evaluation.statistical_family import Comparison,run_family
from scripts.tau_rescore import _iter_run_pairs
from scripts.compare_matching import compare


def test_weight_only_and_cardinality_are_different_objectives():
    matrix=[[1,.4],[.4,0]]
    pairs=[dict(run_id='r',gen_index=i,ref_index=j,ref_name=str(j),similarity=x)
           for i,row in enumerate(matrix) for j,x in enumerate(row)]
    assert rescore_run(pairs,tau=.4,matching_policy='one_to_one').n_matched_pairs==2
    assert rescore_run(pairs,tau=.4,matching_policy='one_to_one_max_weight').n_matched_pairs==1


def test_stream_rejects_interleaved_runs(tmp_path):
    p=tmp_path/'pairs.csv'
    p.write_text('run_id,gen_index,ref_index,similarity\na,0,0,.8\nb,0,0,.8\na,1,0,.8\n')
    with pytest.raises(ValueError,match='not contiguous'):list(_iter_run_pairs(p))


def test_untestable_contrast_retains_holm_slot():
    rows={'a':[{'x':v} for v in range(5)],'b':[{'x':v} for v in range(10,15)]}
    c=Comparison('test','a','b','x')
    single=run_family(family='f',comparisons=[c],data_by_arm=rows).results[0]
    both=run_family(family='f',comparisons=[c,Comparison('missing','z','b','x')],data_by_arm=rows).results
    assert both[0].p_adjusted==pytest.approx(min(1,2*single.p_value))
    assert both[1].p_value is None


def test_declared_families_match_planned_rows():
    root=Path(__file__).resolve().parents[1]
    with (root/'results/ifs-2027/analysis/aggregate-current-v4/wide.csv').open() as f:
        rows=list(csv.DictReader(f))
    for name,count in [('A_grounding',20),('B_granularity',112),('C_ablation',50)]:
        d=json.loads((root/f'config/analysis/families/{name}.json').read_text())
        assert len(d['comparisons'])==count and d['tau_primary']==.4
        for c in d['comparisons']:
            groups=[]
            for arm in ('arm_a','arm_b'):
                ids={r['run_id'] for r in rows if all(str(r[k])==str(v) for k,v in c[arm].items())}
                assert ids, (name,c['name'],arm)
                groups.append(ids)
            assert not groups[0]&groups[1]


def test_matching_rank_reversal_and_missing_policy():
    meta=[dict(run_id=m,model_id=m,corpus='repair',N='10',grounding='rag',
               arm='guided_headline',seed='0',repetition='0',strict_admissible='True') for m in ('A','B')]
    scores=[]
    for m,vs in [('A',[.9,.2]),('B',[.7,.6])]:
        for p,v in zip(['independent_max','one_to_one'],vs):
            scores.append(dict(run_id=m,tau='.4',matching_policy=p,n_generated='10',n_reference='5',
                               precision=str(v),recall_total=str(v),f1_total=str(v)))
    paired,ranks=compare(meta,scores)
    assert len(paired)==2 and all(r['rank_changed__f1_total'] for r in ranks)
    with pytest.raises(ValueError,match='Both'):compare(meta,scores[:-1])


def test_family_cli_writes_counts_and_intervals(tmp_path,monkeypatch):
    from scripts import analyse_family as driver
    from argparse import Namespace
    rows=[dict(run_id=str(i),group=g,score=v,score__status='ok')
          for i,(g,v) in enumerate([('a',v) for v in range(5)]+[('b',v) for v in range(6,11)])]
    wide=tmp_path/'wide.csv'
    with wide.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    family=tmp_path/'family.json'
    family.write_text(json.dumps(dict(family='fixture',comparisons=[dict(name='c',
        arm_a={'group':'a'},arm_b={'group':'b'},metric='score')])))
    out=tmp_path/'results/ifs-2027/analysis/test'
    monkeypatch.setattr(driver,'REPO',tmp_path)
    monkeypatch.setattr(driver,'sha256',lambda p:'test_hash')
    monkeypatch.setattr(driver,'_cli',lambda:Namespace(wide=wide,family=family,output=out,
        no_bootstrap=False,bootstrap_resamples=100))
    assert driver.main()==0
    with (out/'family_results.csv').open() as f:r=next(csv.DictReader(f))
    assert r['planned_a']==r['n_a']=='5'
    with (out/'bootstrap_intervals.csv').open() as f:intervals=list(csv.DictReader(f))
    assert len(intervals)==2 and all(r['status']=='ok' and r['sd'] for r in intervals)
