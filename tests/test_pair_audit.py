import csv
import json

import pytest

from scripts import prepare_pair_audit as prep
from scripts import analyse_pair_audit as analysis


def pair(g, r, sim, rid='run'):
    return dict(run_id=rid, corpus='repair', gen_index=g, gen_name=f'G{g}',
                ref_index=r, ref_name=f'T{r}', similarity=sim)


def test_best_match_tie_and_boundaries():
    rows = [pair(0, 1, .8), pair(0, 0, .8), pair(1, 0, .39), pair(1, 1, .2)]
    chosen = prep.best_pairs(rows)
    assert len(chosen) == 1 and chosen[0]['ref_index'] == 0
    assert [prep.band_for(x) for x in (.4,.5,.6,.7,.8,1)] == [*prep.BANDS, prep.BANDS[-1]]
    with pytest.raises(ValueError, match='Duplicate'):
        prep.best_pairs(rows + [rows[0]])
    with pytest.raises(ValueError, match='Incomplete'):
        prep.best_pairs(rows[:-1])


def test_sampling_reproducible_without_padding():
    rows = [pair(g,0,.45) for g in range(30)] + [pair(40,0,.85)]
    a, counts = prep.sample_pairs(rows,20,42)
    b, _ = prep.sample_pairs(list(reversed(rows)),20,42)
    assert a == b
    assert len(a) == 21
    assert len({(r['run_id'],r['gen_index']) for r in a}) == 21
    assert counts[-1]['shortage'] == 19


def test_kappa_and_missingness():
    a = dict(a='same_concept',b='same_concept',c='related',d='unrelated',e='cannot_judge')
    b = dict(a='same_concept',b='related',c='related',d='unrelated',e='unrelated')
    result = analysis.agreement(a,b)
    assert result['n'] == 4
    assert result['raw_agreement'] == .75
    assert result['kappa'] == pytest.approx((.75-.3125)/(1-.3125))
    assert analysis.agreement({'a':'same_concept'},{'a':'same_concept'})['kappa'] is None
    assert analysis.agreement({'a':'cannot_judge'},{'a':'related'})['n'] == 0


def test_weighted_shares_not_naive_band_average():
    key={'a':{'band':'low'},'b':{'band':'high'}}
    ratings={'a':'unrelated','b':'same_concept'}
    _, weighted=analysis.analyse(key,[dict(band='low',population=90),dict(band='high',population=10)],ratings,ratings)
    assert weighted[0]['estimated_same_share_all'] == .1


def test_rating_validation():
    row=dict(item_id='P001',corpus='repair',generated_feature='G',reference_feature='T',label='same_concept')
    key={'P001':row}
    assert analysis.validate_ratings([row],key) == {'P001':'same_concept'}
    for altered in ({**row,'label':''},{**row,'reference_feature':'changed'}):
        with pytest.raises(ValueError):
            analysis.validate_ratings([altered],key)
    with pytest.raises(ValueError):
        analysis.validate_ratings([row,row],key)
    with pytest.raises(ValueError):
        analysis.validate_ratings([],key)


def test_end_to_end_synthetic(tmp_path):
    inv=tmp_path/'inventory';inv.mkdir()
    runs=inv/'runs.json'
    runs.write_text(json.dumps([dict(run_id='run',corpus='repair',model_id='SECRET_MODEL',completed=True)]))
    pairs=tmp_path/'pairs.csv'
    rows=[pair(i,0,s) for i,s in enumerate((.45,.55,.65,.75,.85))]
    prep.write_csv(pairs,rows,list(rows[0]))
    (tmp_path/'summary.json').write_text(json.dumps(dict(inventory_sha256=prep.sha256(runs),semantic_scored=1)))
    out=tmp_path/'audit'
    argv=['--pairs',str(pairs),'--inventory',str(inv),'--output',str(out)]
    assert prep.main(argv)==0
    blank=prep.read_csv(out/'ratings_R01.csv')
    assert len(blank)==5
    assert set(blank[0])==set(prep.PUBLIC)|{'label','comment'}
    assert 'SECRET_MODEL' not in (out/'ratings_R01.csv').read_text()
    for rater in ('R01','R02'):
        rated=[{**row,'label':'same_concept'} for row in blank]
        prep.write_csv(tmp_path/f'{rater}.csv',rated,list(rated[0]))
    args=['--audit-dir',str(out),'--rater1',str(tmp_path/'R01.csv'),
          '--rater2',str(tmp_path/'R02.csv'),'--output',str(tmp_path/'analysis')]
    assert analysis.main(args)==0
    result=json.loads((tmp_path/'analysis/summary.json').read_text())
    assert result['items']==5 and result['agreement']['raw_agreement']==1
    with pytest.raises(SystemExit):
        prep.main(argv)
    with pytest.raises(SystemExit):
        analysis.main(args)
