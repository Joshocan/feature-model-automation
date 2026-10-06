import json

import pytest

from scripts import evaluate_semantic as driver
from fame.evaluation.reachable import FeaturePartition, validate_reach_inputs


@pytest.mark.parametrize('corpus,docs,reach,total,closed', [('repair',54,58,100,84), ('federation',23,36,130,59)])
def test_real_corpus_reach(corpus, docs, reach, total, closed):
    info = driver._partition_and_reach_for(corpus)
    assert len(info['manifest_ids']) == docs
    assert len(info['direct']) == reach
    assert len(info['reach']) == closed
    assert info['partition'].attested <= info['reach'] <= info['partition'].full
    assert info['rho'] == len(info['reach']) / total
    from fame.evaluation.coverage import extract_nodes
    parents = dict(extract_nodes(driver.GROUND_TRUTH[corpus]))
    assert all(parents[f] is None or parents[f] in info['reach'] for f in info['reach'])


def test_semantic_manifest_rejects_bad_delimiter(tmp_path):
    p = tmp_path / 'manifest.csv'
    p.write_text('doc_id,title\nrep_01,A\n')
    with pytest.raises(ValueError, match='doc_id'):
        driver._read_manifest_docids(p)


def test_calibration_mismatch_rejected(tmp_path, monkeypatch):
    p = tmp_path / 'rho.json'
    p.write_text(json.dumps({'repair': {'rho': 0.1}}))
    monkeypatch.setattr(driver, 'RHO_PATH', p)
    with pytest.raises(ValueError, match='recorded rho'):
        driver._partition_and_reach_for('repair')


@pytest.mark.parametrize('att,org,attribution,docs,names,reason', [
    ({'A'},{'A'},{'A':{'d'}},['d'],['A'],'overlap'),
    ({'A'},set(),{'A':{'d'}},['d'],['A','B'],'cover exactly'),
    ({'A'},set(),{},['d'],['A'],'Attributed features'),
    ({'A'},set(),{'A':{'wrong'}},['d'],['A'],'unknown corpus'),
])
def test_inconsistent_inputs_rejected(att,org,attribution,docs,names,reason):
    with pytest.raises(ValueError, match=reason):
        validate_reach_inputs(FeaturePartition(frozenset(att),frozenset(org)),
                              attribution,docs,names)
