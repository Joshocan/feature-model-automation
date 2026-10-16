import json

import numpy as np
import pytest

from scripts import evaluate_cross_reference as cross


def run(rid='a', model='model', corpus='repair', completed=True):
    return dict(run_id=rid, model_id=model, corpus=corpus, completed=completed,
                arm='guided_headline', N=10, grounding='rag', ordering_id='primary',
                k_doc=5, metamodel_block=True, seed=0, repetition=0)


def fixture_scores(r=None):
    r = r or run()
    return cross.score_conditions(r, ['duplicate', 'duplicate'],
        {'repair': ['A'], 'federation': ['B', 'C']},
        {'repair': np.array([[.9], [.8]]), 'federation': np.array([[.7, .2], [.6, .1]])}, [.4, .8])


def test_existing_policies_and_denominators():
    rows = fixture_scores()
    ind, one = rows[:2]
    assert ind['precision_own'] == 1
    assert ind['recall_total_cross'] == .5
    assert one['precision_own'] == .5
    assert one['recall_total_own'] == 1
    assert one['n_generated'] == 2  # duplicate names are distinct occurrences
    assert rows[2]['precision_cross'] == 0
    assert rows[2]['precision_own_minus_cross'] == 1


def test_opposite_reference_direction():
    row = fixture_scores(run(corpus='federation'))[0]
    assert row['own_reference_corpus'] == 'federation'
    assert row['n_reference_own'] == 2
    assert row['cross_reference_corpus'] == 'repair'
    assert row['n_reference_cross'] == 1


def test_nonrag_headline_has_no_retrieval_depth():
    assert cross.headline({**run(), 'grounding': 'nonrag', 'k_doc': None})
    assert not cross.headline({**run(), 'k_doc': 10})
    assert not cross.headline({**run(), 'arm': 'ablation'})


def test_zero_match_is_zero_but_empty_is_not():
    with pytest.raises(ValueError, match='Empty'):
        cross.score_conditions(run(), [], {'repair': ['A'], 'federation': ['B']}, {}, [.4])
    with pytest.raises(ValueError, match='Invalid cosine'):
        cross.pairs_from_matrix('a', ['A'], ['B'], np.array([[np.nan]]))


def test_headline_ties_empty_cells_and_planned_counts():
    a, b, missing = run(), run('b', 'other'), run('c', 'empty', completed=False)
    baseline = {**run('d'), 'N': 1}
    summary = cross.summarise([a, b, missing, baseline],
                              fixture_scores(a)+fixture_scores(b), [.4], headline_only=True)
    assert len(summary) == 6
    for row in summary:
        if row['model_id'] == 'empty':
            assert row['rank_f1_cross'] is None
            assert row['mean_f1_total_own'] is None
            assert row['planned'] == 1 and row['completed'] == 0
        else:
            assert row['rank_f1_own'] == 1.5
            assert row['planned'] == 1


def test_inventory_cli_without_downloads(tmp_path, monkeypatch):
    import csv
    monkeypatch.setattr(cross, 'REPO', tmp_path)
    contract = tmp_path / 'contract.json'
    contract.write_text('{}')
    monkeypatch.setattr(cross, 'CONTRACT_PATH', contract)
    xml = '<featureModel><struct><and name="Root"><feature name="A"/></and></struct></featureModel>'
    refs = {}
    for corpus in ('repair', 'federation'):
        refs[corpus] = tmp_path / f'{corpus}.xml'
        refs[corpus].write_text(xml)
    monkeypatch.setattr(cross, 'GROUND_TRUTH', refs)
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.] if s == 'Root' else [0., 1.] for s in texts])
    monkeypatch.setattr(cross, '_encoder_or_die', lambda contract: (Encoder(), {'status': 'ok'}))
    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'fm_gen.xml').write_text(xml)
    inv = tmp_path / 'inventory'
    inv.mkdir()
    r = {**run(), 'resolved_path': 'run'}
    (inv / 'runs.json').write_text(json.dumps([r]))
    (inv / 'summary.json').write_text(json.dumps(dict(planned=1, issues=[], contract_sha256=cross.sha256(contract))))
    (inv / 'artifacts.json').write_text(json.dumps([dict(population='main', snapshot='selected',
        resolved_path='run/fm_gen.xml', sha256=cross.sha256(folder / 'fm_gen.xml'))]))
    out = tmp_path / 'results/ifs-2027/analysis/check'
    argv = ['--inventory', str(inv), '--output', str(out)]
    assert cross.main(argv) == 0
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['score_rows'] == 12 and not summary['partial']
    rows = list(csv.DictReader((out / 'scores.csv').open()))
    assert all(float(r['f1_total_own_minus_cross']) == 0 for r in rows)
    with pytest.raises(SystemExit):
        cross.main(argv)  # Never overwrite evidence
    (folder / 'fm_gen.xml').write_text(xml + '\n')
    with pytest.raises(ValueError, match='changed final XML'):
        cross.main(['--inventory', str(inv), '--output', str(out.parent / 'changed')])
