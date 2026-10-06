from scripts.evaluate_expert_astra import flatten, add_repeat_summaries, discover
import pytest


def test_flatten_preserves_unavailable():
    row={}
    flatten(row,'structural',{'dead_feature_count':dict(value=None,status='not_applicable',reason='UNSAT')})
    assert row['structural__dead_feature_count'] is None
    assert row['structural__dead_feature_count__reason']=='UNSAT'


def test_repeat_sd_uses_only_eligible_values():
    rows=[dict(corpus='repair',structural__n_features=v,structural__n_features__status=s)
          for v,s in [(10,'ok'),(20,'ok'),(None,'ineligible')]]
    add_repeat_summaries(rows)
    assert rows[0]['corpus_repeats__structural__n_features__n']==2
    assert rows[0]['corpus_repeats__structural__n_features__mean']==15
    assert rows[0]['corpus_repeats__structural__n_features__sd']==pytest.approx(7.0710678)


def test_missing_attempts_rejected(tmp_path):
    with pytest.raises(ValueError,match='exactly seeds'):
        discover(tmp_path)
