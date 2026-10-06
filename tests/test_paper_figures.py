import pytest
from scripts import plot_paper_figures as p


def test_duplicate_join_keys_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        p.unique([{'run_id':'x'},{'run_id':'x'}],lambda r:r['run_id'])


def test_guided_excludes_robustness_and_ablation():
    assert p.guided(dict(arm='guided_curve',metamodel_block='True'))
    assert p.guided(dict(arm='astra_cross_corpus',metamodel_block='True'))
    assert not p.guided(dict(arm='order_sensitivity',metamodel_block='True'))
    assert not p.guided(dict(arm='ablation',metamodel_block='False'))


def test_missing_score_is_gap_not_zero():
    assert p.score({},'x','one_to_one') is None
    assert p.score({('x',.4,'one_to_one'):{'f1_total':'0'}},'x','one_to_one')==0


def test_unknown_outcome_is_not_silently_dropped(tmp_path):
    wide={'x':dict(run_id='x',arm='guided_headline',metamodel_block='True',
                   model_id=p.MODELS[0],grounding='rag')}
    with pytest.raises(ValueError,match='Unrepresented'):
        p.outcomes(wide,{'x':{'recorded_terminal_status':'provider_error'}},tmp_path)
