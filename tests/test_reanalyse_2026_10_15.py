import pytest
from scripts.reanalyse_2026_10_15 import ci, check, select


def test_bootstrap_constant_and_reproducible():
    assert ci([.25]*4, 'constant') == dict(mean=.25,ci_lower=.25,ci_upper=.25)
    assert ci([.1,.2,.5,.8], 'fixed') == ci([.1,.2,.5,.8], 'fixed')


def test_empty_is_not_zero():
    assert ci([], 'empty') == dict(mean=None,ci_lower=None,ci_upper=None)


def test_sanity_raises():
    with pytest.raises(ValueError,match='stop'):
        check(False, 'stop')


def test_boolean_selector():
    rows=[dict(N='10',metamodel_block='True'),dict(N='1',metamodel_block='False')]
    assert select(rows,dict(N=10,metamodel_block=True)) == rows[:1]
