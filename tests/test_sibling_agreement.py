import numpy as np
import pytest
from fame.evaluation.sibling_agreement import sibling_metrics, correspondence, unpack_pairs


def tree(parents, names=None):
    return [dict(name=(names or [str(i) for i in range(len(parents))])[i], parent_index=p)
            for i, p in enumerate(parents)]


def test_renamed_group_preserves_siblings_without_matching_parent():
    g = tree([None, 0, 1, 1])
    r = tree([None, 0, 1, 1])
    s = sibling_metrics(g, r, {2:2, 3:3})
    assert s['sibling_precision'] == s['sibling_recall'] == s['sibling_f1'] == 1
    assert s['n_parent_evaluable'] == 0


def test_flat_tree_has_perfect_recall_but_not_precision():
    g = tree([None, 0, 0, 0, 0])
    r = tree([None, 0, 0, 1, 1, 2, 2])
    s = sibling_metrics(g, r, {1:3, 2:4, 3:5, 4:6})
    assert s['sibling_recall'] == 1
    assert s['sibling_precision'] == pytest.approx(1/3)
    assert s['sibling_f1'] == .5
    assert (s['sibling_tp'], s['sibling_fp'], s['sibling_fn']) == (2, 4, 0)


def test_split_siblings_have_zero_recall_and_undefined_precision():
    g = tree([None, 0, 0, 1, 2])
    r = tree([None, 0, 1, 1])
    s = sibling_metrics(g, r, {3:2, 4:3})
    assert s['sibling_recall'] == 0 and s['sibling_precision'] is None
    assert s['sibling_f1'] == 0


def test_same_reference_hits_and_roots_excluded():
    s = sibling_metrics(tree([None, 0, 0]), tree([None, 0]), {0:0, 1:1, 2:1})
    assert s['n_eligible_pairs'] == 0
    assert s['n_same_reference_pairs_excluded'] == 1
    assert s['sibling_f1'] is None


def test_mapping_policies_and_exact_duplicate_exclusion():
    g = tree([None, 0, 0], ['R', 'A', 'A'])
    r = tree([None, 0, 0], ['R', 'A', 'B'])
    m = np.array([[1,0,0],[0,.9,.8],[0,.95,.7]])
    assert correspondence(m,g,r,'exact_unique',.4) == {0:0}
    assert correspondence(m,g,r,'independent_max',.4) == {0:0,1:1,2:1}
    one = correspondence(m,g,r,'one_to_one',.4)
    assert len(one) == len(set(one.values())) == 3


def test_matrix_metadata_validation():
    rows = [dict(gen_index=0,gen_name='G',gen_parent_index='',ref_index=0,
                 ref_name='R',ref_parent_index='',similarity=1)]
    assert unpack_pairs(rows)[0].shape == (1,1)
    with pytest.raises(ValueError, match='Duplicate'):
        unpack_pairs(rows*2)


def test_counts_against_brute_force():
    from itertools import combinations
    g = tree([None,0,0,1,1,2,2,2])
    r = tree([None,0,0,1,1,2,2])
    mapping = {1:1,2:2,3:3,4:3,5:4,6:5,7:6}
    tp=fp=fn=0
    for (i,j),(k,l) in combinations(mapping.items(),2):
        if j==l: continue
        predicted=g[i]['parent_index']==g[k]['parent_index']
        expected=r[j]['parent_index']==r[l]['parent_index']
        tp+=predicted and expected
        fp+=predicted and not expected
        fn+=expected and not predicted
    s=sibling_metrics(g,r,mapping)
    assert (s['sibling_tp'],s['sibling_fp'],s['sibling_fn']) == (tp,fp,fn)
