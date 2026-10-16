import pytest
from scripts.reanalyse_provenance_followup import score_mapping


def test_distinguishes_citation_generated_feature_and_range_weighting():
    generated=[{'citations':['d1','d2','d3']},{'citations':['d1']},{'citations':[]}]
    reference=[{'name':'A'},{'name':'B'},{'name':'C'}]
    attrs={'A':{'d1'},'B':{'d1','d2','d3'},'C':{'d1','d2'}}
    result=score_mapping(generated,reference,{0:0,1:1,2:2},attrs,4)
    assert result['applicable']==4
    assert result['agreeing']==2
    assert result['citation_baseline']==pytest.approx(.375)
    assert result['difference']==pytest.approx(.125)
    assert result['generated_feature_probability_sum']/result['applicable_generated_features']==pytest.approx(.5)
    assert result['range_feature_mean']==pytest.approx(.5)


def test_range_counts_reference_once_and_includes_uncited_alignment():
    generated=[{'citations':['d1']},{'citations':['d2']},{'citations':[]}]
    reference=[{'name':'A'},{'name':'B'}]
    result=score_mapping(generated,reference,{0:0,1:0,2:1},{'A':{'d1'},'B':{'d1','d2','d3'}},4)
    assert result['distinct_attested_range_features']==2
    assert result['range_feature_mean']==.5
    assert result['citation_baseline']==.25


def test_unannotated_or_unaligned_citations_not_applicable():
    result=score_mapping([{'citations':['d1']},{'citations':['d1']}],[{'name':'A'}],{0:0},{},4)
    assert result['applicable']==0
    assert result['difference'] is None
    assert result['citation_baseline'] is None
