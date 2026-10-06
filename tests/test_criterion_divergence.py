from __future__ import annotations

import math

import pytest


def test_missing_pairs_and_ties_match_scipy():
    from scipy.stats import spearmanr, kendalltau
    a = [1, 1, 3, 4, None, 100, float('nan')]
    b = [2, 3, 3, 1, 0, None, 100]
    assert spearman_rho(a, b) == pytest.approx(spearmanr(a[:4], b[:4]).statistic)
    assert kendall_tau(a, b) == pytest.approx(kendalltau(a[:4], b[:4]).statistic)
    result = pairwise_correlations([dict(a=x, b=y) for x, y in zip(a, b)],
                                   {'a': 'a', 'b': 'b'})[0]
    assert (result.n, result.n_excluded, result.status) == (4, 3, 'ok')


@pytest.mark.parametrize('values,status', [([], 'insufficient_n'),
                                        ([1], 'insufficient_n'),
                                        ([1, 1], 'constant_input')])
def test_nonestimable_status(values, status):
    result = pairwise_correlations([dict(a=x, b=x) for x in values],
                                   {'a': 'a', 'b': 'b'})[0]
    assert result.status == status
    assert result.spearman_rho is None and result.kendall_tau is None


def test_unequal_lengths_rejected():
    for method in (spearman_rho, kendall_tau):
        with pytest.raises(ValueError, match='equal lengths'):
            method([1, 2], [1])

from fame.evaluation.criterion_divergence import (
    build_rankings,
    kendall_tau,
    pairwise_correlations,
    spearman_rho,
)


# ─────────────────────────────────────────────────────────────────────────────
# rank correlations
# ─────────────────────────────────────────────────────────────────────────────

def test_spearman_rho_of_perfect_agreement_is_one() -> None:
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_spearman_rho_of_perfect_disagreement_is_negative_one() -> None:
    assert spearman_rho([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_rho_ignores_scale_and_offset() -> None:
    rho_raw = spearman_rho([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    rho_shifted = spearman_rho([100.0, 200.0, 300.0], [-1.0, 0.0, 1.0])
    assert rho_raw == pytest.approx(rho_shifted)


def test_kendall_tau_matches_manual_count() -> None:
    # 3 items: (1,1), (2,3), (3,2). Pairs: (1,2) concordant, (1,3) concordant,
    # (2,3) discordant. τ = (2 - 1) / 3 = 1/3.
    assert kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_kendall_tau_drops_none_pairs() -> None:
    # Only 2 valid pairs → τ = (1-0)/1 = 1.
    assert kendall_tau([1, 2, None, 3], [1, 2, 5, 3]) == pytest.approx(1.0)


def test_correlations_return_none_when_insufficient_data() -> None:
    assert spearman_rho([1.0], [2.0]) is None
    assert kendall_tau([1.0], [2.0]) is None


def test_spearman_handles_ties_via_average_rank() -> None:
    # Ties in both sides → all ranks tie → correlation degenerates but must
    # not crash. Should return None (zero denominator).
    assert spearman_rho([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is None


# ─────────────────────────────────────────────────────────────────────────────
# build_rankings
# ─────────────────────────────────────────────────────────────────────────────

def test_build_rankings_orders_descending_and_ranks_ties(sample_cells=None) -> None:
    cell_rows = [
        {"corpus": "repair", "model_id": "ds", "structural": 0.9, "semantic": 0.4},
        {"corpus": "repair", "model_id": "glm", "structural": 0.7, "semantic": 0.7},
        {"corpus": "fed",    "model_id": "ds", "structural": 0.7, "semantic": 0.3},
    ]
    rankings = build_rankings(
        cell_rows, cell_id_keys=("corpus", "model_id"),
        criteria={"structural": "structural", "semantic": "semantic"},
    )
    # Structural: 0.9 (repair/ds) > 0.7 (tied). Semantic: 0.7 > 0.4 > 0.3.
    structural = next(r for r in rankings if r.criterion == "structural")
    assert structural.cell_ids[0] == ("repair", "ds")
    assert structural.scores[0] == 0.9
    # The two 0.7s share rank 2.5.
    tied = [r for r in structural.ranks if r > 1]
    assert tied.count(2.5) == 2

    semantic = next(r for r in rankings if r.criterion == "semantic")
    assert [c[1] for c in semantic.cell_ids] == ["glm", "ds", "ds"]


def test_build_rankings_puts_none_scores_at_tail() -> None:
    cell_rows = [
        {"corpus": "a", "structural": 0.5},
        {"corpus": "b", "structural": None},
        {"corpus": "c", "structural": 0.9},
    ]
    rankings = build_rankings(cell_rows, cell_id_keys=("corpus",),
                               criteria={"structural": "structural"})
    ranking = rankings[0]
    assert ranking.cell_ids[0] == ("c",)
    assert ranking.cell_ids[-1] == ("b",)
    assert ranking.scores[-1] is None


# ─────────────────────────────────────────────────────────────────────────────
# pairwise_correlations
# ─────────────────────────────────────────────────────────────────────────────

def test_pairwise_correlations_detect_agreement() -> None:
    cell_rows = [
        {"structural": 0.1, "semantic": 0.2},
        {"structural": 0.5, "semantic": 0.6},
        {"structural": 0.9, "semantic": 0.95},
    ]
    corrs = pairwise_correlations(cell_rows, {"structural": "structural",
                                                "semantic":   "semantic"})
    assert len(corrs) == 1
    assert corrs[0].spearman_rho == pytest.approx(1.0)
    assert corrs[0].kendall_tau == pytest.approx(1.0)
    assert corrs[0].n == 3


def test_pairwise_correlations_detect_divergence() -> None:
    # Structural strong, semantic reversed — the paper's load-bearing pattern.
    cell_rows = [
        {"structural": 0.1, "semantic": 0.9},
        {"structural": 0.5, "semantic": 0.5},
        {"structural": 0.9, "semantic": 0.1},
    ]
    corrs = pairwise_correlations(cell_rows, {"structural": "structural",
                                                "semantic":   "semantic"})
    assert corrs[0].spearman_rho == pytest.approx(-1.0)
    assert corrs[0].kendall_tau == pytest.approx(-1.0)
