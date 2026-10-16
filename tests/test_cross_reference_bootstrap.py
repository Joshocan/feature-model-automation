import pytest
from scripts import bootstrap_cross_reference as b


def test_mean_gate_rejects_before_bootstrap(monkeypatch):
    cells = {k: {'a': value+.002} for k, value in b.EXPECTED.items()}
    monkeypatch.setattr(b, 'load_differences', lambda *args: cells)
    def forbidden(*args):
        pytest.fail('Bootstrap must not run after a failed sanity check')
    monkeypatch.setattr(b, 'bootstrap', forbidden)
    assert b.main([]) == 1


def test_constant_paired_difference_and_determinism():
    # Own and cross can both vary: a constant within-run delta has zero width.
    cells = {k: {'z': value, 'a': value, 'b': value} for k, value in b.EXPECTED.items()}
    b.check_means(cells)
    first = b.bootstrap(cells, 20261014)
    assert first == b.bootstrap(cells, 20261014)
    for corpus, tau, n, mean, lo, hi in first:
        assert n == 3
        assert lo == pytest.approx(mean)
        assert hi == pytest.approx(mean)


def test_sort_order_does_not_change_draws():
    cells = {k: {'z': .1, 'a': .3, 'b': -.1} for k in b.EXPECTED}
    reverse = {k: dict(reversed(list(v.items()))) for k, v in cells.items()}
    assert b.bootstrap(cells, 42) == b.bootstrap(reverse, 42)
