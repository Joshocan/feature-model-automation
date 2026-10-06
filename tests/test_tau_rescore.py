from __future__ import annotations

import pytest

from fame.evaluation.tau_rescore import rescore_all, rescore_run


def _pair(run_id, gen, ref, sim):
    return dict(run_id=run_id, gen_name=gen, ref_name=ref, similarity=sim,
                 gen_index=hash((run_id, gen)) % 1000,
                 ref_index=hash((run_id, ref)) % 1000)


def test_rescore_run_independent_max_counts_hits() -> None:
    pairs = [
        _pair("r1", "Alpha", "Alpha", 0.90),
        _pair("r1", "Alpha", "Beta",  0.20),
        _pair("r1", "Beta",  "Alpha", 0.85),  # duplicate hit on Alpha
        _pair("r1", "Beta",  "Beta",  0.10),
        _pair("r1", "Gamma", "Alpha", 0.10),  # no hits at τ=0.5
        _pair("r1", "Gamma", "Beta",  0.10),
    ]
    score = rescore_run(pairs, tau=0.5, matching_policy="independent_max")
    # Reference nodes matched at τ=0.5: {Alpha}. Generated matched: {Alpha, Beta}.
    assert score.n_generated == 3
    assert score.n_reference == 2
    assert score.n_generated_matched == 2
    assert score.n_reference_matched == 1
    assert score.matched_reference_ids == ["Alpha"]
    assert score.precision == pytest.approx(2 / 3)
    assert score.recall_total == pytest.approx(0.5)


def test_rescore_run_one_to_one_penalises_duplicates() -> None:
    pairs = [
        _pair("r1", "Alpha", "Alpha", 0.9),
        _pair("r1", "Beta",  "Alpha", 0.8),  # both hit Alpha
        _pair("r1", "Gamma", "Alpha", 0.7),
        _pair("r1", "Alpha", "Beta",  0.1),
        _pair("r1", "Beta",  "Beta",  0.1),
        _pair("r1", "Gamma", "Beta",  0.1),
    ]
    indep = rescore_run(pairs, tau=0.5, matching_policy="independent_max")
    strict = rescore_run(pairs, tau=0.5, matching_policy="one_to_one")
    # Independent max says all three gen features "hit" (they hit Alpha).
    assert indep.precision == 1.0
    # One-to-one can pair Alpha with only one gen node → precision drops.
    assert strict.n_matched_pairs == 1
    assert strict.precision == pytest.approx(1 / 3)


def test_rescore_run_empty_pairs_returns_zero_measurement() -> None:
    score = rescore_run([], tau=0.4)
    assert score.precision == 0.0
    assert score.recall_total == 0.0
    assert score.f1_total == 0.0


def test_rescore_all_expands_over_tau_and_policy() -> None:
    pairs = [
        _pair("r1", "Alpha", "Alpha", 0.9),
        _pair("r1", "Alpha", "Beta",  0.1),
        _pair("r2", "Alpha", "Alpha", 0.35),
    ]
    scores = rescore_all(pairs, taus=(0.3, 0.5),
                          matching_policies=("independent_max",))
    assert len(scores) == 4
    runs = {s.run_id for s in scores}
    assert runs == {"r1", "r2"}
    taus = {s.tau for s in scores}
    assert taus == {0.3, 0.5}


def test_unknown_matching_policy_raises() -> None:
    with pytest.raises(ValueError, match="unknown matching_policy"):
        rescore_run([_pair("r", "a", "a", 0.5)], tau=0.4, matching_policy="bogus")


def test_rescore_run_higher_tau_shrinks_matches_monotonically() -> None:
    pairs = [
        _pair("r1", "Alpha", "Alpha", 0.35),
        _pair("r1", "Beta",  "Beta",  0.55),
        _pair("r1", "Gamma", "Gamma", 0.75),
    ]
    low = rescore_run(pairs, tau=0.3)
    mid = rescore_run(pairs, tau=0.5)
    high = rescore_run(pairs, tau=0.7)
    assert low.n_reference_matched >= mid.n_reference_matched >= high.n_reference_matched


def test_rescore_preserves_duplicate_name_occurrences() -> None:
    pairs = [
        dict(run_id="r", gen_index=0, gen_name="Same", ref_index=0,
             ref_name="Same", similarity=.9),
        dict(run_id="r", gen_index=1, gen_name="Same", ref_index=0,
             ref_name="Same", similarity=.9),
    ]
    independent = rescore_run(pairs, tau=.4)
    one_to_one = rescore_run(pairs, tau=.4, matching_policy="one_to_one")
    assert independent.n_generated == 2
    assert independent.precision == 1
    assert one_to_one.precision == .5
