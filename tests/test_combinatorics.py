import math

import pytest

from valpredictor.models.combinatorics import (
    combine_win_rates,
    expected_map_probs_from_pool,
    match_win_probability,
    score_distribution,
)


def test_bo1_reduces_to_the_map_probability():
    dist = score_distribution([0.7], best_of=1)
    assert dist[(1, 0)] == pytest.approx(0.7)
    assert dist[(0, 1)] == pytest.approx(0.3)


def test_distribution_sums_to_one():
    for probs, bo in [([0.6, 0.5, 0.4], 3), ([0.5] * 5, 5), ([0.9, 0.1, 0.5], 3)]:
        dist = score_distribution(probs, best_of=bo)
        assert sum(dist.values()) == pytest.approx(1.0)


def test_bo3_only_reaches_reachable_scores():
    dist = score_distribution([0.5, 0.5, 0.5], best_of=3)
    assert set(dist.keys()) == {(2, 0), (2, 1), (1, 2), (0, 2)}


def test_bo3_certain_sweep():
    dist = score_distribution([1.0, 1.0, 0.5], best_of=3)
    assert dist[(2, 0)] == pytest.approx(1.0)
    # the third map's probability is irrelevant since the match never reaches it
    assert (1, 2) not in dist or dist[(1, 2)] == pytest.approx(0.0)


def test_match_win_probability_matches_manual_bo3_formula():
    p = 0.6
    dist = score_distribution([p, p, p], best_of=3)
    manual = p**2 + 2 * p**2 * (1 - p)  # win 2-0, or lose map1 or map2 then win out
    assert match_win_probability([p, p, p], best_of=3) == pytest.approx(manual)
    assert match_win_probability([p, p, p], best_of=3) == pytest.approx(sum(v for k, v in dist.items() if k[0] == 2))


def test_score_distribution_requires_enough_probs():
    with pytest.raises(ValueError):
        score_distribution([0.5, 0.5], best_of=3)


def test_combine_win_rates_is_symmetric_at_half():
    assert combine_win_rates(0.5, 0.5) == pytest.approx(0.5)


def test_combine_win_rates_favors_higher_rate():
    assert combine_win_rates(0.7, 0.3) > 0.5
    assert combine_win_rates(0.3, 0.7) < 0.5
    a = combine_win_rates(0.7, 0.3)
    b = combine_win_rates(0.3, 0.7)
    assert a == pytest.approx(1 - b)


def test_expected_map_probs_from_pool_repeats_combined_rate():
    probs = expected_map_probs_from_pool(0.6, 0.5, best_of=3)
    assert len(probs) == 3
    assert all(p == pytest.approx(probs[0]) for p in probs)
    assert probs[0] == pytest.approx(combine_win_rates(0.6, 0.5))
