import pytest

from valpredictor.models.combinatorics import match_win_probability, p_distance, score_distribution


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


def test_the_favourite_most_likely_wins_2_0_and_the_score_is_antisymmetric():
    ab = score_distribution([0.6] * 3, 3)
    ba = score_distribution([0.4] * 3, 3)
    assert max(ab, key=ab.get) == (2, 0)
    for (x, y), p in ab.items():
        assert ba[(y, x)] == pytest.approx(p)


def test_p_distance():
    assert p_distance(score_distribution([0.5, 0.5, 0.5], 3), 3) == pytest.approx(0.5)  # 2-1 + 1-2
    assert p_distance(score_distribution([0.5] * 5, 5), 5) == pytest.approx(0.375)       # 3-2 + 2-3
    assert p_distance(score_distribution([0.7], 1), 1) is None
