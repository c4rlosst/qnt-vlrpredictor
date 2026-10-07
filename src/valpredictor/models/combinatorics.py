"""Match-winner probability and exact-score distribution from a per-map win
probability, respecting Bo1/Bo3/Bo5 early termination (a Bo3 ends 2-0 without
a third map being played).

Maps are treated as independent. A model of correlated maps (one strength
shock per series) fit only marginally better on real series (p ~ 0.05) and
independence implies a third map in 48.6% of Bo3s against 46.7% +/- 2.3%
observed, so the simpler version stays.
"""

from __future__ import annotations


def score_distribution(map_probs: list[float], best_of: int) -> dict[tuple[int, int], float]:
    """`map_probs[i]` = P(team1 wins map i+1), for as many maps as could
    possibly be played (len(map_probs) must be >= best_of). Returns
    {(team1_maps_won, team2_maps_won): probability}, summing to 1.0.
    """
    needed = best_of // 2 + 1
    if len(map_probs) < best_of:
        raise ValueError(f"need at least {best_of} map probabilities for a Bo{best_of}, got {len(map_probs)}")

    results: dict[tuple[int, int], float] = {}

    def recurse(idx: int, a_wins: int, b_wins: int, prob_so_far: float) -> None:
        if a_wins == needed or b_wins == needed:
            key = (a_wins, b_wins)
            results[key] = results.get(key, 0.0) + prob_so_far
            return
        p = map_probs[idx]
        recurse(idx + 1, a_wins + 1, b_wins, prob_so_far * p)
        recurse(idx + 1, a_wins, b_wins + 1, prob_so_far * (1 - p))

    recurse(0, 0, 0, 1.0)
    return results


def match_win_probability(map_probs: list[float], best_of: int) -> float:
    needed = best_of // 2 + 1
    return sum(p for (a, _), p in score_distribution(map_probs, best_of).items() if a == needed)


def p_distance(dist: dict[tuple[int, int], float], best_of: int) -> float | None:
    """P(the series goes to its last possible map): Bo3 -> a 2-1, Bo5 -> a 3-2. None for a Bo1."""
    if best_of < 3:
        return None
    return sum(p for (a, b), p in dist.items() if a + b == best_of)
