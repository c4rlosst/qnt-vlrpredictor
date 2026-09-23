"""Derives match-winner probability and map-score distribution from a list
of per-map P(team1 wins) probabilities, respecting Bo1/Bo3/Bo5 early
termination (e.g. a Bo3 ends 2-0 without a third map being played).

Independence assumption: maps are treated as conditionally independent given
their own features (the per-map model already encodes map-specific team
strength; it does not model within-match momentum). This is a documented
simplification — see README.
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
    dist = score_distribution(map_probs, best_of)
    needed = best_of // 2 + 1
    return sum(p for (a, b), p in dist.items() if a == needed)


def expected_map_probs_from_pool(
    team1_map_pool_rate: float, team2_map_pool_rate: float, best_of: int
) -> list[float]:
    """Pre-veto fallback: when we don't know which specific maps will be
    played, approximate every remaining map with each team's map-pool-averaged
    win rate turned into a single P(team1 wins an "average" map)."""
    combined = combine_win_rates(team1_map_pool_rate, team2_map_pool_rate)
    return [combined] * best_of


def combine_win_rates(rate_a: float, rate_b: float) -> float:
    """Combines two independent win-rate estimates into a single head-to-head
    probability via the log-odds average (equivalent to a simple Bradley-Terry
    pairing of each team's rate against a common baseline of 0.5)."""
    import math

    def logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    return sigmoid(logit(rate_a) - logit(rate_b))
