"""Derives match-winner probability and map-score distribution from a list
of per-map P(team1 wins) probabilities, respecting Bo1/Bo3/Bo5 early
termination (e.g. a Bo3 ends 2-0 without a third map being played).

`score_distribution` treats maps as independent. Real series aren't: the
teams' true strength is shared by every map, so a team that wins map 1 is
more likely to be the stronger one. `shared_score_distribution` models that
with one strength shock per series (logit-normal, sd `tau`) while keeping
each map's marginal win probability exactly what the model said. On the full
tier-1 season (484 out-of-fold series, scripts/fit_series_tau.py) the best tau
is ~0.6: suggestive rather than conclusive evidence against independence
(likelihood +1.1, p ~ 0.07), but it reproduces how often Bo3s really go to a
third map (46.7% +/- 2.3% real; independence implies 49.6%, tau 0.6 implies 45.8%).
"""

from __future__ import annotations

import numpy as np

_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(21)  # E[f(Z)], Z ~ N(0, 1)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _marginal_preserving_logits(probs: np.ndarray, tau: float) -> np.ndarray:
    """Logit centres c_i with E_Z[sigmoid(c_i + tau*Z)] == p_i, found by bisection."""
    p = np.clip(probs, 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))
    lo, hi = logit - 6 * tau - 1, logit + 6 * tau + 1
    for _ in range(60):
        mid = (lo + hi) / 2
        mean = (_WEIGHTS[None, :] * _sigmoid(mid[:, None] + tau * _NODES[None, :])).sum(axis=1)
        lo, hi = np.where(mean < p, mid, lo), np.where(mean < p, hi, mid)
    return (lo + hi) / 2


def shared_score_distribution(
    map_probs: list[float], best_of: int, tau: float
) -> dict[tuple[int, int], float]:
    """Score distribution with a shared per-series strength shock (see module docstring)."""
    if tau <= 0:
        return score_distribution(map_probs, best_of)
    centres = _marginal_preserving_logits(np.asarray(map_probs, dtype=float), tau)
    out: dict[tuple[int, int], float] = {}
    for z, weight in zip(_NODES, _WEIGHTS):
        for score, p in score_distribution(list(_sigmoid(centres + tau * z)), best_of).items():
            out[score] = out.get(score, 0.0) + weight * p
    return out


def p_distance(dist: dict[tuple[int, int], float], best_of: int) -> float | None:
    """P(the series goes to its last possible map): Bo3 -> a 2-1, Bo5 -> a 3-2. None for a Bo1."""
    if best_of < 3:
        return None
    return sum(p for (a, b), p in dist.items() if a + b == best_of)


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


def match_win_probability(map_probs: list[float], best_of: int, tau: float = 0.0) -> float:
    dist = shared_score_distribution(map_probs, best_of, tau)
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
