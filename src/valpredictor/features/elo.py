"""Elo rating: one rating per team, updated after every series.

Pure logic, no I/O. Always read a team's rating BEFORE calling `update()` for
that same match, so a prediction row never sees a rating that already contains
the result it is trying to predict.
"""

from __future__ import annotations


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A against B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


class EloTracker:
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: dict[int, float] = {}

    def rating(self, team_id: int) -> float:
        return self._ratings.get(team_id, self.initial_rating)

    def update(self, team_a: int, team_b: int, a_won: bool) -> tuple[float, float]:
        """Update both teams' ratings after a result. Returns (new_a, new_b)."""
        ra, rb = self.rating(team_a), self.rating(team_b)
        exp_a = expected_score(ra, rb)
        actual_a = 1.0 if a_won else 0.0
        new_a = ra + self.k_factor * (actual_a - exp_a)
        new_b = rb + self.k_factor * ((1.0 - actual_a) - (1.0 - exp_a))
        self._ratings[team_a] = new_a
        self._ratings[team_b] = new_b
        return new_a, new_b

    def regress(self, team_id: int, fraction: float) -> float:
        """Pull a team's rating `fraction` (0..1) of the way back to the
        starting rating — used when its lineup changed, since past results
        were earned by different players."""
        fraction = min(max(fraction, 0.0), 1.0)
        r = self.rating(team_id)
        self._ratings[team_id] = r + (self.initial_rating - r) * fraction
        return self._ratings[team_id]
