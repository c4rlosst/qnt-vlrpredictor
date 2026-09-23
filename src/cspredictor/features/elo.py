"""Elo rating trackers: one overall (match-level) rating per team, and one
rating per (team, map) pair. Pure logic, no I/O — easy to unit test and to
reason about independently of the scraper/DB.

Usage is always: read the *pre*-update rating for a team (or team+map) via
`rating()` / `map_rating()` BEFORE calling `update()` / `update_map()` for
that same match, so downstream feature rows never see a rating that already
incorporates the outcome they're trying to predict.
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


class MapEloTracker:
    """Same idea as EloTracker but keyed on (team_id, map_name)."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 24.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: dict[tuple[int, str], float] = {}

    def rating(self, team_id: int, map_name: str) -> float:
        return self._ratings.get((team_id, map_name), self.initial_rating)

    def update(self, team_a: int, team_b: int, map_name: str, a_won: bool) -> tuple[float, float]:
        ra, rb = self.rating(team_a, map_name), self.rating(team_b, map_name)
        exp_a = expected_score(ra, rb)
        actual_a = 1.0 if a_won else 0.0
        new_a = ra + self.k_factor * (actual_a - exp_a)
        new_b = rb + self.k_factor * ((1.0 - actual_a) - (1.0 - exp_a))
        self._ratings[(team_a, map_name)] = new_a
        self._ratings[(team_b, map_name)] = new_b
        return new_a, new_b
