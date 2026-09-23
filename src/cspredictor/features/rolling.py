"""Chronological trackers for form, head-to-head, rest, congestion, and
roster stability. Same pre/post-update discipline as elo.py: always read a
team's state before calling the corresponding update for the current match.
"""

from __future__ import annotations

import datetime as dt
from collections import deque


class FormTracker:
    """Rolling win rate over the last N results, both overall and per-map."""

    def __init__(self, windows: list[int]):
        self.windows = sorted(windows)
        self.max_window = max(windows)
        self._overall: dict[int, deque[int]] = {}
        self._per_map: dict[tuple[int, str], deque[int]] = {}

    def win_rates(self, team_id: int) -> dict[int, float | None]:
        history = self._overall.get(team_id, deque())
        out: dict[int, float | None] = {}
        for w in self.windows:
            recent = list(history)[-w:]
            out[w] = (sum(recent) / len(recent)) if recent else None
        return out

    def map_win_rate(self, team_id: int, map_name: str) -> float | None:
        history = self._per_map.get((team_id, map_name), deque())
        if not history:
            return None
        return sum(history) / len(history)

    def update(self, team_id: int, won: bool, map_name: str | None = None) -> None:
        self._overall.setdefault(team_id, deque(maxlen=self.max_window)).append(int(won))
        if map_name:
            self._per_map.setdefault((team_id, map_name), deque(maxlen=200)).append(int(won))


class H2HTracker:
    """Recency-weighted head-to-head win rate between two teams."""

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self._history: dict[frozenset, list[int]] = {}  # 1 if the *lower* team_id won

    def win_rate(self, team_a: int, team_b: int) -> tuple[float | None, int]:
        """Returns (team_a's weighted win rate vs team_b, num prior meetings)."""
        key = frozenset((team_a, team_b))
        results = self._history.get(key, [])
        if not results:
            return None, 0
        lower_id = min(team_a, team_b)
        weights = [self.decay ** i for i in range(len(results))][::-1]
        weighted_sum = sum(r * w for r, w in zip(results, weights))
        total_weight = sum(weights)
        lower_win_rate = weighted_sum / total_weight
        rate = lower_win_rate if team_a == lower_id else 1.0 - lower_win_rate
        return rate, len(results)

    def update(self, team_a: int, team_b: int, a_won: bool) -> None:
        key = frozenset((team_a, team_b))
        lower_id = min(team_a, team_b)
        result = int(a_won) if team_a == lower_id else int(not a_won)
        self._history.setdefault(key, []).append(result)


class RestAndCongestionTracker:
    """Rest days since last match, and match count in a trailing window."""

    def __init__(self, congestion_window_days: int = 14):
        self.congestion_window_days = congestion_window_days
        self._last_date: dict[int, dt.date] = {}
        self._recent_dates: dict[int, list[dt.date]] = {}

    def rest_days(self, team_id: int, current_date: dt.date) -> int | None:
        last = self._last_date.get(team_id)
        return (current_date - last).days if last else None

    def congestion(self, team_id: int, current_date: dt.date) -> int:
        cutoff = current_date - dt.timedelta(days=self.congestion_window_days)
        dates = self._recent_dates.get(team_id, [])
        return sum(1 for d in dates if cutoff <= d < current_date)

    def update(self, team_id: int, match_date: dt.date) -> None:
        self._last_date[team_id] = match_date
        dates = self._recent_dates.setdefault(team_id, [])
        dates.append(match_date)
        cutoff = match_date - dt.timedelta(days=self.congestion_window_days * 3)
        self._recent_dates[team_id] = [d for d in dates if d >= cutoff]


class RosterTracker:
    """Roster-stability heuristic: tracks each team's most recent lineup and
    the date it last changed. A stand-in flag fires when the current match's
    roster differs from the immediately preceding one.

    Simplification (documented, not a bug): this treats any single-player
    change as "the roster changed" — it doesn't try to distinguish a
    one-match illness stand-in from a permanent roster overhaul. Good enough
    to capture "this team is not playing its usual five", refine later if
    the feature proves valuable.
    """

    def __init__(self):
        self._last_roster: dict[int, frozenset[int]] = {}
        self._changed_on: dict[int, dt.date] = {}

    def stability_days(self, team_id: int, current_date: dt.date) -> int | None:
        changed_on = self._changed_on.get(team_id)
        return (current_date - changed_on).days if changed_on else None

    def is_standin_match(self, team_id: int, current_roster: frozenset[int]) -> bool | None:
        prev = self._last_roster.get(team_id)
        if prev is None or not current_roster:
            return None
        return prev != current_roster

    def update(self, team_id: int, current_roster: frozenset[int], match_date: dt.date) -> None:
        if not current_roster:
            return
        prev = self._last_roster.get(team_id)
        if prev != current_roster:
            self._changed_on[team_id] = match_date
        self._last_roster[team_id] = current_roster
