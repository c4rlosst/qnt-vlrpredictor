"""Roster tracking: each team's latest lineups, used to tell when a lineup has
changed (which pulls the team's Elo back toward average) and to show how
settled a lineup is.
"""

from __future__ import annotations

import datetime as dt
from collections import deque


class RosterTracker:
    """Tracks each team's most recent lineup, the date it last changed, and its
    last few lineups (for a continuity score)."""

    def __init__(self, history: int = 10):
        self._last_roster: dict[int, frozenset[int]] = {}
        self._changed_on: dict[int, dt.date] = {}
        self._history: dict[int, deque[frozenset[int]]] = {}
        self._history_len = history

    def last_roster(self, team_id: int) -> frozenset[int]:
        return self._last_roster.get(team_id, frozenset())

    def stability_days(self, team_id: int, current_date: dt.date) -> int | None:
        """Days since the lineup last changed (None until a lineup has been seen)."""
        changed_on = self._changed_on.get(team_id)
        return (current_date - changed_on).days if changed_on else None

    def players_changed(self, team_id: int, current_roster: frozenset[int]) -> int | None:
        """How many players in `current_roster` were not in the previous lineup."""
        prev = self._last_roster.get(team_id)
        if prev is None or not current_roster:
            return None
        return len(current_roster - prev)

    def continuity(self, team_id: int, current_roster: frozenset[int]) -> float | None:
        """0..1: how much of `current_roster` appeared in the team's last few
        lineups, averaged. 1.0 = the same players as always; a freshly rebuilt
        lineup scores low until the new players have played together."""
        history = self._history.get(team_id)
        if not history or not current_roster:
            return None
        return sum(len(current_roster & past) / len(current_roster) for past in history) / len(history)

    def update(self, team_id: int, current_roster: frozenset[int], match_date: dt.date) -> None:
        if not current_roster:
            return
        prev = self._last_roster.get(team_id)
        if prev != current_roster:
            self._changed_on[team_id] = match_date
        self._last_roster[team_id] = current_roster
        self._history.setdefault(team_id, deque(maxlen=self._history_len)).append(current_roster)
