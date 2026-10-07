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


def side_rounds(
    score1: int, score2: int, team1_first_side: str,
) -> tuple[int, int, int, int]:
    """Regulation rounds each team PLAYED per side on one map, as
    (t1_atk_played, t1_def_played, t2_atk_played, t2_def_played).

    A map is two 12-round halves (sides swap at half-time); a map that ends
    early plays fewer second-half rounds. Overtime rounds are ignored: their
    side split isn't listed per team, and they're a tiny share of rounds.
    """
    regulation = min(score1 + score2, 24)
    first_half = min(12, regulation)
    second_half = regulation - first_half
    if team1_first_side == "atk":
        return first_half, second_half, second_half, first_half
    return second_half, first_half, first_half, second_half


class SideTracker:
    """Attack / defence strength per team, and how attacker-friendly each map is.

    A team's side "edge" is its round win rate on that side minus the league's
    average for that side, over its last `window` maps and shrunk toward the
    league average by `prior_rounds` pseudo-rounds, so a team with two maps of
    history sits near 0 (unknown) instead of at an extreme.
    """

    def __init__(self, window: int = 20, prior_rounds: float = 48.0):
        self.window = window
        self.prior_rounds = prior_rounds
        # per team: deque of (atk_won, atk_played, def_won, def_played), one entry per map
        self._teams: dict[int, deque[tuple[int, int, int, int]]] = {}
        self._league = [0, 0]  # attack rounds won / played by attackers, all teams
        self._maps: dict[str, list[int]] = {}  # map -> [attack rounds won, played] league-wide

    @property
    def league_atk_rate(self) -> float:
        won, played = self._league
        return won / played if played else 0.5

    def _shrunk(self, won: float, played: float, prior: float) -> float:
        return (won + self.prior_rounds * prior) / (played + self.prior_rounds)

    def edges(self, team_id: int) -> tuple[float, float]:
        """(attack_edge, defence_edge): 0.0 = league average, +0.05 = wins 5 points more
        of its rounds than an average team on that side."""
        history = self._teams.get(team_id)
        league_atk = self.league_atk_rate
        if not history:
            return 0.0, 0.0
        atk_w = sum(h[0] for h in history)
        atk_p = sum(h[1] for h in history)
        def_w = sum(h[2] for h in history)
        def_p = sum(h[3] for h in history)
        return (
            self._shrunk(atk_w, atk_p, league_atk) - league_atk,
            self._shrunk(def_w, def_p, 1.0 - league_atk) - (1.0 - league_atk),
        )

    def map_atk_bias(self, map_name: str) -> float:
        """How attacker-friendly a map is: its league-wide attack round win rate minus 0.5."""
        won, played = self._maps.get(map_name, (0, 0))
        return self._shrunk(won, played, self.league_atk_rate) - 0.5

    def update(
        self, team1: int, team2: int, map_name: str, score1: int, score2: int,
        t1_atk_won: int, t1_def_won: int, t2_atk_won: int, t2_def_won: int, team1_first_side: str,
    ) -> bool:
        """Record one finished map. Returns False (and records nothing) if the
        side counts don't add up, i.e. the scraped numbers can't be trusted."""
        t1_atk_p, t1_def_p, t2_atk_p, t2_def_p = side_rounds(score1, score2, team1_first_side)
        # every round has one winner: attackers' wins + defenders' wins = rounds played on that side
        if t1_atk_won + t2_def_won != t1_atk_p or t1_def_won + t2_atk_won != t2_atk_p:
            return False
        self._teams.setdefault(team1, deque(maxlen=self.window)).append((t1_atk_won, t1_atk_p, t1_def_won, t1_def_p))
        self._teams.setdefault(team2, deque(maxlen=self.window)).append((t2_atk_won, t2_atk_p, t2_def_won, t2_def_p))
        atk_won, atk_played = t1_atk_won + t2_atk_won, t1_atk_p + t2_atk_p
        self._league[0] += atk_won
        self._league[1] += atk_played
        entry = self._maps.setdefault(map_name, [0, 0])
        entry[0] += atk_won
        entry[1] += atk_played
        return True


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

    def __init__(self, history: int = 10):
        self._last_roster: dict[int, frozenset[int]] = {}
        self._changed_on: dict[int, dt.date] = {}
        self._history: dict[int, deque[frozenset[int]]] = {}
        self._history_len = history

    def last_roster(self, team_id: int) -> frozenset[int]:
        return self._last_roster.get(team_id, frozenset())

    def stability_days(self, team_id: int, current_date: dt.date) -> int | None:
        changed_on = self._changed_on.get(team_id)
        return (current_date - changed_on).days if changed_on else None

    def is_standin_match(self, team_id: int, current_roster: frozenset[int]) -> bool | None:
        prev = self._last_roster.get(team_id)
        if prev is None or not current_roster:
            return None
        return prev != current_roster

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
