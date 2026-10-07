"""Replays match history in time order to build each team's Elo, and a row per
past map (what the two Elo ratings were just before it was played, and who
won). `predict.py` reads the same trackers for a team's *current* state.

Every row is read from the trackers *before* that match's own result is folded
in, so no row ever sees its own outcome or the future. Lineups are known
before a match starts, so a lineup change is applied to the Elo before the
match's rows are written.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from valpredictor.config import load_config
from valpredictor.features.elo import EloTracker
from valpredictor.features.roster import RosterTracker


def load_raw_tables(conn) -> dict[str, pd.DataFrame]:
    matches = pd.read_sql_query(
        """
        SELECT id, vlr_id, match_date, team1_id, team2_id, best_of
        FROM matches
        WHERE match_date IS NOT NULL AND team1_id IS NOT NULL AND team2_id IS NOT NULL
        ORDER BY COALESCE(unix_timestamp_ms, 0), vlr_id
        """,
        conn,
    )
    maps = pd.read_sql_query(
        """
        SELECT match_id, map_order, map_name, team1_score, team2_score
        FROM maps
        WHERE team1_score IS NOT NULL AND team2_score IS NOT NULL
        ORDER BY match_id, map_order
        """,
        conn,
    )
    rosters = pd.read_sql_query("SELECT match_id, team_id, player_id FROM rosters", conn)
    return {"matches": matches, "maps": maps, "rosters": rosters}


@dataclass
class ReplayState:
    """Trackers as they stand after replaying all of history, i.e. "now"."""

    elo: EloTracker
    roster: RosterTracker
    rows: list[dict]


def replay(conn, config: dict | None = None) -> ReplayState:
    cfg = (config or load_config())["features"]["elo"]
    tables = load_raw_tables(conn)
    matches, maps, rosters = tables["matches"], tables["maps"], tables["rosters"]

    shrink_per_player = float(cfg.get("roster_change_shrink_per_player", 0.0))
    shrink_cap = float(cfg.get("roster_change_shrink_cap", 0.5))
    maps_by_match = {mid: g.sort_values("map_order") for mid, g in maps.groupby("match_id")}
    rosters_by_match_team = {
        (mid, tid): frozenset(g["player_id"]) for (mid, tid), g in rosters.groupby(["match_id", "team_id"])
    }

    state = ReplayState(
        elo=EloTracker(cfg["initial_rating"], cfg["k_factor"]), roster=RosterTracker(), rows=[]
    )

    for match in matches.itertuples():
        match_date = dt.date.fromisoformat(match.match_date)
        t1, t2 = match.team1_id, match.team2_id
        match_maps = maps_by_match.get(match.id)
        if match_maps is None or match_maps.empty:
            continue

        t1_roster = rosters_by_match_team.get((match.id, t1), frozenset())
        t2_roster = rosters_by_match_team.get((match.id, t2), frozenset())

        # A changed lineup means a team's past results were earned by other
        # players: pull its Elo back toward average in proportion to how many
        # players are new.
        for team, roster in ((t1, t1_roster), (t2, t2_roster)):
            changed = state.roster.players_changed(team, roster)
            if changed and shrink_per_player:
                state.elo.regress(team, min(shrink_cap, shrink_per_player * changed))

        elo1, elo2 = state.elo.rating(t1), state.elo.rating(t2)
        maps_won = {t1: 0, t2: 0}
        for m in match_maps.itertuples():
            if m.team1_score == m.team2_score:
                continue  # indeterminate; shouldn't happen
            team1_won = bool(m.team1_score > m.team2_score)
            state.rows.append(
                {
                    "match_id": match.id,
                    "map_order": m.map_order,
                    "map_name": m.map_name or "unknown",
                    "match_date": match.match_date,
                    "best_of": match.best_of,
                    "team1_id": t1,
                    "team2_id": t2,
                    "team1_elo": elo1,
                    "team2_elo": elo2,
                    "elo_diff": elo1 - elo2,
                    "team1_won_map": int(team1_won),
                }
            )
            maps_won[t1] += int(team1_won)
            maps_won[t2] += int(not team1_won)

        if maps_won[t1] != maps_won[t2]:
            state.elo.update(t1, t2, maps_won[t1] > maps_won[t2])
        state.roster.update(t1, t1_roster, match_date)
        state.roster.update(t2, t2_roster, match_date)

    return state


def build_map_table(conn, config: dict | None = None) -> pd.DataFrame:
    return pd.DataFrame(replay(conn, config).rows)


def current_team_snapshot(state: ReplayState, team_id: int, as_of: dt.date | None = None) -> dict:
    """A team's rating and roster situation as of today, for display and prediction."""
    as_of = as_of or dt.date.today()
    roster = state.roster.last_roster(team_id)
    return {
        "elo": state.elo.rating(team_id),
        "roster_continuity": state.roster.continuity(team_id, roster),
        "days_since_lineup_change": state.roster.stability_days(team_id, as_of),
    }
