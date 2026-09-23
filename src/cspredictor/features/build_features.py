"""Builds the leakage-safe map-level training table from the SQLite DB, and
exposes the same chronological-replay trackers so `predict.py` can ask for a
team's *current* (as-of-today) state using identical logic to training time.

Chronological replay: matches are processed in date order; every feature for
a match is read from trackers *before* that match's own result is folded in,
so no row ever sees information from its own outcome or from the future.

Match-level state (Elo, head-to-head, rest, congestion, roster) is snapshotted
once per match and reused for every map in that match, since those facts are
true "as of the start of the match day". Map-level state (per-map Elo, per-map
form) updates map-by-map *within* a multi-map match, since maps in a Bo3/Bo5
are genuinely sequential in time and a map 2 prediction can legitimately use
map 1's result from the same match.
"""

from __future__ import annotations

import bisect
import datetime as dt
from dataclasses import dataclass

import pandas as pd

from cspredictor.config import load_config
from cspredictor.features.elo import EloTracker, MapEloTracker
from cspredictor.features.rolling import (
    FormTracker,
    H2HTracker,
    RestAndCongestionTracker,
    RosterTracker,
)

PAIRED_TEAM_COLUMNS = [
    "elo",
    "map_elo",
    "form_5",
    "form_10",
    "form_20",
    "map_winrate",
    "rank",
    "rank_points",
    "rest_days",
    "congestion",
    "roster_stability_days",
    "standin",
]


def _to_date(date_str: str | None) -> dt.date | None:
    return dt.date.fromisoformat(date_str) if date_str else None


class _RankLookup:
    """Per-team sorted (date, rank, points) list, queried via bisect for the
    latest snapshot strictly before a given date."""

    def __init__(self, ranking_df: pd.DataFrame):
        self._by_team: dict[int, list[tuple[dt.date, int | None, int | None]]] = {}
        for team_id, group in ranking_df.groupby("team_id"):
            rows = sorted(
                (
                    (dt.date.fromisoformat(r.snapshot_date), r.rank, r.points)
                    for r in group.itertuples()
                ),
                key=lambda t: t[0],
            )
            self._by_team[team_id] = rows

    def lookup(self, team_id: int, before_date: dt.date) -> tuple[int | None, int | None]:
        rows = self._by_team.get(team_id)
        if not rows:
            return None, None
        dates = [r[0] for r in rows]
        idx = bisect.bisect_left(dates, before_date) - 1
        if idx < 0:
            return None, None
        _, rank, points = rows[idx]
        return rank, points


def load_raw_tables(conn) -> dict[str, pd.DataFrame]:
    matches = pd.read_sql_query(
        """
        SELECT id, hltv_id, match_date, team1_id, team2_id, best_of, is_lan
        FROM matches
        WHERE match_date IS NOT NULL AND team1_id IS NOT NULL AND team2_id IS NOT NULL
        ORDER BY match_date, hltv_id
        """,
        conn,
    )
    maps = pd.read_sql_query(
        """
        SELECT match_id, map_order, map_name, team1_score, team2_score,
               winner_team_id, picked_by_team_id
        FROM maps
        WHERE team1_score IS NOT NULL AND team2_score IS NOT NULL
        ORDER BY match_id, map_order
        """,
        conn,
    )
    rosters = pd.read_sql_query("SELECT match_id, team_id, player_id FROM rosters", conn)
    rankings = pd.read_sql_query(
        "SELECT team_id, snapshot_date, rank, points FROM ranking_snapshots", conn
    )
    return {"matches": matches, "maps": maps, "rosters": rosters, "rankings": rankings}


@dataclass
class ReplayState:
    """Trackers as they stand after replaying all of history — i.e. "now"."""

    elo: EloTracker
    map_elo: MapEloTracker
    form: FormTracker
    h2h: H2HTracker
    rest_cong: RestAndCongestionTracker
    roster: RosterTracker
    rank_lookup: _RankLookup
    rows: list[dict]


def _team_snapshot(state: ReplayState, team_id: int, as_of: dt.date) -> dict:
    """The same paired-column values used at training time, computed fresh
    for `team_id` as of `as_of` — used both mid-replay (as_of = that match's
    date) and for live prediction (as_of = today)."""
    rank, points = state.rank_lookup.lookup(team_id, as_of)
    return {
        "elo": state.elo.rating(team_id),
        "form_5": state.form.win_rates(team_id)[5],
        "form_10": state.form.win_rates(team_id)[10],
        "form_20": state.form.win_rates(team_id)[20],
        "rest_days": state.rest_cong.rest_days(team_id, as_of),
        "congestion": state.rest_cong.congestion(team_id, as_of),
        "roster_stability_days": state.roster.stability_days(team_id, as_of),
        "rank": rank,
        "rank_points": points,
    }


def replay(conn, config: dict | None = None) -> ReplayState:
    cfg = (config or load_config())["features"]
    tables = load_raw_tables(conn)
    matches, maps, rosters, rankings = (
        tables["matches"], tables["maps"], tables["rosters"], tables["rankings"]
    )

    maps_by_match = {mid: g.sort_values("map_order") for mid, g in maps.groupby("match_id")}
    rosters_by_match_team = {
        (mid, tid): frozenset(g["player_id"]) for (mid, tid), g in rosters.groupby(["match_id", "team_id"])
    }

    state = ReplayState(
        elo=EloTracker(cfg["elo"]["initial_rating"], cfg["elo"]["k_factor"]),
        map_elo=MapEloTracker(cfg["elo"]["initial_rating"], cfg["elo"]["map_k_factor"]),
        form=FormTracker(cfg["rolling_windows"]),
        h2h=H2HTracker(),
        rest_cong=RestAndCongestionTracker(cfg["recent_days_congestion"]),
        roster=RosterTracker(),
        rank_lookup=_RankLookup(rankings),
        rows=[],
    )

    for match in matches.itertuples():
        match_date = _to_date(match.match_date)
        if match_date is None:
            continue
        t1, t2 = match.team1_id, match.team2_id
        match_maps = maps_by_match.get(match.id)
        if match_maps is None or match_maps.empty:
            continue

        t1_snap = _team_snapshot(state, t1, match_date)
        t2_snap = _team_snapshot(state, t2, match_date)
        h2h_rate, h2h_n = state.h2h.win_rate(t1, t2)

        t1_roster = rosters_by_match_team.get((match.id, t1), frozenset())
        t2_roster = rosters_by_match_team.get((match.id, t2), frozenset())
        t1_standin = state.roster.is_standin_match(t1, t1_roster)
        t2_standin = state.roster.is_standin_match(t2, t2_roster)

        maps_won = {t1: 0, t2: 0}

        for m in match_maps.itertuples():
            if m.team1_score == m.team2_score:
                continue  # shouldn't happen, but skip indeterminate maps defensively
            team1_won_map = bool(m.team1_score > m.team2_score)
            map_name = m.map_name or "unknown"

            pick = 0
            if m.picked_by_team_id == t1:
                pick = 1
            elif m.picked_by_team_id == t2:
                pick = -1

            state.rows.append(
                {
                    "match_id": match.id,
                    "map_order": m.map_order,
                    "map_name": map_name,
                    "match_date": match.match_date,
                    "best_of": match.best_of,
                    "is_lan": match.is_lan,
                    "team1_id": t1,
                    "team2_id": t2,
                    "team1_pick": pick,
                    "h2h_team1_rate": h2h_rate,
                    "h2h_n": h2h_n,
                    "team1_won_map": int(team1_won_map),
                    "team1_elo": t1_snap["elo"],
                    "team2_elo": t2_snap["elo"],
                    "team1_map_elo": state.map_elo.rating(t1, map_name),
                    "team2_map_elo": state.map_elo.rating(t2, map_name),
                    "team1_form_5": t1_snap["form_5"],
                    "team2_form_5": t2_snap["form_5"],
                    "team1_form_10": t1_snap["form_10"],
                    "team2_form_10": t2_snap["form_10"],
                    "team1_form_20": t1_snap["form_20"],
                    "team2_form_20": t2_snap["form_20"],
                    "team1_map_winrate": state.form.map_win_rate(t1, map_name),
                    "team2_map_winrate": state.form.map_win_rate(t2, map_name),
                    "team1_rank": t1_snap["rank"],
                    "team2_rank": t2_snap["rank"],
                    "team1_rank_points": t1_snap["rank_points"],
                    "team2_rank_points": t2_snap["rank_points"],
                    "team1_rest_days": t1_snap["rest_days"],
                    "team2_rest_days": t2_snap["rest_days"],
                    "team1_congestion": t1_snap["congestion"],
                    "team2_congestion": t2_snap["congestion"],
                    "team1_roster_stability_days": t1_snap["roster_stability_days"],
                    "team2_roster_stability_days": t2_snap["roster_stability_days"],
                    "team1_standin": t1_standin,
                    "team2_standin": t2_standin,
                }
            )

            # per-map trackers update map-by-map, within the match
            state.map_elo.update(t1, t2, map_name, team1_won_map)
            state.form.update(t1, team1_won_map, map_name)
            state.form.update(t2, not team1_won_map, map_name)

            maps_won[t1] += int(team1_won_map)
            maps_won[t2] += int(not team1_won_map)

        # --- match-level trackers update once, after all maps are processed ---
        if maps_won[t1] != maps_won[t2]:
            t1_won_match = maps_won[t1] > maps_won[t2]
            state.elo.update(t1, t2, t1_won_match)
            state.h2h.update(t1, t2, t1_won_match)
        state.rest_cong.update(t1, match_date)
        state.rest_cong.update(t2, match_date)
        state.roster.update(t1, t1_roster, match_date)
        state.roster.update(t2, t2_roster, match_date)

    return state


def build_map_training_table(conn, config: dict | None = None) -> pd.DataFrame:
    return pd.DataFrame(replay(conn, config).rows)


def current_team_snapshot(
    state: ReplayState, team_id: int, map_name: str | None = None, as_of: dt.date | None = None
) -> dict:
    """Live, as-of-today feature snapshot for one team, for use at prediction
    time. `map_name` also fills in the per-map Elo/win-rate fields when known
    (post-veto); leave it None for a pre-veto, map-agnostic snapshot."""
    as_of = as_of or dt.date.today()
    snap = _team_snapshot(state, team_id, as_of)
    snap["map_elo"] = state.map_elo.rating(team_id, map_name) if map_name else None
    snap["map_winrate"] = state.form.map_win_rate(team_id, map_name) if map_name else None
    return snap


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate each row with team1/team2 swapped so the model can't learn
    a spurious "team1 column tends to win" bias from listing order."""
    mirrored = df.copy()
    for col in PAIRED_TEAM_COLUMNS:
        mirrored[f"team1_{col}"], mirrored[f"team2_{col}"] = (
            df[f"team2_{col}"].to_numpy(),
            df[f"team1_{col}"].to_numpy(),
        )
    mirrored["team1_id"], mirrored["team2_id"] = df["team2_id"].to_numpy(), df["team1_id"].to_numpy()
    mirrored["team1_pick"] = -df["team1_pick"].to_numpy()
    mirrored["h2h_team1_rate"] = 1.0 - df["h2h_team1_rate"].to_numpy(dtype=float)
    mirrored["team1_won_map"] = 1 - df["team1_won_map"].to_numpy()
    return pd.concat([df, mirrored], ignore_index=True)


def to_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Collapses paired team1_*/team2_* columns into model-ready diff features."""
    out = pd.DataFrame(index=df.index)
    out["match_id"] = df["match_id"]
    out["map_order"] = df["map_order"]
    out["map_name"] = df["map_name"].astype("category")
    out["match_date"] = df["match_date"]
    out["team1_id"] = df["team1_id"]
    out["team2_id"] = df["team2_id"]
    out["best_of"] = df["best_of"]
    out["is_lan"] = df["is_lan"]

    out["elo_diff"] = df["team1_elo"] - df["team2_elo"]
    out["map_elo_diff"] = df["team1_map_elo"] - df["team2_map_elo"]
    out["form_5_diff"] = df["team1_form_5"] - df["team2_form_5"]
    out["form_10_diff"] = df["team1_form_10"] - df["team2_form_10"]
    out["form_20_diff"] = df["team1_form_20"] - df["team2_form_20"]
    out["map_winrate_diff"] = df["team1_map_winrate"] - df["team2_map_winrate"]
    # lower HLTV rank number = better team, so this diff is positive when team1 is favored
    out["rank_diff"] = df["team2_rank"] - df["team1_rank"]
    out["rank_points_diff"] = df["team1_rank_points"] - df["team2_rank_points"]
    out["rest_days_diff"] = df["team1_rest_days"] - df["team2_rest_days"]
    out["congestion_diff"] = df["team2_congestion"] - df["team1_congestion"]
    out["roster_stability_diff"] = df["team1_roster_stability_days"] - df["team2_roster_stability_days"]
    out["standin_diff"] = df["team2_standin"].astype("Int64") - df["team1_standin"].astype("Int64")
    out["h2h_team1_rate"] = df["h2h_team1_rate"]
    out["h2h_n"] = df["h2h_n"]
    out["team1_pick"] = df["team1_pick"]

    out["target"] = df["team1_won_map"]
    return out


def snapshot_pair_to_model_row(
    t1_snap: dict,
    t2_snap: dict,
    *,
    h2h_team1_rate: float | None,
    h2h_n: int,
    team1_pick: int,
    best_of: int | None,
    is_lan: bool | None,
    map_name: str | None,
) -> dict:
    """Same diff-feature logic as `to_model_matrix`, applied to a single live
    pair of snapshots (see `current_team_snapshot`) for prediction."""

    def diff(key, reverse=False):
        a, b = t1_snap.get(key), t2_snap.get(key)
        if a is None or b is None:
            return None
        return (b - a) if reverse else (a - b)

    return {
        "map_name": map_name or "unknown",
        "best_of": best_of,
        "is_lan": is_lan,
        "elo_diff": diff("elo"),
        "map_elo_diff": diff("map_elo"),
        "form_5_diff": diff("form_5"),
        "form_10_diff": diff("form_10"),
        "form_20_diff": diff("form_20"),
        "map_winrate_diff": diff("map_winrate"),
        "rank_diff": diff("rank", reverse=True),
        "rank_points_diff": diff("rank_points"),
        "rest_days_diff": diff("rest_days"),
        "congestion_diff": diff("congestion", reverse=True),
        "roster_stability_diff": diff("roster_stability_days"),
        "standin_diff": None,
        "h2h_team1_rate": h2h_team1_rate,
        "h2h_n": h2h_n,
        "team1_pick": team1_pick,
    }
