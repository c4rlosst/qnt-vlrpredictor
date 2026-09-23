"""End-to-end prediction for a hypothetical or upcoming match: resolves team
names, snapshots current (as-of-today) state, runs the map model, and derives
a match-winner probability + map-score distribution via combinatorics.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import lightgbm as lgb
import pandas as pd

from cspredictor.config import load_config
from cspredictor.features.build_features import (
    current_team_snapshot,
    replay,
    snapshot_pair_to_model_row,
)
from cspredictor.models.combinatorics import (
    combine_win_rates,
    expected_map_probs_from_pool,
    match_win_probability,
    score_distribution,
)
from cspredictor.models.map_model import predict_proba
from cspredictor.storage.db import find_team_id_by_name


class TeamNotFoundError(ValueError):
    pass


def predict_match(
    conn: sqlite3.Connection,
    model: lgb.Booster,
    team1_name: str,
    team2_name: str,
    best_of: int = 3,
    maps: list[str] | None = None,
    picks: dict[str, str] | None = None,
    is_lan: bool | None = None,
    config: dict | None = None,
) -> dict:
    cfg = config or load_config()
    team1_id = find_team_id_by_name(conn, team1_name)
    team2_id = find_team_id_by_name(conn, team2_name)
    if team1_id is None:
        raise TeamNotFoundError(f"no team found matching {team1_name!r}")
    if team2_id is None:
        raise TeamNotFoundError(f"no team found matching {team2_name!r}")

    state = replay(conn, cfg)
    today = dt.date.today()
    picks = picks or {}

    if maps:
        rows = []
        for map_name in maps:
            t1_snap = current_team_snapshot(state, team1_id, map_name, today)
            t2_snap = current_team_snapshot(state, team2_id, map_name, today)
            h2h_rate, h2h_n = state.h2h.win_rate(team1_id, team2_id)
            picker = picks.get(map_name)
            pick_flag = 0
            if picker and picker.strip().lower() == team1_name.strip().lower():
                pick_flag = 1
            elif picker and picker.strip().lower() == team2_name.strip().lower():
                pick_flag = -1
            rows.append(
                snapshot_pair_to_model_row(
                    t1_snap, t2_snap,
                    h2h_team1_rate=h2h_rate, h2h_n=h2h_n, team1_pick=pick_flag,
                    best_of=best_of, is_lan=is_lan, map_name=map_name,
                )
            )
        row_df = pd.DataFrame(rows)
        map_probs = predict_proba(model, row_df).tolist()
        mode = "post-veto"
    else:
        t1_snap = current_team_snapshot(state, team1_id, None, today)
        t2_snap = current_team_snapshot(state, team2_id, None, today)
        h2h_rate, h2h_n = state.h2h.win_rate(team1_id, team2_id)
        form1 = t1_snap.get("form_10")
        form2 = t2_snap.get("form_10")
        if form1 is not None and form2 is not None:
            generic_prob = combine_win_rates(form1, form2)
        else:
            generic_prob = 0.5
        map_probs = expected_map_probs_from_pool(generic_prob, 0.5, best_of)
        maps = [f"map{i+1} (unknown pre-veto)" for i in range(best_of)]
        mode = "pre-veto (approximate — map pool unknown)"

    dist = score_distribution(map_probs, best_of)
    team1_win_prob = match_win_probability(map_probs, best_of)

    return {
        "team1": team1_name,
        "team2": team2_name,
        "team1_id": team1_id,
        "team2_id": team2_id,
        "best_of": best_of,
        "mode": mode,
        "maps": maps,
        "map_probs": map_probs,
        "team1_win_prob": team1_win_prob,
        "team2_win_prob": 1.0 - team1_win_prob,
        "score_distribution": dist,
    }


def format_prediction(result: dict) -> str:
    lines = [
        f"{result['team1']} vs {result['team2']}  (Bo{result['best_of']}, {result['mode']})",
        "",
        f"  {result['team1']}: {result['team1_win_prob']:.1%}",
        f"  {result['team2']}: {result['team2_win_prob']:.1%}",
        "",
        "Map probabilities (P(team1 wins)):",
    ]
    for map_name, p in zip(result["maps"], result["map_probs"]):
        lines.append(f"  {map_name:20s} {p:.1%}")
    lines.append("")
    lines.append("Score distribution:")
    for (a, b), p in sorted(result["score_distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {result['team1']} {a}-{b} {result['team2']}: {p:.1%}")
    return "\n".join(lines)
