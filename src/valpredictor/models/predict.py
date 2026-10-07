"""Prediction for a matchup: each team's current Elo -> a per-map win chance
-> match-winner probability and exact-score distribution.

With one rating per team the per-map chance is the same on every map, so
which maps get played (the veto) cannot change it.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

from valpredictor.config import load_config
from valpredictor.features.build_features import ReplayState, current_team_snapshot, replay
from valpredictor.models.combinatorics import p_distance, score_distribution
from valpredictor.models.elo_model import EloModel
from valpredictor.storage.db import find_team_id_by_name


class TeamNotFoundError(ValueError):
    pass


def _resolve_team(conn: sqlite3.Connection, name: str) -> int:
    team_id = find_team_id_by_name(conn, name)
    if team_id is None:
        raise TeamNotFoundError(f"no team found matching {name!r} (not in the database yet?)")
    return team_id


def predict_match(
    conn: sqlite3.Connection,
    model: EloModel,
    team1_name: str,
    team2_name: str,
    best_of: int = 3,
    config: dict | None = None,
    state: ReplayState | None = None,
) -> dict:
    cfg = config or load_config()
    team1_id = _resolve_team(conn, team1_name)
    team2_id = _resolve_team(conn, team2_name)

    state = state or replay(conn, cfg)
    today = dt.date.today()
    ctx1, ctx2 = current_team_snapshot(state, team1_id, today), current_team_snapshot(state, team2_id, today)

    p_map = model.map_prob(ctx1["elo"] - ctx2["elo"])
    dist = score_distribution([p_map] * best_of, best_of)
    needed = best_of // 2 + 1
    team1_win_prob = sum(p for (a, b), p in dist.items() if a == needed)

    return {
        "team1": team1_name,
        "team2": team2_name,
        "best_of": best_of,
        "elo_gap": ctx1["elo"] - ctx2["elo"],
        "p_map_team1": p_map,
        "context": {
            "team1": {**ctx1, "elo": round(ctx1["elo"])},
            "team2": {**ctx2, "elo": round(ctx2["elo"])},
        },
        "team1_win_prob": team1_win_prob,
        "team2_win_prob": 1.0 - team1_win_prob,
        "p_distance": p_distance(dist, best_of),
        "score_distribution": dist,
    }


def result_to_json(result: dict) -> dict:
    """JSON-safe copy of a predict_match result (tuple keys -> 'a-b' strings)."""
    out = dict(result)
    out["score_distribution"] = {f"{a}-{b}": p for (a, b), p in result["score_distribution"].items()}
    return out


def format_prediction(result: dict) -> str:
    t1, t2 = result["team1"], result["team2"]
    lines = [
        f"{t1} vs {t2}  (Bo{result['best_of']})",
        "",
        f"  {t1}: {result['team1_win_prob']:.1%}",
        f"  {t2}: {result['team2_win_prob']:.1%}",
        "",
        f"Chance of winning any single map: {t1} {result['p_map_team1']:.1%} (Elo gap {result['elo_gap']:+.0f})",
        "",
        "Team context:",
    ]
    for key in ("team1", "team2"):
        c = result["context"][key]
        cont = "n/a" if c["roster_continuity"] is None else f"{c['roster_continuity']:.0%}"
        since = "n/a" if c["days_since_lineup_change"] is None else f"{c['days_since_lineup_change']}d"
        lines.append(f"  {result[key]:16s} Elo {c['elo']} · lineup continuity {cont} · same lineup {since}")
    lines.append("")
    if result.get("p_distance") is not None:
        last = result["best_of"] // 2 + 1
        lines.append(f"Goes to a deciding map (a {last}-{last - 1} series): {result['p_distance']:.0%}")
    lines.append("Score distribution:")
    for (a, b), p in sorted(result["score_distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {t1} {a}-{b} {t2}: {p:.1%}")
    return "\n".join(lines)
