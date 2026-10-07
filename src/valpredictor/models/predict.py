"""End-to-end prediction for a hypothetical or upcoming match: resolves teams,
snapshots current (as-of-today) state, runs the map model, and derives a
match-winner probability + map-score distribution via combinatorics.

Two modes:
  - post-veto: the maps (and who picked them) are known -> one model call per map.
  - pre-veto:  the veto is simulated (each team bans its worst maps and picks
    its best) for both possible veto orders and the two outcomes are averaged.

The series maths shares one team-strength shock across a series' maps
(`model.series_strength_sd`), because maps in a series are positively
correlated; see models/combinatorics.py.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import lightgbm as lgb
import pandas as pd

from valpredictor.config import load_config
from valpredictor.features.build_features import (
    ReplayState,
    current_team_snapshot,
    replay,
    snapshot_pair_to_model_row,
)
from valpredictor.models.combinatorics import match_win_probability, p_distance, shared_score_distribution
from valpredictor.models.map_model import predict_proba
from valpredictor.models.veto import simulate_veto, trim_pool
from valpredictor.storage.db import find_team_id_by_name

POOL_LOOKBACK_DAYS = 120
POOL_MIN_PLAYS = 3


class TeamNotFoundError(ValueError):
    pass


def active_map_pool(conn: sqlite3.Connection, as_of: dt.date | None = None) -> dict[str, float]:
    """{map_name: weight} for maps recently played, weights summing to 1.
    Falls back to every map ever played if the recent window is too thin."""
    as_of = as_of or dt.date.today()
    since = (as_of - dt.timedelta(days=POOL_LOOKBACK_DAYS)).isoformat()
    rows = conn.execute(
        """
        SELECT m.map_name AS map_name, COUNT(*) AS n
        FROM maps m JOIN matches mt ON mt.id = m.match_id
        WHERE mt.match_date >= ? AND m.map_name IS NOT NULL
        GROUP BY m.map_name HAVING COUNT(*) >= ?
        """,
        (since, POOL_MIN_PLAYS),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT map_name, COUNT(*) AS n FROM maps WHERE map_name IS NOT NULL GROUP BY map_name"
        ).fetchall()
    total = sum(r["n"] for r in rows)
    return {r["map_name"]: r["n"] / total for r in rows} if total else {}


def _map_probs(
    state: ReplayState,
    model: lgb.Booster,
    team1_id: int,
    team2_id: int,
    map_names: list[str],
    pick_flags: list[int],
    best_of: int,
    ctx: dict,
    as_of: dt.date,
) -> list[float]:
    """P(team1 wins each map). Evaluated in both team orders and averaged, so
    the result is exactly antisymmetric (P(A beats B) == 1 - P(B beats A)).
    `ctx` carries the match context: is_international, event_tier, stakes."""
    h2h_rate, h2h_n = state.h2h.win_rate(team1_id, team2_id)
    forward, mirrored = [], []
    for map_name, flag in zip(map_names, pick_flags):
        s1 = current_team_snapshot(state, team1_id, map_name, as_of)
        s2 = current_team_snapshot(state, team2_id, map_name, as_of)
        common = dict(
            best_of=best_of, map_name=map_name, h2h_n=h2h_n,
            map_atk_bias=state.side.map_atk_bias(map_name), **ctx,
        )
        forward.append(
            snapshot_pair_to_model_row(s1, s2, h2h_team1_rate=h2h_rate, team1_pick=flag, **common)
        )
        mirrored.append(
            snapshot_pair_to_model_row(
                s2, s1, h2h_team1_rate=None if h2h_rate is None else 1.0 - h2h_rate, team1_pick=-flag, **common
            )
        )
    p_fwd = predict_proba(model, pd.DataFrame(forward)).to_numpy()
    p_mir = predict_proba(model, pd.DataFrame(mirrored)).to_numpy()
    return list((p_fwd + (1.0 - p_mir)) / 2.0)


def _team_context(state: ReplayState, team_id: int, as_of: dt.date) -> dict:
    """Headline inputs for display: rating, recent form and roster situation."""
    snap = current_team_snapshot(state, team_id, None, as_of)
    return {
        "elo": round(snap["elo"]),
        "form_10": snap["form_10"],
        "rest_days": snap["rest_days"],
        "roster_continuity": snap["roster_continuity"],
        "days_since_lineup_change": snap["roster_stability_days"],
        # round win rate vs an average team on that side, in percentage points
        "atk_edge_pts": round(snap["atk_edge"] * 100, 1),
        "def_edge_pts": round(snap["def_edge"] * 100, 1),
    }


def _resolve_team(conn: sqlite3.Connection, name: str) -> int:
    team_id = find_team_id_by_name(conn, name)
    if team_id is None:
        raise TeamNotFoundError(f"no team found matching {name!r} (not in the database yet?)")
    return team_id


def _pick_flags(maps: list[str], picked_by: dict[str, int]) -> list[int]:
    """+1 if team 1 picked the map, -1 if team 2 did, 0 for the decider."""
    return [{1: 1, 2: -1}.get(picked_by.get(m, 0), 0) for m in maps]


def predict_match(
    conn: sqlite3.Connection,
    model: lgb.Booster,
    team1_name: str,
    team2_name: str,
    best_of: int = 3,
    maps: list[str] | None = None,
    picks: dict[str, str] | None = None,
    is_international: bool | None = None,
    config: dict | None = None,
    state: ReplayState | None = None,
    event_tier: int | None = None,
    stakes: int | None = None,
) -> dict:
    cfg = config or load_config()
    team1_id = _resolve_team(conn, team1_name)
    team2_id = _resolve_team(conn, team2_name)
    tau = float(cfg.get("model", {}).get("series_strength_sd", 0.0))

    state = state or replay(conn, cfg)
    today = dt.date.today()
    picks = picks or {}
    pool = None
    scenarios = None
    if event_tier is None and is_international:
        event_tier = 2
    ctx = dict(is_international=is_international, event_tier=event_tier, stakes=stakes)

    if maps:
        picked_by = {}
        for map_name in maps:
            picker = (picks.get(map_name) or "").strip().lower()
            if picker == team1_name.strip().lower():
                picked_by[map_name] = 1
            elif picker == team2_name.strip().lower():
                picked_by[map_name] = 2
        map_probs = _map_probs(
            state, model, team1_id, team2_id, maps, _pick_flags(maps, picked_by), best_of, ctx, today
        )
        # a Bo3 veto can list fewer maps than slots only if the series can't go the distance
        while len(map_probs) < best_of:
            map_probs.append(map_probs[-1])
        dist = shared_score_distribution(map_probs, best_of, tau)
        mode = "post-veto"
    else:
        weights = trim_pool(active_map_pool(conn, today))
        if not weights:
            raise ValueError("no maps in the database to build a pre-veto estimate from")
        pool_maps = list(weights)
        neutral = _map_probs(state, model, team1_id, team2_id, pool_maps, [0] * len(pool_maps), best_of, ctx, today)
        by_map = dict(zip(pool_maps, neutral))
        pool = {m: (weights[m], by_map[m]) for m in pool_maps}

        # play out the veto for both possible first-bans and average the two series
        scenarios, dists = [], []
        for first_team in (1, 2):
            order, picked_by = simulate_veto(by_map, best_of, first_team)
            probs = _map_probs(
                state, model, team1_id, team2_id, order, _pick_flags(order, picked_by), best_of, ctx, today
            )
            while len(probs) < best_of:
                probs.append(probs[-1])
            dists.append(shared_score_distribution(probs, best_of, tau))
            scenarios.append({
                "first_ban": first_team, "maps": order, "picked_by": [picked_by.get(m, 0) for m in order],
                "map_probs": probs,
            })
        dist = {k: (dists[0].get(k, 0.0) + dists[1].get(k, 0.0)) / 2 for k in set(dists[0]) | set(dists[1])}
        maps, map_probs = scenarios[0]["maps"], scenarios[0]["map_probs"]
        mode = "pre-veto (simulated veto)"

    needed = best_of // 2 + 1
    team1_win_prob = sum(p for (a, b), p in dist.items() if a == needed)

    return {
        "team1": team1_name,
        "team2": team2_name,
        "team1_id": team1_id,
        "team2_id": team2_id,
        "best_of": best_of,
        "mode": mode,
        "maps": maps,
        "map_probs": map_probs,
        "pool": pool,
        "scenarios": scenarios,
        "series_strength_sd": tau,
        "context": {
            "team1": _team_context(state, team1_id, today),
            "team2": _team_context(state, team2_id, today),
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
    out["map_probs"] = [float(p) for p in result["map_probs"]]
    if result.get("pool"):
        out["pool"] = {m: {"weight": float(w), "p_team1": float(p)} for m, (w, p) in result["pool"].items()}
    if result.get("scenarios"):
        out["scenarios"] = [{**s, "map_probs": [float(p) for p in s["map_probs"]]} for s in result["scenarios"]]
    return out


def format_prediction(result: dict) -> str:
    t1, t2 = result["team1"], result["team2"]
    lines = [
        f"{t1} vs {t2}  (Bo{result['best_of']}, {result['mode']})",
        "",
        f"  {t1}: {result['team1_win_prob']:.1%}",
        f"  {t2}: {result['team2_win_prob']:.1%}",
        "",
        "Team context:",
    ]
    for key in ("team1", "team2"):
        c = result["context"][key]
        form = "n/a" if c["form_10"] is None else f"{c['form_10']:.0%}"
        cont = "n/a" if c["roster_continuity"] is None else f"{c['roster_continuity']:.0%}"
        since = "n/a" if c["days_since_lineup_change"] is None else f"{c['days_since_lineup_change']}d"
        lines.append(
            f"  {result[key]:16s} Elo {c['elo']} · last-10 maps {form} · lineup continuity {cont} · same lineup {since}"
            f" · attack {c['atk_edge_pts']:+.1f} / defence {c['def_edge_pts']:+.1f} pts"
        )
    lines.append("")

    if result.get("scenarios"):
        lines.append("Simulated veto (each team bans its worst maps, picks its best):")
        for s in result["scenarios"]:
            first = result["team1"] if s["first_ban"] == 1 else result["team2"]
            parts = []
            for m, picker, p in zip(s["maps"], s["picked_by"], s["map_probs"]):
                who = {1: f"{t1} pick", 2: f"{t2} pick"}.get(picker, "decider")
                parts.append(f"{m} ({who}, {t1} {p:.0%})")
            lines.append(f"  {first} bans first: " + " | ".join(parts))
    elif result.get("pool"):
        lines.append(f"Map pool (P({t1} wins), weight):")
        for m, (w, p) in sorted(result["pool"].items(), key=lambda kv: -kv[1][0]):
            lines.append(f"  {m:12s} {p:6.1%}   (weight {w:.0%})")
    else:
        lines.append(f"Map probabilities (P({t1} wins)):")
        for map_name, p in zip(result["maps"], result["map_probs"]):
            lines.append(f"  {map_name:20s} {p:.1%}")
    lines.append("")

    if result.get("p_distance") is not None:
        lines.append(f"Goes to a deciding map (a {result['best_of'] // 2 + 1}-{result['best_of'] // 2} series): "
                     f"{result['p_distance']:.0%}")
    lines.append("Score distribution:")
    for (a, b), p in sorted(result["score_distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {t1} {a}-{b} {t2}: {p:.1%}")
    return "\n".join(lines)
