"""Generates a synthetic match history into its own SQLite DB, purely to
smoke-test the feature/model/backtest/predict pipeline end-to-end.

THIS IS NOT REAL VALORANT DATA. It exists so the feature/model/backtest/web-UI
pipeline can be smoke-tested without scraping anything.
It simulates teams with a fixed hidden skill + per-map affinity via a
Bradley-Terry win model, so a correctly-implemented pipeline should be able
to recover signal from it and beat the naive-form / elo-only baselines.

Usage:
    python scripts/generate_synthetic_data.py --out data/synthetic_demo.db
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valpredictor.storage import db  # noqa: E402

MAPS = ["Ascent", "Bind", "Haven", "Lotus", "Split", "Sunset", "Abyss"]


def simulate(out_path: Path, n_teams: int, n_weeks: int, seed: int) -> None:
    rng = random.Random(seed)

    conn = db.get_connection(out_path)

    teams = []
    for i in range(n_teams):
        skill = rng.gauss(0, 1.0)
        map_affinity = {m: rng.gauss(0, 0.4) for m in MAPS}
        players = [f"team{i}_player{j}" for j in range(5)]
        team_id = db.upsert_team(conn, f"Synthetic Team {i}", vlr_id=1000 + i)
        teams.append({"id": team_id, "skill": skill, "affinity": map_affinity, "players": players})

    event_id = db.upsert_event(conn, "Synthetic League", vlr_id=9000, is_international=False)

    start = dt.date.today() - dt.timedelta(weeks=n_weeks)
    match_vlr_id = 1
    for week in range(n_weeks):
        match_date = start + dt.timedelta(weeks=week)

        # a handful of random pairings per week
        pool = teams[:]
        rng.shuffle(pool)
        for a, b in zip(pool[::2], pool[1::2]):
            best_of = rng.choice([1, 3, 3, 3])  # mostly Bo3, some Bo1
            match_vlr_id += 1
            match_id = db.upsert_match(
                conn, vlr_id=match_vlr_id, match_url=f"/matches/{match_vlr_id}/synthetic",
                event_id=event_id, unix_timestamp_ms=int(dt.datetime.combine(match_date, dt.time()).timestamp() * 1000),
                team1_id=a["id"], team2_id=b["id"], best_of=best_of,
                team1_score=None, team2_score=None, is_international=rng.random() < 0.2,
            )

            picked_maps = rng.sample(MAPS, k=best_of)
            map_rows = []
            a_maps_won = b_maps_won = 0
            needed = best_of // 2 + 1
            for order, map_name in enumerate(picked_maps, start=1):
                if a_maps_won == needed or b_maps_won == needed:
                    break
                logit = (a["skill"] - b["skill"]) + (a["affinity"][map_name] - b["affinity"][map_name])
                p_a = 1.0 / (1.0 + math.exp(-logit))
                a_wins = rng.random() < p_a
                winner_score = rng.randint(13, 16)
                loser_score = rng.randint(2, winner_score - 2 if winner_score > 4 else 0)
                s1, s2 = (winner_score, loser_score) if a_wins else (loser_score, winner_score)
                picked_by = a["id"] if order == 1 else (b["id"] if order == 2 else None)
                map_rows.append(
                    {
                        "map_order": order, "map_name": map_name, "team1_score": s1, "team2_score": s2,
                        "team1_id": a["id"], "team2_id": b["id"], "picked_by_team_id": picked_by,
                    }
                )
                a_maps_won += int(a_wins)
                b_maps_won += int(not a_wins)

            db.replace_maps(conn, match_id, map_rows)
            conn.execute(
                "UPDATE matches SET team1_score = ?, team2_score = ?, winner_team_id = ? WHERE id = ?",
                (a_maps_won, b_maps_won, a["id"] if a_maps_won > b_maps_won else b["id"], match_id),
            )

            # occasionally swap one player (stand-in) for a bit of roster-feature signal
            a_roster = a["players"][:4] + [rng.choice(a["players"])] if rng.random() < 0.1 else a["players"]
            b_roster = b["players"][:4] + [rng.choice(b["players"])] if rng.random() < 0.1 else b["players"]
            a_ids = [db.upsert_player(conn, name) for name in a_roster]
            b_ids = [db.upsert_player(conn, name) for name in b_roster]
            db.replace_roster(conn, match_id, a["id"], a_ids)
            db.replace_roster(conn, match_id, b["id"], b_ids)

        conn.commit()

    n_matches = conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"]
    n_maps = conn.execute("SELECT COUNT(*) c FROM maps").fetchone()["c"]
    print(f"wrote {n_matches} matches / {n_maps} maps / {n_teams} teams to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic_demo.db")
    parser.add_argument("--teams", type=int, default=24)
    parser.add_argument("--weeks", type=int, default=78)  # ~1.5 years
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    simulate(Path(args.out), args.teams, args.weeks, args.seed)
