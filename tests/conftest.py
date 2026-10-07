import datetime as dt
import random

from valpredictor.storage import db

LEAGUE_MAPS = ["Ascent", "Bind", "Haven", "Lotus"]


def make_league_db(path, n_matches: int = 120, seed: int = 7):
    """A small simulated league (6 teams with fixed skills) in a fresh SQLite DB."""
    conn = db.get_connection(path)
    rng = random.Random(seed)
    skills = [1.2, 0.6, 0.0, -0.6, -1.2, 0.3]
    teams = [db.upsert_team(conn, f"Team {i}", vlr_id=100 + i) for i in range(len(skills))]
    start = dt.date.today() - dt.timedelta(days=90)
    for n in range(n_matches):
        a, b = rng.sample(range(len(teams)), 2)
        day = start + dt.timedelta(days=n * 3 // 4)
        mid = db.upsert_match(
            conn, vlr_id=n + 1, match_url=f"/{n + 1}/x", event_id=None,
            unix_timestamp_ms=int(dt.datetime.combine(day, dt.time(12), tzinfo=dt.timezone.utc).timestamp() * 1000),
            team1_id=teams[a], team2_id=teams[b], best_of=3, team1_score=None, team2_score=None,
            is_international=False,
        )
        rows, wa, wb = [], 0, 0
        for order, name in enumerate(rng.sample(LEAGUE_MAPS, 3), start=1):
            if 2 in (wa, wb):
                break
            a_wins = rng.random() < 1 / (1 + 2.718 ** -(skills[a] - skills[b]))
            rows.append({"map_order": order, "map_name": name, "team1_score": 13 if a_wins else 8,
                         "team2_score": 8 if a_wins else 13, "team1_id": teams[a], "team2_id": teams[b],
                         "picked_by_team_id": teams[a] if order == 1 else None})
            wa, wb = wa + a_wins, wb + (not a_wins)
        db.replace_maps(conn, mid, rows)
    conn.commit()
    return conn
