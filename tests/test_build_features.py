"""End-to-end test of the chronological replay against a small synthetic
history (no scraping involved), using the real DB helpers so this exercises the
same code path production ingestion uses.
"""

import datetime as dt

import pytest

from valpredictor.features import build_features as bf
from valpredictor.storage import db

TEST_CONFIG = {"features": {"elo": {"initial_rating": 1500.0, "k_factor": 32.0}}}


@pytest.fixture
def conn(tmp_path):
    return db.get_connection(tmp_path / "test.db")


@pytest.fixture
def seeded_conn(conn):
    team_a = db.upsert_team(conn, "Team A", vlr_id=1)
    team_b = db.upsert_team(conn, "Team B", vlr_id=2)

    # match 1: Bo1, Team A wins Ascent
    match1_id = db.upsert_match(
        conn, vlr_id=1, match_url="/matches/1/a-vs-b", event_id=None,
        unix_timestamp_ms=int(dt.datetime(2026, 1, 1).timestamp() * 1000),
        team1_id=team_a, team2_id=team_b, best_of=1, team1_score=1, team2_score=0,
    )
    db.replace_maps(conn, match1_id, [
        {"map_order": 1, "map_name": "Ascent", "team1_score": 16, "team2_score": 10,
         "team1_id": team_a, "team2_id": team_b},
    ])
    a_players_1 = [db.upsert_player(conn, f"a{i}") for i in range(1, 6)]
    b_players_1 = [db.upsert_player(conn, f"b{i}") for i in range(1, 6)]
    db.replace_roster(conn, match1_id, team_a, a_players_1)
    db.replace_roster(conn, match1_id, team_b, b_players_1)

    # match 2: Bo3, two weeks later. Team A wins 2-1. Team B fields one stand-in.
    match2_id = db.upsert_match(
        conn, vlr_id=2, match_url="/matches/2/a-vs-b-2", event_id=None,
        unix_timestamp_ms=int(dt.datetime(2026, 1, 15).timestamp() * 1000),
        team1_id=team_a, team2_id=team_b, best_of=3, team1_score=2, team2_score=1,
    )
    db.replace_maps(conn, match2_id, [
        {"map_order": 1, "map_name": "Ascent", "team1_score": 16, "team2_score": 5,
         "team1_id": team_a, "team2_id": team_b},
        {"map_order": 2, "map_name": "Lotus", "team1_score": 10, "team2_score": 16,
         "team1_id": team_a, "team2_id": team_b},
        {"map_order": 3, "map_name": "Haven", "team1_score": 16, "team2_score": 12,
         "team1_id": team_a, "team2_id": team_b},
    ])
    b_players_2 = b_players_1[:4] + [db.upsert_player(conn, "b_standin")]
    db.replace_roster(conn, match2_id, team_a, a_players_1)  # unchanged
    db.replace_roster(conn, match2_id, team_b, b_players_2)  # one player swapped

    conn.commit()
    return conn, team_a, team_b


def test_first_match_is_rated_from_the_starting_elo(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_table(conn, TEST_CONFIG)
    row = df[df["match_id"] == 1].iloc[0]
    assert (row["team1_elo"], row["team2_elo"], row["elo_diff"]) == (1500.0, 1500.0, 0.0)
    assert row["team1_won_map"] == 1


def test_elo_carries_into_the_second_match_and_is_constant_within_it(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_table(conn, TEST_CONFIG)
    match2 = df[df["match_id"] == 2].sort_values("map_order")
    assert list(match2["team1_won_map"]) == [1, 0, 1]
    # Team A won match 1, so it goes into match 2 rated above Team B ...
    assert match2.iloc[0]["team1_elo"] > 1500.0 > match2.iloc[0]["team2_elo"]
    assert match2.iloc[0]["elo_diff"] > 0
    # ... and the rating only changes between series, never between maps of one series
    assert match2["team1_elo"].nunique() == 1


def test_rows_never_see_their_own_result(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_table(conn, TEST_CONFIG)
    # match 1's rating is the pre-match 1500 even though A went on to win it
    assert df[df["match_id"] == 1].iloc[0]["team1_elo"] == 1500.0
    state = bf.replay(conn, TEST_CONFIG)
    assert state.elo.rating(team_a) > df[df["match_id"] == 2].iloc[0]["team1_elo"]  # match 2 moved it again


def test_lineup_change_pulls_elo_back_toward_average(seeded_conn):
    conn, team_a, team_b = seeded_conn
    shrink = {"features": {"elo": {**TEST_CONFIG["features"]["elo"],
                                   "roster_change_shrink_per_player": 0.2, "roster_change_shrink_cap": 0.5}}}
    plain = bf.build_map_table(conn, TEST_CONFIG)
    shrunk = bf.build_map_table(conn, shrink)

    plain_m2 = plain[plain["match_id"] == 2].iloc[0]
    shrunk_m2 = shrunk[shrunk["match_id"] == 2].iloc[0]
    # Team B lost match 1 (Elo below 1500) and fielded 1 new player in match 2:
    # its Elo is pulled 20% back toward 1500. Team A's lineup is unchanged.
    assert plain_m2["team2_elo"] < shrunk_m2["team2_elo"] < 1500.0
    assert shrunk_m2["team1_elo"] == plain_m2["team1_elo"]


def test_snapshot_reports_elo_and_lineup_continuity(seeded_conn):
    conn, team_a, team_b = seeded_conn
    match1_date, match2_date = (
        dt.date.fromisoformat(r[0])
        for r in conn.execute("SELECT match_date FROM matches ORDER BY vlr_id").fetchall()
    )
    state = bf.replay(conn, TEST_CONFIG)
    today = match2_date + dt.timedelta(days=5)
    a = bf.current_team_snapshot(state, team_a, today)
    b = bf.current_team_snapshot(state, team_b, today)
    assert a["elo"] > b["elo"]
    assert a["roster_continuity"] == pytest.approx(1.0)
    assert b["roster_continuity"] == pytest.approx(0.9)  # 4 of 5 players were in lineup 1, all 5 in lineup 2
    assert a["days_since_lineup_change"] == (today - match1_date).days  # unchanged since the first match
    assert b["days_since_lineup_change"] == (today - match2_date).days  # the stand-in arrived in match 2


def test_no_shrink_when_lineup_data_is_missing(conn):
    a = db.upsert_team(conn, "A", vlr_id=11)
    b = db.upsert_team(conn, "B", vlr_id=12)
    for n, day in enumerate((1, 8), start=1):
        mid = db.upsert_match(
            conn, vlr_id=n, match_url=None, event_id=None,
            unix_timestamp_ms=int(dt.datetime(2026, 2, day).timestamp() * 1000),
            team1_id=a, team2_id=b, best_of=1, team1_score=1, team2_score=0,
        )
        db.replace_maps(conn, mid, [{"map_order": 1, "map_name": "Lotus", "team1_score": 13, "team2_score": 5,
                                     "team1_id": a, "team2_id": b}])
    conn.commit()
    cfg = {"features": {"elo": {**TEST_CONFIG["features"]["elo"], "roster_change_shrink_per_player": 0.5}}}
    df = bf.build_map_table(conn, cfg)
    assert df.iloc[1]["team1_elo"] > 1500.0  # Elo moved normally, untouched by any shrink
    assert bf.current_team_snapshot(bf.replay(conn, cfg), a)["roster_continuity"] is None  # unknown, no crash
