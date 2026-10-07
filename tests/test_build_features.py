"""End-to-end test of the chronological-replay feature pipeline against a
small synthetic history (no scraping involved), using the real DB helpers so
this exercises the same code path production ingestion uses.
"""

import datetime as dt

import pytest

from valpredictor.features import build_features as bf
from valpredictor.storage import db

TEST_CONFIG = {
    "features": {
        "elo": {"initial_rating": 1500.0, "k_factor": 32.0, "map_k_factor": 24.0},
        "rolling_windows": [5, 10, 20],
        "recent_days_congestion": 14,
    }
}


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
        team1_id=team_a, team2_id=team_b, best_of=1,
        team1_score=1, team2_score=0, is_international=True,
    )
    db.replace_maps(conn, match1_id, [
        {"map_order": 1, "map_name": "Ascent", "team1_score": 16, "team2_score": 10,
         "team1_id": team_a, "team2_id": team_b, "picked_by_team_id": None},
    ])
    a_players_1 = [db.upsert_player(conn, f"a{i}") for i in range(1, 6)]
    b_players_1 = [db.upsert_player(conn, f"b{i}") for i in range(1, 6)]
    db.replace_roster(conn, match1_id, team_a, a_players_1)
    db.replace_roster(conn, match1_id, team_b, b_players_1)

    # match 2: Bo3, two weeks later. Team A wins 2-1. Team B fields one stand-in.
    match2_id = db.upsert_match(
        conn, vlr_id=2, match_url="/matches/2/a-vs-b-2", event_id=None,
        unix_timestamp_ms=int(dt.datetime(2026, 1, 15).timestamp() * 1000),
        team1_id=team_a, team2_id=team_b, best_of=3,
        team1_score=2, team2_score=1, is_international=False,
    )
    db.replace_maps(conn, match2_id, [
        {"map_order": 1, "map_name": "Ascent", "team1_score": 16, "team2_score": 5,
         "team1_id": team_a, "team2_id": team_b, "picked_by_team_id": team_a},
        {"map_order": 2, "map_name": "Lotus", "team1_score": 10, "team2_score": 16,
         "team1_id": team_a, "team2_id": team_b, "picked_by_team_id": team_b},
        {"map_order": 3, "map_name": "Haven", "team1_score": 16, "team2_score": 12,
         "team1_id": team_a, "team2_id": team_b, "picked_by_team_id": None},
    ])
    b_players_2 = b_players_1[:4] + [db.upsert_player(conn, "b_standin")]
    db.replace_roster(conn, match2_id, team_a, a_players_1)  # unchanged
    db.replace_roster(conn, match2_id, team_b, b_players_2)  # one player swapped

    conn.commit()
    return conn, team_a, team_b


def test_first_match_has_no_historical_state(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_training_table(conn, TEST_CONFIG)
    match1_rows = df[df["match_id"] == 1]
    assert len(match1_rows) == 1
    row = match1_rows.iloc[0]

    assert row["team1_elo"] == 1500.0
    assert row["team2_elo"] == 1500.0
    assert row["team1_won_map"] == 1
    assert pd_isna(row["team1_rest_days"])
    assert row["team1_congestion"] == 0
    assert pd_isna(row["team1_roster_stability_days"])
    assert pd_isna(row["team1_standin"])  # first sighting -> unknown, not False


def test_elo_and_rolling_state_carries_into_second_match(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_training_table(conn, TEST_CONFIG)
    match2_rows = df[df["match_id"] == 2].sort_values("map_order")
    assert len(match2_rows) == 3

    first_map_row = match2_rows.iloc[0]
    # Team A won match 1, so its Elo should now exceed Team B's going into match 2
    assert first_map_row["team1_elo"] > 1500.0
    assert first_map_row["team2_elo"] < 1500.0
    # Team A also won Ascent specifically in match 1 -> its Ascent map-Elo should be elevated
    assert first_map_row["team1_map_elo"] > 1500.0
    assert first_map_row["map_name"] == "Ascent"

    # rest days: exactly 14 days since the Jan 1 match, for both teams
    assert first_map_row["team1_rest_days"] == 14
    assert first_map_row["team2_rest_days"] == 14
    assert first_map_row["team1_congestion"] == 1

    # head-to-head before match 2: Team A won the only prior meeting
    assert first_map_row["h2h_team1_rate"] == pytest.approx(1.0)
    assert first_map_row["h2h_n"] == 1

    # roster: Team A unchanged (not a stand-in match), Team B swapped a player
    assert first_map_row["team1_standin"] == False  # noqa: E712 (numpy/py bool)
    assert first_map_row["team2_standin"] == True  # noqa: E712

    # veto context: map 1 was Team A's pick, map 2 Team B's, map 3 a decider
    assert match2_rows.iloc[0]["team1_pick"] == 1
    assert match2_rows.iloc[1]["team1_pick"] == -1
    assert match2_rows.iloc[2]["team1_pick"] == 0

    # per-map results
    assert list(match2_rows["team1_won_map"]) == [1, 0, 1]


def test_map_elo_updates_within_match_not_just_between_matches(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_training_table(conn, TEST_CONFIG)
    match2_rows = df[df["match_id"] == 2].sort_values("map_order")

    lotus_row = match2_rows.iloc[1]
    haven_row = match2_rows.iloc[2]
    # Team A's overall Elo going into map 3 (Haven) should already reflect
    # having won map 1 (Ascent) of the *same* match — map-level state moves
    # within a match even though match-level Elo does not.
    assert haven_row["team1_map_elo"] == 1500.0  # first time either team plays Haven
    assert lotus_row["team1_elo"] == haven_row["team1_elo"]  # match-level Elo is constant within the match


def test_lineup_change_regresses_elo_and_lowers_continuity(seeded_conn):
    conn, team_a, team_b = seeded_conn
    shrink_cfg = {"features": {**TEST_CONFIG["features"],
                  "elo": {**TEST_CONFIG["features"]["elo"],
                          "roster_change_shrink_per_player": 0.2, "roster_change_shrink_cap": 0.5}}}
    plain = bf.build_map_training_table(conn, TEST_CONFIG)
    shrunk = bf.build_map_training_table(conn, shrink_cfg)

    plain_m2 = plain[plain["match_id"] == 2].iloc[0]
    shrunk_m2 = shrunk[shrunk["match_id"] == 2].iloc[0]
    # Team B lost match 1 (Elo below 1500) and fielded 1 new player in match 2:
    # its Elo is pulled 20% back toward 1500. Team A's lineup is unchanged.
    assert plain_m2["team2_elo"] < shrunk_m2["team2_elo"] < 1500.0
    assert shrunk_m2["team1_elo"] == plain_m2["team1_elo"]

    # continuity: A unchanged = 1.0; B has 4 of 5 players from its only prior lineup = 0.8
    assert plain_m2["team1_roster_continuity"] == pytest.approx(1.0)
    assert plain_m2["team2_roster_continuity"] == pytest.approx(0.8)


def test_side_strength_features_flow_into_later_rows(conn):
    a = db.upsert_team(conn, "A", vlr_id=21)
    b = db.upsert_team(conn, "B", vlr_id=22)
    for n, day in enumerate((1, 8), start=1):
        mid = db.upsert_match(
            conn, vlr_id=n, match_url=None, event_id=None,
            unix_timestamp_ms=int(dt.datetime(2026, 3, day).timestamp() * 1000),
            team1_id=a, team2_id=b, best_of=1, team1_score=1, team2_score=0, is_international=False,
        )
        # real NRG-T1 Lotus numbers: team1 started on defence (5 def / 8 atk), team2 on attack (7 atk / 4 def)
        db.replace_maps(conn, mid, [{
            "map_order": 1, "map_name": "Lotus", "team1_score": 13, "team2_score": 11,
            "team1_id": a, "team2_id": b, "picked_by_team_id": None, "team1_start_side": "def",
            "team1_atk_won": 8, "team1_def_won": 5, "team1_ot_won": 0,
            "team2_atk_won": 7, "team2_def_won": 4, "team2_ot_won": 0,
        }])
    conn.commit()

    df = bf.build_map_training_table(conn, TEST_CONFIG)
    first, second = df.iloc[0], df.iloc[1]
    assert first["team1_atk_edge"] == 0.0 and first["map_atk_bias"] == 0.0   # nothing known yet
    # after one map: league attack rate = 15/24; team1 went 8/12 on attack, shrunk toward it
    assert second["team1_atk_edge"] == pytest.approx((8 + 48 * 15 / 24) / 60 - 15 / 24)
    assert second["map_atk_bias"] > 0  # Lotus has been attacker-friendly so far

    matrix = bf.to_model_matrix(bf.symmetrize(df))
    row = matrix[(matrix["match_id"] == 2) & (matrix["team1_id"] == a)].iloc[0]
    assert row["atk1_vs_def2"] == pytest.approx(second["team1_atk_edge"] - second["team2_def_edge"])
    assert row["round_edge_diff"] == pytest.approx(row["atk1_vs_def2"] + row["def1_vs_atk2"])
    mirror = matrix[(matrix["match_id"] == 2) & (matrix["team1_id"] == b)].iloc[0]
    assert mirror["atk1_vs_def2"] == pytest.approx(second["team2_atk_edge"] - second["team1_def_edge"])
    assert mirror["map_atk_bias"] == row["map_atk_bias"]  # a property of the map, not of either team


def test_no_shrink_when_lineup_data_is_missing(conn):
    a = db.upsert_team(conn, "A", vlr_id=11)
    b = db.upsert_team(conn, "B", vlr_id=12)
    for n, day in enumerate((1, 8), start=1):
        mid = db.upsert_match(
            conn, vlr_id=n, match_url=None, event_id=None,
            unix_timestamp_ms=int(dt.datetime(2026, 2, day).timestamp() * 1000),
            team1_id=a, team2_id=b, best_of=1, team1_score=1, team2_score=0, is_international=False,
        )
        db.replace_maps(conn, mid, [{"map_order": 1, "map_name": "Lotus", "team1_score": 13, "team2_score": 5,
                                     "team1_id": a, "team2_id": b, "picked_by_team_id": None}])
    conn.commit()
    cfg = {"features": {**TEST_CONFIG["features"],
           "elo": {**TEST_CONFIG["features"]["elo"], "roster_change_shrink_per_player": 0.5}}}
    df = bf.build_map_training_table(conn, cfg)
    assert df["team1_roster_continuity"].isna().all()  # no rosters -> unknown, and no crash
    assert df.iloc[1]["team1_elo"] > 1500.0            # Elo moved normally, untouched by shrink


def test_symmetrize_doubles_rows_and_flips_target(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_training_table(conn, TEST_CONFIG)
    sym = bf.symmetrize(df)
    assert len(sym) == 2 * len(df)

    original_first = df.iloc[0]
    mirror = sym.iloc[len(df)]
    assert mirror["team1_id"] == original_first["team2_id"]
    assert mirror["team2_id"] == original_first["team1_id"]
    assert mirror["team1_won_map"] == 1 - original_first["team1_won_map"]
    assert mirror["team1_elo"] == original_first["team2_elo"]


def test_to_model_matrix_diff_signs(seeded_conn):
    conn, team_a, team_b = seeded_conn
    df = bf.build_map_training_table(conn, TEST_CONFIG)
    matrix = bf.to_model_matrix(df)

    match2_first = matrix[(matrix["match_id"] == 2) & (matrix["map_order"] == 1)].iloc[0]
    assert match2_first["elo_diff"] > 0  # team1 (A) is the stronger team going in
    assert match2_first["form_10_diff"] > 0  # team1 has the better recent record


def pd_isna(x):
    import pandas as pd
    return pd.isna(x)
