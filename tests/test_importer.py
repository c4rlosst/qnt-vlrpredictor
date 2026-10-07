from pathlib import Path

import pytest

from valpredictor.features.build_features import build_map_table
from valpredictor.importer import import_csv, parse_csv
from valpredictor.storage import db

HEADER = "match_id,date,team1,team2,best_of,event,map_order,map,score1,score2\n"
GOOD = HEADER + (
    "9000001,2026-10-07,NRG,T1,3,Valorant Champions 2026,1,Lotus,13,11\n"
    "9000001,2026-10-07,NRG,T1,3,Valorant Champions 2026,2,Summit,13,7\n"
    "9000002,2026-10-08,LOUD,Paper Rex,3,Valorant Champions 2026,1,Haven,9,13\n"
    "9000002,2026-10-08,LOUD,Paper Rex,3,Valorant Champions 2026,2,Ascent,13,10\n"
    "9000002,2026-10-08,LOUD,Paper Rex,3,Valorant Champions 2026,3,Bind,13,11\n"
)


@pytest.fixture
def conn(tmp_path):
    return db.get_connection(tmp_path / "test.db")


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "maps.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_import_populates_matches_and_maps(conn, tmp_path):
    result = import_csv(conn, _write(tmp_path, GOOD))
    assert result.ok and result.matches_imported == 2 and result.maps_imported == 5
    assert sorted(result.new_teams) == ["LOUD", "NRG", "Paper Rex", "T1"]

    m1 = conn.execute("SELECT * FROM matches WHERE vlr_id = 9000001").fetchone()
    assert (m1["team1_score"], m1["team2_score"], m1["best_of"], m1["match_date"]) == (2, 0, 3, "2026-10-07")
    assert conn.execute("SELECT name FROM teams WHERE id = ?", (m1["winner_team_id"],)).fetchone()["name"] == "NRG"

    maps = conn.execute(
        "SELECT map_name FROM maps WHERE match_id = ? ORDER BY map_order", (m1["id"],)
    ).fetchall()
    assert [r["map_name"] for r in maps] == ["Lotus", "Summit"]

    m2 = conn.execute("SELECT id FROM matches WHERE vlr_id = 9000002").fetchone()
    assert conn.execute("SELECT COUNT(*) c FROM maps WHERE match_id = ?", (m2["id"],)).fetchone()["c"] == 3


def test_imported_data_flows_into_the_feature_table(conn, tmp_path):
    import_csv(conn, _write(tmp_path, GOOD))
    df = build_map_table(conn)
    assert len(df) == 5
    assert set(df["map_name"]) == {"Lotus", "Summit", "Haven", "Ascent", "Bind"}


def test_dry_run_writes_nothing(conn, tmp_path):
    result = import_csv(conn, _write(tmp_path, GOOD), dry_run=True)
    assert result.ok and result.matches_imported == 2
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 0
    assert conn.execute("SELECT COUNT(*) c FROM teams").fetchone()["c"] == 0


def test_reimport_skips_existing_and_replace_overwrites(conn, tmp_path):
    path = _write(tmp_path, GOOD)
    import_csv(conn, path)
    again = import_csv(conn, path)
    assert again.matches_imported == 0 and sorted(again.matches_skipped) == [9000001, 9000002]

    # correct match 1 so T1 (not NRG) won 2-0
    fixed = GOOD.replace("Lotus,13,11", "Lotus,11,13").replace("Summit,13,7", "Summit,7,13")
    replaced = import_csv(conn, _write(tmp_path, fixed), replace=True)
    assert replaced.matches_imported == 2
    row = conn.execute("SELECT team1_score, team2_score FROM matches WHERE vlr_id = 9000001").fetchone()
    assert (row["team1_score"], row["team2_score"]) == (0, 2)
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 2  # replaced, not duplicated
    assert conn.execute("SELECT COUNT(*) c FROM maps").fetchone()["c"] == 5


def test_all_problems_reported_and_nothing_imported(conn, tmp_path):
    bad = HEADER + (
        "9000001,2026-13-45,NRG,T1,3,Champions,1,Lotus,13,11\n"          # bad date
        "9000002,2026-10-07,NRG,T1,4,Champions,1,Lotus,13,11\n"          # bad best_of
        "9000003,2026-10-07,NRG,T1,3,Champions,1,Lotus,13,13\n"          # tie
        "9000005,2026-10-07,NRG,T1,3,Champions,1,Lotus,13,11\n"          # series incomplete (1-0 in Bo3)
        "9000006,2026-10-07,NRG,NRG,1,Champions,1,Lotus,13,11\n"         # same team
    )
    result = import_csv(conn, _write(tmp_path, bad))
    text = "\n".join(result.errors)
    assert not result.ok and result.matches_imported == 0
    for needle in ("date must look like", "best_of must be 1, 3 or 5", "cannot end in a tie",
                   "series incomplete", "same team"):
        assert needle in text, needle
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 0


def test_missing_column_and_inconsistent_match(conn, tmp_path):
    result = import_csv(conn, _write(tmp_path, "match_id,date\n1,2026-01-01\n"))
    assert "missing required column(s)" in result.errors[0]

    mixed = HEADER + (
        "9000001,2026-10-07,NRG,T1,3,Champions,1,Lotus,13,11\n"
        "9000001,2026-10-07,NRG,Sentinels,3,Champions,2,Summit,13,7\n"
    )
    assert any("team2" in e and "line 2" in e for e in import_csv(conn, _write(tmp_path, mixed)).errors)


def test_too_many_maps_or_map_after_series_decided(conn, tmp_path):
    text = HEADER + (
        "1,2026-10-07,A,B,3,Cup,1,Lotus,13,5\n"
        "1,2026-10-07,A,B,3,Cup,2,Bind,13,5\n"
        "1,2026-10-07,A,B,3,Cup,3,Haven,13,5\n"
    )
    assert any("already decided" in e for e in import_csv(conn, _write(tmp_path, text)).errors)


def test_similar_team_name_warns_about_typo(conn, tmp_path):
    db.upsert_team(conn, "Paper Rex", vlr_id=5)
    text = HEADER + "1,2026-10-07,Paper Rexx,LOUD,1,Cup,1,Lotus,13,5\n"
    result = import_csv(conn, _write(tmp_path, text), dry_run=True)
    assert result.ok and any("Paper Rexx" in w and "Paper Rex" in w for w in result.warnings)


def test_existing_team_is_reused_case_insensitively(conn, tmp_path):
    existing = db.upsert_team(conn, "Paper Rex", vlr_id=5)
    import_csv(conn, _write(tmp_path, HEADER + "1,2026-10-07,paper rex,LOUD,1,Cup,1,Lotus,13,5\n"))
    row = conn.execute("SELECT team1_id FROM matches WHERE vlr_id = 1").fetchone()
    assert row["team1_id"] == existing
    assert conn.execute("SELECT COUNT(*) c FROM teams WHERE name LIKE 'paper rex'").fetchone()["c"] == 1


def test_excel_bom_is_tolerated(conn, tmp_path):
    p = tmp_path / "bom.csv"
    p.write_bytes(b"\xef\xbb\xbf" + GOOD.encode("utf-8"))
    matches, errors = parse_csv(p)
    assert not errors and len(matches) == 2
