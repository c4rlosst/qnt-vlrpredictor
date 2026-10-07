"""Tests the scrape->parse->upsert wiring in ingest.py using a fake client
that serves the real (trimmed) fixture HTML instead of hitting the network."""

import re
from pathlib import Path

import pytest

from valpredictor.scraping import ingest
from valpredictor.storage import db

FIXTURES = Path(__file__).parent / "fixtures"
MATCH_URL = "/754732/nrg-vs-t1-valorant-champions-2026-ubqf"


class FakeClient:
    """Stands in for VLRClient: same `.get(path)` interface, no network."""

    def __init__(self, html: str):
        self._html = html
        self.calls = []

    def get(self, path_or_url: str, force_refresh: bool = False) -> str:
        self.calls.append(path_or_url)
        return self._html


@pytest.fixture
def conn(tmp_path):
    return db.get_connection(tmp_path / "test.db")


@pytest.fixture
def client():
    return FakeClient((FIXTURES / "match_detail.html").read_text(encoding="utf-8"))


def test_ingest_match_populates_all_tables(conn, client):
    match_id = ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)
    assert match_id is not None
    assert client.calls == [MATCH_URL]

    match = conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
    assert match["vlr_id"] == 754732
    assert match["best_of"] == 3
    assert (match["team1_score"], match["team2_score"]) == (2, 0)
    assert match["is_international"] == 1
    assert match["match_date"] == "2026-10-07"

    team1 = conn.execute("SELECT * FROM teams WHERE id = ?", (match["team1_id"],)).fetchone()
    team2 = conn.execute("SELECT * FROM teams WHERE id = ?", (match["team2_id"],)).fetchone()
    assert (team1["name"], team1["vlr_id"]) == ("NRG", 1034)
    assert (team2["name"], team2["vlr_id"]) == ("T1", 14)
    assert match["winner_team_id"] == team1["id"]

    event = conn.execute("SELECT * FROM events WHERE id = ?", (match["event_id"],)).fetchone()
    assert (event["name"], event["vlr_id"]) == ("Valorant Champions 2026", 2766)

    maps = conn.execute("SELECT * FROM maps WHERE match_id = ? ORDER BY map_order", (match_id,)).fetchall()
    assert [m["map_name"] for m in maps] == ["Lotus", "Summit"]
    assert maps[0]["picked_by_team_id"] == team2["id"]  # T1 picked Lotus
    assert maps[1]["picked_by_team_id"] == team1["id"]  # NRG picked Summit
    assert [m["winner_team_id"] for m in maps] == [team1["id"], team1["id"]]

    roster = conn.execute(
        "SELECT p.name FROM rosters r JOIN players p ON p.id = r.player_id WHERE r.match_id = ? AND r.team_id = ?",
        (match_id, team1["id"]),
    ).fetchall()
    assert sorted(r["name"] for r in roster) == ["Ethan", "brawk", "keiko", "mada", "skuba"]


def test_ingest_stores_side_data(conn, client):
    match_id = ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)
    lotus = conn.execute("SELECT * FROM maps WHERE match_id = ? AND map_order = 1", (match_id,)).fetchone()
    assert lotus["team1_start_side"] == "def"
    assert (lotus["team1_atk_won"], lotus["team1_def_won"], lotus["team1_ot_won"]) == (8, 5, 0)
    assert (lotus["team2_atk_won"], lotus["team2_def_won"], lotus["team2_ot_won"]) == (7, 4, 0)


def test_reparse_from_cache_backfills_side_data_without_network(conn, client):
    match_id = ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)
    conn.execute("UPDATE maps SET team1_start_side = NULL, team1_atk_won = NULL, team2_atk_won = NULL")
    conn.commit()

    class CachedClient(FakeClient):
        def has_cached(self, path):
            return True

    cached = CachedClient(client._html)
    updated, missing = ingest.reparse_from_cache(conn, cached)
    assert (updated, missing) == (1, 0)
    row = conn.execute("SELECT team1_start_side, team1_atk_won FROM maps WHERE match_id = ? AND map_order = 1", (match_id,)).fetchone()
    assert (row["team1_start_side"], row["team1_atk_won"]) == ("def", 8)
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 1  # overwritten, not duplicated

    class EmptyCache(FakeClient):
        def has_cached(self, path):
            return False

    assert ingest.reparse_from_cache(conn, EmptyCache(client._html)) == (0, 1)


def test_old_database_gets_side_columns_added(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE maps (id INTEGER PRIMARY KEY, match_id INTEGER, map_order INTEGER, map_name TEXT,"
                " team1_score INTEGER, team2_score INTEGER, winner_team_id INTEGER, picked_by_team_id INTEGER)")
    old.commit()
    old.close()

    conn = db.get_connection(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(maps)")}
    assert {"team1_start_side", "team1_atk_won", "team2_def_won", "team2_ot_won"} <= cols


def test_ingest_match_is_idempotent(conn, client):
    first = ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)
    second = ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)
    assert first == second
    assert client.calls == [MATCH_URL]  # second call never touched the network
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 1


def test_ingest_skips_unfinished_matches(conn):
    html = re.sub(r"\bfinal\b", "upcoming", (FIXTURES / "match_detail.html").read_text(encoding="utf-8"), count=1)
    assert "upcoming" in html
    fake = FakeClient(html)
    assert ingest.ingest_match(conn, fake, vlr_match_id=754732, match_url=MATCH_URL) is None
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 0


def test_event_matches_filter():
    inc = re.compile("Champions Tour|Masters|Valorant Champions", re.I)
    exc = re.compile("Game Changers", re.I)
    assert ingest.event_matches("Valorant Champions 2026", inc, exc)
    assert ingest.event_matches("Champions Tour 2026: Pacific Stage 1", inc, exc)
    assert not ingest.event_matches("Game Changers Championship", inc, exc)
    assert not ingest.event_matches("Some Local Cup", inc, exc)
    assert ingest.event_matches("anything", None, None)
