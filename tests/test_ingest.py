"""Tests the scrape->parse->upsert wiring in ingest.py using a fake client
that serves the fixture HTML instead of hitting the network."""

from pathlib import Path

import pytest

from cspredictor.scraping import ingest
from cspredictor.storage import db

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient:
    """Stands in for HLTVClient: same `.get(path)` interface, no network."""

    def __init__(self, html: str):
        self._html = html
        self.calls = []

    def get(self, path_or_url: str, force_refresh: bool = False) -> str:
        self.calls.append(path_or_url)
        return self._html


@pytest.fixture
def conn(tmp_path):
    return db.get_connection(tmp_path / "test.db")


def test_ingest_match_populates_all_tables(conn):
    html = (FIXTURES / "match_detail.html").read_text(encoding="utf-8")
    client = FakeClient(html)

    match_id = ingest.ingest_match(conn, client, hltv_match_id=2500001, match_url="/matches/2500001/a-vs-b")
    assert match_id is not None
    assert client.calls == ["/matches/2500001/a-vs-b"]

    match_row = conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
    assert match_row["hltv_id"] == 2500001
    assert match_row["best_of"] == 3
    assert match_row["team1_score"] == 2  # Team Alpha won 2 maps (Mirage, Anubis)
    assert match_row["team2_score"] == 1
    assert match_row["is_lan"] == 1

    team1 = conn.execute("SELECT * FROM teams WHERE id = ?", (match_row["team1_id"],)).fetchone()
    team2 = conn.execute("SELECT * FROM teams WHERE id = ?", (match_row["team2_id"],)).fetchone()
    assert team1["name"] == "Team Alpha"
    assert team2["name"] == "Team Beta"
    assert match_row["winner_team_id"] == team1["id"]

    event_row = conn.execute("SELECT * FROM events WHERE id = ?", (match_row["event_id"],)).fetchone()
    assert event_row["name"] == "Some Event"
    assert event_row["hltv_id"] == 7001

    maps = conn.execute(
        "SELECT * FROM maps WHERE match_id = ? ORDER BY map_order", (match_id,)
    ).fetchall()
    assert len(maps) == 3
    assert [m["map_name"] for m in maps] == ["Mirage", "Inferno", "Anubis"]
    assert maps[0]["picked_by_team_id"] == team1["id"]
    assert maps[1]["picked_by_team_id"] == team2["id"]
    assert maps[2]["picked_by_team_id"] is None
    assert maps[0]["winner_team_id"] == team1["id"]
    assert maps[1]["winner_team_id"] == team2["id"]

    rosters = conn.execute(
        "SELECT r.team_id, p.name FROM rosters r JOIN players p ON p.id = r.player_id WHERE r.match_id = ?",
        (match_id,),
    ).fetchall()
    team1_names = sorted(r["name"] for r in rosters if r["team_id"] == team1["id"])
    assert team1_names == ["alphaFive", "alphaFour", "alphaOne", "alphaThree", "alphaTwo"]


def test_ingest_match_is_idempotent(conn):
    html = (FIXTURES / "match_detail.html").read_text(encoding="utf-8")
    client = FakeClient(html)

    first_id = ingest.ingest_match(conn, client, hltv_match_id=2500001, match_url="/matches/2500001/a-vs-b")
    second_id = ingest.ingest_match(conn, client, hltv_match_id=2500001, match_url="/matches/2500001/a-vs-b")

    assert first_id == second_id
    assert client.calls == ["/matches/2500001/a-vs-b"]  # second call skipped the fetch entirely
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 1
