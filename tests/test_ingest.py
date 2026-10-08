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
    assert [m["winner_team_id"] for m in maps] == [team1["id"], team1["id"]]

    roster = conn.execute(
        "SELECT p.name FROM rosters r JOIN players p ON p.id = r.player_id WHERE r.match_id = ? AND r.team_id = ?",
        (match_id, team1["id"]),
    ).fetchall()
    assert sorted(r["name"] for r in roster) == ["Ethan", "brawk", "keiko", "mada", "skuba"]


def test_ingest_retries_live_when_the_cache_predates_the_match_finishing(conn):
    """A match page cached while the match was still live/upcoming must not
    block ingestion forever once the match has actually finished: the first
    (cached) fetch comes back non-final, so ingest_match must force a live
    re-fetch rather than concluding the match is unfinished."""
    final_html = (FIXTURES / "match_detail.html").read_text(encoding="utf-8")
    stale_html = re.sub(r"\bfinal\b", "live", final_html, count=1)

    class StaleThenFreshClient:
        def __init__(self):
            self.calls: list[tuple[str, bool]] = []

        def get(self, path_or_url: str, force_refresh: bool = False) -> str:
            self.calls.append((path_or_url, force_refresh))
            return final_html if force_refresh else stale_html

    client = StaleThenFreshClient()
    match_id = ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)

    assert match_id is not None
    assert client.calls == [(MATCH_URL, False), (MATCH_URL, True)]  # cached first, then forced live
    row = conn.execute("SELECT team1_score, team2_score FROM matches WHERE id = ?", (match_id,)).fetchone()
    assert (row["team1_score"], row["team2_score"]) == (2, 0)


def test_ingest_does_not_retry_when_the_cache_is_already_final(conn, client):
    """The common case (page cached after the match finished) must not pay for
    an extra live request."""
    ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL)
    assert client.calls == [MATCH_URL]  # one fetch only


def test_ingest_still_skips_a_match_thats_genuinely_unfinished(conn):
    live_html = re.sub(r"\bfinal\b", "live", (FIXTURES / "match_detail.html").read_text(encoding="utf-8"), count=1)

    class AlwaysLiveClient:
        def __init__(self):
            self.calls = []

        def get(self, path_or_url: str, force_refresh: bool = False) -> str:
            self.calls.append((path_or_url, force_refresh))
            return live_html

    client = AlwaysLiveClient()
    assert ingest.ingest_match(conn, client, vlr_match_id=754732, match_url=MATCH_URL) is None
    assert client.calls == [(MATCH_URL, False), (MATCH_URL, True)]  # retried once, still not final
    assert conn.execute("SELECT COUNT(*) c FROM matches").fetchone()["c"] == 0


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
