"""Parser tests against real vlr.gg markup (trimmed copies of pages fetched on
2026-10-07 during Valorant Champions 2026; see fixtures/README.md)."""

import datetime as dt
from pathlib import Path

from valpredictor.scraping.parsers import parse_match_detail, parse_results_page
from valpredictor.scraping.results import parse_date_label

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_results_page():
    rows = parse_results_page(_read("results_page.html"))
    assert len(rows) == 3

    first = rows[0]
    assert first.vlr_match_id == 754732
    assert first.match_url == "/754732/nrg-vs-t1-valorant-champions-2026-ubqf"
    assert (first.team1_name, first.team2_name) == ("NRG", "T1")
    assert (first.team1_score, first.team2_score) == (2, 0)
    assert first.status == "completed"
    assert first.event_name == "Valorant Champions 2026"
    assert first.series == "Playoffs–Upper Quarterfinals"
    assert first.date_label == "Wed, October 7, 2026"

    # the second date header applies to the second and third rows
    assert rows[1].date_label == rows[2].date_label == "Sun, October 4, 2026"
    assert (rows[1].team1_name, rows[1].team2_name) == ("FUT Esports", "T1")
    assert (rows[1].team1_score, rows[1].team2_score) == (0, 2)


def test_parse_schedule_page_upcoming_live_and_tbd():
    rows = parse_results_page(_read("schedule_page.html"))
    by_id = {r.vlr_match_id: r for r in rows}

    live = by_id[754733]
    assert (live.team1_name, live.team2_name) == ("Paper Rex", "LOUD")
    assert live.status == "live"

    upcoming = by_id[754730]
    assert (upcoming.team1_name, upcoming.team2_name) == ("100 Thieves", "G2 Esports")
    assert upcoming.status == "upcoming"
    assert (upcoming.team1_score, upcoming.team2_score) == (None, None)  # "–" placeholder

    tbd = by_id[754738]
    assert (tbd.team1_name, tbd.team2_name) == ("TBD", "TBD")


def test_parse_date_label():
    assert parse_date_label("Wed, October 7, 2026") == dt.date(2026, 10, 7)
    assert parse_date_label("Sun, October 4, 2026") == dt.date(2026, 10, 4)
    assert parse_date_label(None) is None
    assert parse_date_label("garbage") is None


def test_parse_match_detail_completed_bo3():
    d = parse_match_detail(_read("match_detail.html"), vlr_match_id=754732)

    assert (d.team1_name, d.team1_vlr_id) == ("NRG", 1034)
    assert (d.team2_name, d.team2_vlr_id) == ("T1", 14)
    assert d.event_name == "Valorant Champions 2026"
    assert d.event_vlr_id == 2766
    assert d.best_of == 3
    assert d.status == "final"
    # 2026-10-07 05:00:00 UTC
    assert d.unix_timestamp_ms == int(dt.datetime(2026, 10, 7, 5, tzinfo=dt.timezone.utc).timestamp() * 1000)

    # Abyss "remains" but was never played (NRG won 2-0), so only two maps
    assert [(m.map_name, m.team1_score, m.team2_score) for m in d.maps] == [
        ("Lotus", 13, 11),
        ("Summit", 13, 7),
    ]

    assert [name for name, _ in d.team1_lineup] == ["brawk", "keiko", "mada", "Ethan", "skuba"]
    assert [name for name, _ in d.team2_lineup] == ["BuZz", "Meteor", "iZu", "Munchkin", "stax"]
    assert d.team1_lineup[0] == ("brawk", 2172)


def test_parse_pre_match_and_live_odds():
    d = parse_match_detail(_read("match_with_odds.html"), vlr_match_id=754733)
    assert len(d.odds) == 5
    pre = [o for o in d.odds if o.kind == "pre-match"]
    assert [(o.team1_odds, o.team2_odds) for o in pre] == [(1.69, 2.12)]
    assert all(o.team1_odds > 1 and o.team2_odds > 1 for o in d.odds)


def test_parsers_fail_soft_on_empty_input():
    assert parse_results_page("<html><body></body></html>") == []
    d = parse_match_detail("<html><body></body></html>", vlr_match_id=1)
    assert d.team1_name is None and d.maps == []
