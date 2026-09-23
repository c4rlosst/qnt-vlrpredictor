"""Tests the parser's traversal/typing logic against hand-written fixture
HTML (see fixtures/README.md) — NOT verified against real hltv.org markup,
since this dev environment can't reach the site. Treat a pass here as "the
code does what it's supposed to with well-formed input shaped like the
documented HLTV structure", not "this will work against the live site".
"""

from pathlib import Path

from cspredictor.scraping.parsers import parse_match_detail, parse_rankings_page, parse_results_page
from cspredictor.scraping.results import parse_date_label

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_results_page():
    html = (FIXTURES / "results_page.html").read_text(encoding="utf-8")
    rows = parse_results_page(html)
    assert len(rows) == 3

    row1 = rows[0]
    assert row1.hltv_match_id == 2500001
    assert row1.team1_name == "Team Alpha"
    assert row1.team2_name == "Team Beta"
    assert row1.team1_score == 2
    assert row1.team2_score == 0
    assert row1.date_label == "Results for 22nd September 2026"
    assert row1.event_name == "Some Event"

    row2 = rows[1]
    assert row2.hltv_match_id == 2500002
    assert row2.team1_score == 1
    assert row2.team2_score == 16
    assert row2.bo1_map == "Mirage"

    row3 = rows[2]
    assert row3.date_label == "Results for 21st September 2026"
    assert row3.team1_name == "Team Epsilon"


def test_parse_date_label():
    import datetime as dt

    assert parse_date_label("Results for 22nd September 2026") == dt.date(2026, 9, 22)
    assert parse_date_label("Results for 21st September 2026") == dt.date(2026, 9, 21)
    assert parse_date_label("Results for today") == dt.date.today()
    assert parse_date_label(None) is None
    assert parse_date_label("garbage") is None


def test_parse_match_detail():
    html = (FIXTURES / "match_detail.html").read_text(encoding="utf-8")
    detail = parse_match_detail(html, hltv_match_id=2500001)

    assert detail.team1_name == "Team Alpha"
    assert detail.team2_name == "Team Beta"
    assert detail.unix_timestamp_ms == 1758536400000
    assert detail.event_name == "Some Event"
    assert detail.event_hltv_id == 7001
    assert detail.best_of == 3
    assert detail.is_lan is True

    assert len(detail.maps) == 3
    m1, m2, m3 = detail.maps
    assert m1.map_name == "Mirage"
    assert m1.team1_score == 13
    assert m1.team2_score == 8
    assert m1.picked_by == "Team Alpha"

    assert m2.map_name == "Inferno"
    assert m2.picked_by == "Team Beta"

    assert m3.map_name == "Anubis"
    assert m3.picked_by is None  # decider / leftover map, not picked by either team

    assert detail.team1_lineup == ["alphaOne", "alphaTwo", "alphaThree", "alphaFour", "alphaFive"]
    assert detail.team2_lineup == ["betaOne", "betaTwo", "betaThree", "betaFour", "betaFive"]


def test_parse_rankings_page():
    html = (FIXTURES / "rankings_page.html").read_text(encoding="utf-8")
    rows = parse_rankings_page(html)
    assert len(rows) == 2
    assert rows[0].rank == 1
    assert rows[0].team_name == "Team Alpha"
    assert rows[0].team_hltv_id == 1001
    assert rows[0].points == 650
    assert rows[1].rank == 2
    assert rows[1].team_hltv_id == 1002


def test_parsers_fail_soft_on_missing_fields():
    html = "<html><body><div class='result-con'></div></body></html>"
    rows = parse_results_page(html)
    assert rows == []  # no link -> skipped, not a crash

    empty_detail = parse_match_detail("<html><body></body></html>", hltv_match_id=999)
    assert empty_detail.team1_name is None
    assert empty_detail.maps == []
