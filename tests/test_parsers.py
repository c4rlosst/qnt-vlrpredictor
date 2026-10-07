"""Parser tests against real vlr.gg markup (trimmed copies of pages fetched on
2026-10-07 during Valorant Champions 2026; see fixtures/README.md)."""

import datetime as dt
from pathlib import Path

from valpredictor.scraping.parsers import (
    is_international_event,
    parse_match_detail,
    parse_results_page,
    parse_veto,
)
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
    assert d.is_international is True
    # 2026-10-07 05:00:00 UTC
    assert d.unix_timestamp_ms == int(dt.datetime(2026, 10, 7, 5, tzinfo=dt.timezone.utc).timestamp() * 1000)

    # Abyss "remains" but was never played (NRG won 2-0), so only two maps
    assert [(m.map_name, m.team1_score, m.team2_score) for m in d.maps] == [
        ("Lotus", 13, 11),
        ("Summit", 13, 7),
    ]
    assert [m.picked_by for m in d.maps] == [2, 1]  # T1 picked Lotus, NRG picked Summit

    assert [(s.team, s.action, s.map_name) for s in d.veto][-1] == (None, "remains", "Abyss")
    assert len(d.veto) == 7

    assert [name for name, _ in d.team1_lineup] == ["brawk", "keiko", "mada", "Ethan", "skuba"]
    assert [name for name, _ in d.team2_lineup] == ["BuZz", "Meteor", "iZu", "Munchkin", "stax"]
    assert d.team1_lineup[0] == ("brawk", 2172)


def test_parse_map_sides_from_real_fixture():
    d = parse_match_detail(_read("match_detail.html"), vlr_match_id=754732)
    lotus, summit = d.maps

    # header "5 / 8" for NRG then "7 / 4" for T1: first number = first-half side
    assert (lotus.team1_sides.first_side, lotus.team1_sides.def_won, lotus.team1_sides.atk_won) == ("def", 5, 8)
    assert (lotus.team2_sides.first_side, lotus.team2_sides.atk_won, lotus.team2_sides.def_won) == ("atk", 7, 4)
    # every first-half round has exactly one winner: 5 (NRG def) + 7 (T1 atk) = 12
    assert lotus.team1_sides.def_won + lotus.team2_sides.atk_won == 12
    assert lotus.team1_sides.atk_won + lotus.team2_sides.def_won == 12

    # a 13-7 map only plays 8 second-half rounds
    assert summit.team1_sides.atk_won + summit.team2_sides.def_won == 8
    assert summit.team1_sides.ot_won == summit.team2_sides.ot_won == 0


_OT_MAP = """
<div class="vm-stats-game" data-game-id="1"><div class="vm-stats-game-header">
 <div class="team"><div class="score">12 </div><div><div class="team-name">A</div>
   <span class="mod-t">8</span> / <span class="mod-ct">4</span> / <span class="mod-ot">0</span></div></div>
 <div class="map"><div class="map-name">Lotus</div></div>
 <div class="team mod-right"><div><div class="team-name">B</div>
   <span class="mod-ct">4</span> / <span class="mod-t">8</span> / <span class="mod-ot">2</span></div>
   <div class="score mod-win">14</div></div>
</div></div>"""


def test_parse_map_sides_with_overtime():
    d = parse_match_detail(f"<html><body>{_OT_MAP}</body></html>", vlr_match_id=1)
    (m,) = d.maps
    assert (m.team1_score, m.team2_score) == (12, 14)
    assert (m.team1_sides.first_side, m.team1_sides.atk_won, m.team1_sides.def_won, m.team1_sides.ot_won) == ("atk", 8, 4, 0)
    assert (m.team2_sides.first_side, m.team2_sides.atk_won, m.team2_sides.def_won, m.team2_sides.ot_won) == ("def", 8, 4, 2)


def test_inconsistent_starting_sides_are_discarded():
    both_attack_first = _OT_MAP.replace('<span class="mod-ct">4</span> / <span class="mod-t">8</span> / <span class="mod-ot">2</span>',
                                       '<span class="mod-t">4</span> / <span class="mod-ct">8</span> / <span class="mod-ot">2</span>')
    (m,) = parse_match_detail(f"<html><body>{both_attack_first}</body></html>", vlr_match_id=1).maps
    assert m.team1_sides is None and m.team2_sides is None  # can't both start on attack


def test_parse_veto():
    steps = parse_veto("100 Thieves ban Bind; G2 Esports pick Haven; Team Vitality ban Split; Lotus remains")
    assert [(s.team, s.action, s.map_name) for s in steps] == [
        ("100 Thieves", "ban", "Bind"),
        ("G2 Esports", "pick", "Haven"),
        ("Team Vitality", "ban", "Split"),
        (None, "remains", "Lotus"),
    ]
    assert parse_veto(None) == []
    assert parse_veto("") == []


def test_is_international_event():
    assert is_international_event("Valorant Champions 2026") is True
    assert is_international_event("Valorant Masters Toronto 2026") is True
    assert is_international_event("Champions Tour 2026: Americas Stage 1") is False
    assert is_international_event("Esports World Cup 2026") is True
    assert is_international_event("Esports World Cup 2026: Americas Qualifier") is False
    assert is_international_event(None) is None


def test_parsers_fail_soft_on_empty_input():
    assert parse_results_page("<html><body></body></html>") == []
    d = parse_match_detail("<html><body></body></html>", vlr_match_id=1)
    assert d.team1_name is None and d.maps == [] and d.veto == []
