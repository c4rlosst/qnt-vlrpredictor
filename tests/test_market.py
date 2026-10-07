from pathlib import Path

import pytest

from valpredictor.market import implied_team1_probability
from valpredictor.scraping.parsers import OddsLine, parse_match_detail
from valpredictor.upcoming import pre_match_market

FIXTURES = Path(__file__).parent / "fixtures"


def test_implied_probability_removes_the_bookmaker_margin():
    # 1/1.69 + 1/2.12 = 1.063 (a 6% margin); normalised, PRX is 55.6%
    p, books = implied_team1_probability([OddsLine("pre-match", 1.69, 2.12)], "pre-match")
    assert p == pytest.approx(0.5565, abs=1e-3) and books == 1
    # an even market stays even whatever the margin
    assert implied_team1_probability([OddsLine("pre-match", 1.9, 1.9)], "pre-match")[0] == pytest.approx(0.5)


def test_implied_probability_averages_books_and_filters_by_kind():
    lines = [OddsLine("pre-match", 1.5, 2.6), OddsLine("pre-match", 1.6, 2.4), OddsLine("live", 6.1, 1.1)]
    p, books = implied_team1_probability(lines, "pre-match")
    assert books == 2 and 0.6 < p < 0.66
    assert implied_team1_probability(lines, "live")[0] < 0.2
    assert implied_team1_probability([], "pre-match") is None
    assert implied_team1_probability(lines[:1], "live") is None


def test_pre_match_market_from_the_real_page_ignores_live_odds():
    d = parse_match_detail((FIXTURES / "match_with_odds.html").read_text(encoding="utf-8"), 754733)
    market = pre_match_market(d)
    assert market["team1"] == pytest.approx(0.5565, abs=1e-3) and market["team1"] + market["team2"] == pytest.approx(1.0)
    assert market["books"] == 1
