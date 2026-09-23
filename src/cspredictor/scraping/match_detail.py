"""Fetch and parse a single HLTV match page."""

from __future__ import annotations

from cspredictor.scraping.client import HLTVClient
from cspredictor.scraping.parsers import MatchDetail, parse_match_detail


def fetch_match_detail(client: HLTVClient, hltv_match_id: int, match_url: str) -> MatchDetail:
    html = client.get(match_url)
    return parse_match_detail(html, hltv_match_id)
