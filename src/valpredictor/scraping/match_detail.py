"""Fetch and parse a single vlr.gg match page."""

from __future__ import annotations

from valpredictor.scraping.client import VLRClient
from valpredictor.scraping.parsers import MatchDetail, parse_match_detail


def fetch_match_detail(
    client: VLRClient, vlr_match_id: int, match_url: str, force_refresh: bool = False
) -> MatchDetail:
    html = client.get(match_url, force_refresh=force_refresh)
    return parse_match_detail(html, vlr_match_id)
