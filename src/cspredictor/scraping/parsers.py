"""HTML parsers for HLTV pages.

IMPORTANT — read this before trusting any output of this module:

This environment has no outbound network access to hltv.org, so none of the
selectors below have been run against a real, live HLTV page. They're written
against HLTV's long-standing, widely-documented markup conventions (the same
structure referenced by most open-source HLTV scrapers), but HLTV tweaks its
markup occasionally and this code has NOT been visually verified.

Every extraction is isolated in its own small function and fails soft
(returns None / logs a warning) instead of raising, so a single drifted
selector doesn't take down a whole page parse. Before relying on scraped
data, run `python scripts/save_page.py <url>` locally, then
`python scripts/inspect_parse.py <url> results|match|ranking` to pretty-print
what the parser extracted next to the raw HTML and fix any selector that's
off. See README "Verifying the scraper" for the full workflow.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_ID_FROM_MATCH_URL = re.compile(r"/matches/(\d+)/([^/?#]+)")
_ID_FROM_TEAM_URL = re.compile(r"/team/(\d+)/([^/?#]+)")
_ID_FROM_EVENT_URL = re.compile(r"/events/(\d+)/([^/?#]+)")
_VETO_LINE = re.compile(
    r"^\s*\d+\.\s+(?P<team>.+?)\s+(?P<action>picked|removed)\s+(?P<map>.+?)\s*$",
    re.IGNORECASE,
)
_LEFTOVER_LINE = re.compile(r"^\s*(?P<map>.+?)\s+was left over\s*$", re.IGNORECASE)
_BEST_OF = re.compile(r"Best of (\d+)", re.IGNORECASE)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _text(tag: Tag | None) -> str | None:
    if tag is None:
        return None
    t = tag.get_text(strip=True)
    return t or None


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


# --------------------------------------------------------------------------
# Results list page: https://www.hltv.org/results (and its date/filter params)
# --------------------------------------------------------------------------

@dataclass
class ResultRow:
    hltv_match_id: int
    match_url: str
    date_label: str | None  # raw text from the sublist header, e.g. "Results for 22nd September 2026"
    team1_name: str | None
    team2_name: str | None
    team1_score: int | None
    team2_score: int | None
    bo1_map: str | None  # only present when the list shows a single map (Bo1)
    event_name: str | None


def parse_results_page(html: str) -> list[ResultRow]:
    soup = _soup(html)
    rows: list[ResultRow] = []

    sublists = soup.select(".results-sublist") or [soup]
    for sublist in sublists:
        headline = sublist.select_one(".standard-headline")
        date_label = _text(headline)

        for con in sublist.select(".result-con"):
            link = con.select_one("a.a-reset") or con.find("a")
            if link is None or not link.get("href"):
                logger.warning("result-con with no link, skipping")
                continue
            m = _ID_FROM_MATCH_URL.search(link["href"])
            if not m:
                logger.warning("could not extract match id from %s", link["href"])
                continue
            match_id = int(m.group(1))

            team_divs = con.select(".team")
            team1_name = _text(team_divs[0]) if len(team_divs) > 0 else None
            team2_name = _text(team_divs[1]) if len(team_divs) > 1 else None

            score_spans = con.select(".result-score span")
            team1_score = _int(_text(score_spans[0])) if len(score_spans) > 0 else None
            team2_score = _int(_text(score_spans[1])) if len(score_spans) > 1 else None

            bo1_map = _text(con.select_one(".map-text"))
            event_name = _text(con.select_one(".event-name")) or _text(
                con.select_one(".event-logo-container img[alt]")
            )

            rows.append(
                ResultRow(
                    hltv_match_id=match_id,
                    match_url=link["href"],
                    date_label=date_label,
                    team1_name=team1_name,
                    team2_name=team2_name,
                    team1_score=team1_score,
                    team2_score=team2_score,
                    bo1_map=bo1_map,
                    event_name=event_name,
                )
            )

    return rows


# --------------------------------------------------------------------------
# Match detail page: https://www.hltv.org/matches/{id}/{slug}
# --------------------------------------------------------------------------

@dataclass
class MapResult:
    map_order: int
    map_name: str | None
    team1_score: int | None
    team2_score: int | None
    picked_by: str | None  # team name, or None for a decider/leftover map


@dataclass
class MatchDetail:
    hltv_match_id: int
    team1_name: str | None
    team2_name: str | None
    unix_timestamp_ms: int | None
    event_name: str | None
    event_hltv_id: int | None
    best_of: int | None
    maps: list[MapResult] = field(default_factory=list)
    team1_lineup: list[str] = field(default_factory=list)
    team2_lineup: list[str] = field(default_factory=list)
    is_lan: bool | None = None  # best-effort; verify against a real page


def _parse_team_names(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    box = soup.select_one(".teamsBox")
    if box is None:
        return None, None
    t1 = _text(box.select_one(".team1-gradient .text-ellipsis")) or _text(
        box.select_one(".team1")
    )
    t2 = _text(box.select_one(".team2-gradient .text-ellipsis")) or _text(
        box.select_one(".team2")
    )
    return t1, t2


def _parse_veto(html_text: str) -> dict[str, str]:
    """Map lowercased map name -> picking team name, from the veto prose block."""
    picks: dict[str, str] = {}
    for line in html_text.splitlines():
        m = _VETO_LINE.match(line)
        if m and m.group("action").lower() == "picked":
            picks[m.group("map").strip().lower()] = m.group("team").strip()
    return picks


def parse_match_detail(html: str, hltv_match_id: int) -> MatchDetail:
    soup = _soup(html)

    team1_name, team2_name = _parse_team_names(soup)

    time_div = soup.select_one(".timeAndEvent .time[data-unix]")
    unix_ms = None
    if time_div is not None and time_div.get("data-unix"):
        try:
            unix_ms = int(time_div["data-unix"])
        except ValueError:
            logger.warning("non-numeric data-unix for match %s", hltv_match_id)

    event_link = soup.select_one(".timeAndEvent .event a[href]")
    event_name = _text(event_link)
    event_hltv_id = None
    if event_link is not None and event_link.get("href"):
        m = _ID_FROM_EVENT_URL.search(event_link["href"])
        if m:
            event_hltv_id = int(m.group(1))

    best_of = None
    bo_div = soup.select_one(".timeAndEvent .preformatted-text")
    if bo_div is not None:
        m = _BEST_OF.search(_text(bo_div) or "")
        if m:
            best_of = int(m.group(1))

    veto_div = soup.select_one(".veto-box .preformatted-text, .standard-box.veto-box")
    picks_by_map = _parse_veto(veto_div.get_text("\n", strip=True)) if veto_div else {}

    maps: list[MapResult] = []
    for i, holder in enumerate(soup.select(".mapholder"), start=1):
        map_name = _text(holder.select_one(".mapname"))
        scores = holder.select(".results-team-score")
        s1 = _int(_text(scores[0])) if len(scores) > 0 else None
        s2 = _int(_text(scores[1])) if len(scores) > 1 else None
        picked_by = picks_by_map.get((map_name or "").lower())
        maps.append(
            MapResult(map_order=i, map_name=map_name, team1_score=s1, team2_score=s2, picked_by=picked_by)
        )

    lineups = soup.select(".lineup")
    team1_lineup = [
        _text(p) for p in (lineups[0].select(".player-nick") if len(lineups) > 0 else [])
    ]
    team2_lineup = [
        _text(p) for p in (lineups[1].select(".player-nick") if len(lineups) > 1 else [])
    ]

    is_lan = None
    page_text = soup.get_text(" ", strip=True)
    if re.search(r"\bLAN\b", page_text):
        is_lan = True
    elif re.search(r"\bOnline\b", page_text):
        is_lan = False

    return MatchDetail(
        hltv_match_id=hltv_match_id,
        team1_name=team1_name,
        team2_name=team2_name,
        unix_timestamp_ms=unix_ms,
        event_name=event_name,
        event_hltv_id=event_hltv_id,
        best_of=best_of,
        maps=maps,
        team1_lineup=[p for p in team1_lineup if p],
        team2_lineup=[p for p in team2_lineup if p],
        is_lan=is_lan,
    )


# --------------------------------------------------------------------------
# Rankings page: https://www.hltv.org/ranking/teams/{year}/{month}/{day}
# --------------------------------------------------------------------------

@dataclass
class RankingRow:
    rank: int | None
    team_name: str | None
    team_hltv_id: int | None
    points: int | None


def parse_rankings_page(html: str) -> list[RankingRow]:
    soup = _soup(html)
    rows: list[RankingRow] = []

    for team_div in soup.select(".ranked-team"):
        rank = _int(_text(team_div.select_one(".position")))
        name = _text(team_div.select_one(".name"))
        points = _int(_text(team_div.select_one(".points")))

        team_hltv_id = None
        link = team_div.select_one("a.moreLink[href], a[href*='/team/']")
        if link is not None and link.get("href"):
            m = _ID_FROM_TEAM_URL.search(link["href"])
            if m:
                team_hltv_id = int(m.group(1))

        rows.append(RankingRow(rank=rank, team_name=name, team_hltv_id=team_hltv_id, points=points))

    return rows
