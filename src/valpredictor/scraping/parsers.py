"""HTML parsers for vlr.gg pages.

Selectors here were written against real vlr.gg pages (a results page, the
upcoming-matches schedule, and match pages) and the trimmed copies in
tests/fixtures are real markup. vlr.gg can still change its markup; if a field
starts coming back None, run `python scripts/inspect_parse.py <url> match` and
fix the one selector involved.

Every extraction fails soft (None / empty list) so one drifted selector never
takes down a whole page.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_MATCH_ID = re.compile(r"^/(\d+)/")
_TEAM_ID = re.compile(r"/team/(\d+)/")
_EVENT_ID = re.compile(r"/event/(\d+)/")
_PLAYER_ID = re.compile(r"/player/(\d+)/")
_BEST_OF = re.compile(r"\bBo(\d)\b", re.IGNORECASE)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    t = " ".join(text.split())
    return t or None


def _text(tag: Tag | None) -> str | None:
    return _clean(tag.get_text(" ", strip=True)) if tag is not None else None


def _own_text(tag: Tag | None) -> str | None:
    """Text of a tag's direct text nodes only (ignores child elements)."""
    if tag is None:
        return None
    return _clean("".join(tag.find_all(string=True, recursive=False)))


def _int(text: str | None) -> int | None:
    if text is None:
        return None
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


# --------------------------------------------------------------------------
# Results list (/matches/results) and schedule (/matches) share one layout
# --------------------------------------------------------------------------

@dataclass
class ResultRow:
    vlr_match_id: int
    match_url: str
    date_label: str | None  # e.g. "Wed, October 7, 2026"
    time_label: str | None
    team1_name: str | None
    team2_name: str | None
    team1_score: int | None
    team2_score: int | None
    status: str | None  # "completed" | "upcoming" | "live"
    event_name: str | None
    series: str | None


def parse_results_page(html: str) -> list[ResultRow]:
    soup = _soup(html)
    rows: list[ResultRow] = []

    for label in soup.select(".wf-label.mod-large"):
        date_label = _own_text(label)
        card = label.find_next_sibling("div", class_="wf-card")
        if card is None:
            continue

        for item in card.select("a.match-item"):
            href = item.get("href") or ""
            m = _MATCH_ID.match(href)
            if not m:
                logger.warning("match-item without a numeric id: %r", href)
                continue

            teams = item.select(".match-item-vs-team")
            names = [_text(t.select_one(".match-item-vs-team-name .text-of")) for t in teams]
            scores = [_int(_text(t.select_one(".match-item-vs-team-score"))) for t in teams]
            names += [None] * (2 - len(names))
            scores += [None] * (2 - len(scores))

            status = _text(item.select_one(".ml-status"))
            rows.append(
                ResultRow(
                    vlr_match_id=int(m.group(1)),
                    match_url=href,
                    date_label=date_label,
                    time_label=_text(item.select_one(".match-item-time")),
                    team1_name=names[0],
                    team2_name=names[1],
                    team1_score=scores[0],
                    team2_score=scores[1],
                    status=status.lower() if status else None,
                    event_name=_own_text(item.select_one(".match-item-event")),
                    series=_text(item.select_one(".match-item-event-series")),
                )
            )

    return rows


# --------------------------------------------------------------------------
# Match page: /{id}/{slug}
# --------------------------------------------------------------------------

@dataclass
class OddsLine:
    """One bookmaker's decimal odds for the two teams, as vlr.gg lists them."""

    kind: str  # "pre-match" | "live"
    team1_odds: float
    team2_odds: float


@dataclass
class MapResult:
    map_order: int
    map_name: str | None
    team1_score: int | None
    team2_score: int | None


@dataclass
class MatchDetail:
    vlr_match_id: int
    team1_name: str | None = None
    team1_vlr_id: int | None = None
    team2_name: str | None = None
    team2_vlr_id: int | None = None
    unix_timestamp_ms: int | None = None
    event_name: str | None = None
    event_vlr_id: int | None = None
    best_of: int | None = None
    status: str | None = None  # "final" | "upcoming" | "live"
    odds: list[OddsLine] = field(default_factory=list)
    maps: list[MapResult] = field(default_factory=list)
    team1_lineup: list[tuple[str, int | None]] = field(default_factory=list)
    team2_lineup: list[tuple[str, int | None]] = field(default_factory=list)


def _parse_odds(soup: BeautifulSoup) -> list[OddsLine]:
    lines = []
    for item in soup.select(".match-bet-item"):
        note = (_text(item.select_one(".match-bet-item-note")) or "").lower()
        try:
            a = float(_text(item.select_one(".match-bet-item-odds.mod-1")))
            b = float(_text(item.select_one(".match-bet-item-odds.mod-2")))
        except (TypeError, ValueError):
            continue
        if a > 1.0 and b > 1.0 and note in ("pre-match", "live"):
            lines.append(OddsLine(kind=note, team1_odds=a, team2_odds=b))
    return lines


def _parse_lineups(soup: BeautifulSoup) -> tuple[list, list]:
    block = soup.select_one('.vm-stats-game[data-game-id="all"]')
    if block is None:
        return [], []
    lineups: list[list[tuple[str, int | None]]] = []
    for table in block.select(".ovw-table")[:2]:
        players = []
        for row in table.select(".ovw-row:not(.mod-head)"):
            link = row.select_one(".ovw-cell.mod-player a[href^='/player/']")
            name = _text(link.select_one(".ovw-player-name")) if link else None
            if not name:
                continue
            pm = _PLAYER_ID.search(link["href"])
            players.append((name, int(pm.group(1)) if pm else None))
        lineups.append(players)
    lineups += [[]] * (2 - len(lineups))
    return lineups[0], lineups[1]


def _parse_maps(soup: BeautifulSoup) -> list[MapResult]:
    maps: list[MapResult] = []
    games = [g for g in soup.select(".vm-stats-game[data-game-id]") if g.get("data-game-id") != "all"]
    for order, game in enumerate(games, start=1):
        header = game.select_one(".vm-stats-game-header")
        if header is None:
            continue
        scores = [_int(_text(s.select_one(".score"))) for s in header.select(".team")[:2]]
        scores += [None] * (2 - len(scores))
        if scores[0] is None or scores[1] is None:
            continue  # map listed but not played

        name_tag = header.select_one(".map-name")
        if name_tag is not None:
            for picked in name_tag.select(".picked"):
                picked.extract()  # the "PICK" badge sits inside the name element
        maps.append(MapResult(map_order=order, map_name=_text(name_tag), team1_score=scores[0], team2_score=scores[1]))
    return maps


def parse_match_detail(html: str, vlr_match_id: int) -> MatchDetail:
    soup = _soup(html)
    detail = MatchDetail(vlr_match_id=vlr_match_id)

    names, ids = [], []
    for link in soup.select(".match-header-link")[:2]:
        names.append(_text(link.select_one(".wf-title-med")))
        tm = _TEAM_ID.search(link.get("href") or "")
        ids.append(int(tm.group(1)) if tm else None)
    names += [None] * (2 - len(names))
    ids += [None] * (2 - len(ids))
    detail.team1_name, detail.team2_name = names
    detail.team1_vlr_id, detail.team2_vlr_id = ids

    ts = soup.select_one(".match-header-date .moment-tz-convert[data-utc-ts]")
    if ts is not None:
        try:
            utc = dt.datetime.strptime(ts["data-utc-ts"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            detail.unix_timestamp_ms = int(utc.timestamp() * 1000)
        except ValueError:
            logger.warning("unparseable data-utc-ts %r", ts.get("data-utc-ts"))

    event_link = soup.select_one("a.match-header-event")
    if event_link is not None:
        em = _EVENT_ID.search(event_link.get("href") or "")
        detail.event_vlr_id = int(em.group(1)) if em else None
        detail.event_name = _text(event_link.select_one("div > div"))

    for note in soup.select(".match-header-vs-note"):
        text = _text(note) or ""
        bm = _BEST_OF.search(text)
        if bm:
            detail.best_of = int(bm.group(1))
        elif text.lower() in ("final", "upcoming", "live"):
            detail.status = text.lower()

    detail.maps = _parse_maps(soup)
    detail.team1_lineup, detail.team2_lineup = _parse_lineups(soup)
    detail.odds = _parse_odds(soup)
    return detail
