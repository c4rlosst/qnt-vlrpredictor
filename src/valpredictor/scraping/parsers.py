"""HTML parsers for vlr.gg pages.

Selectors here were written against real vlr.gg pages (a results page, the
upcoming-matches schedule, and a completed Bo3 match page) and the trimmed
copies in tests/fixtures are real markup, not hand-written approximations.
vlr.gg can still change its markup; if a field starts coming back None, run
`python scripts/inspect_parse.py <url> match` and fix the one selector involved.

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
_VETO_STEP = re.compile(r"^(?P<team>.+?)\s+(?P<action>ban|pick)\s+(?P<map>\S+)$", re.IGNORECASE)
_VETO_REMAINS = re.compile(r"^(?P<map>\S+)\s+remains$", re.IGNORECASE)
_INTERNATIONAL_EVENT = re.compile(
    r"\bMasters\b|\bValorant Champions\b|\bEsports World Cup\b|\bLock//In\b", re.IGNORECASE
)


def is_international_event(event_name: str | None) -> bool | None:
    """Event-name heuristic: Masters / Champions / EWC are international LANs.
    Anything else is reported as not-international (False), which is a
    simplification — some regional finals are LAN too."""
    if not event_name:
        return None
    if re.search(r"qualifier", event_name, re.IGNORECASE):
        return False  # regional qualifiers for an international event are not themselves international
    return bool(_INTERNATIONAL_EVENT.search(event_name))


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
            event_box = item.select_one(".match-item-event")

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
                    event_name=_own_text(event_box),
                    series=_text(item.select_one(".match-item-event-series")),
                )
            )

    return rows


# --------------------------------------------------------------------------
# Match page: /{id}/{slug}
# --------------------------------------------------------------------------

@dataclass
class VetoStep:
    team: str | None  # None for the "X remains" leftover map
    action: str  # "ban" | "pick" | "remains"
    map_name: str


@dataclass
class TeamSides:
    """Rounds one team won per side on one map, as listed in the map header
    ('8 / 4 / 0' = 8 attack, 4 defence, 0 overtime; the first side listed is
    the side the team STARTED on)."""

    first_side: str | None  # "atk" | "def"
    atk_won: int
    def_won: int
    ot_won: int


@dataclass
class MapResult:
    map_order: int
    map_name: str | None
    team1_score: int | None
    team2_score: int | None
    picked_by: int | None  # 1 or 2 for the picking team; None for the decider
    team1_sides: TeamSides | None = None
    team2_sides: TeamSides | None = None


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
    series: str | None = None
    best_of: int | None = None
    status: str | None = None  # "final" | "upcoming" | "live"
    is_international: bool | None = None
    veto: list[VetoStep] = field(default_factory=list)
    maps: list[MapResult] = field(default_factory=list)
    team1_lineup: list[tuple[str, int | None]] = field(default_factory=list)
    team2_lineup: list[tuple[str, int | None]] = field(default_factory=list)


def parse_veto(note: str | None) -> list[VetoStep]:
    """'T1 ban Split; NRG pick Summit; Abyss remains' -> ordered VetoSteps."""
    steps: list[VetoStep] = []
    for part in (note or "").split(";"):
        part = part.strip()
        if not part:
            continue
        m = _VETO_REMAINS.match(part)
        if m:
            steps.append(VetoStep(team=None, action="remains", map_name=m.group("map")))
            continue
        m = _VETO_STEP.match(part)
        if m:
            steps.append(
                VetoStep(team=m.group("team").strip(), action=m.group("action").lower(), map_name=m.group("map"))
            )
    return steps


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


def _parse_team_sides(team_div: Tag) -> TeamSides | None:
    won = {"mod-t": 0, "mod-ct": 0, "mod-ot": 0}
    first_side = None
    for span in team_div.select("span.mod-t, span.mod-ct, span.mod-ot"):
        kind = next(c for c in span.get("class", []) if c in won)
        value = _int(_text(span))
        if value is None:
            return None
        won[kind] = value
        if first_side is None and kind != "mod-ot":
            first_side = "atk" if kind == "mod-t" else "def"
    if first_side is None:
        return None
    return TeamSides(first_side=first_side, atk_won=won["mod-t"], def_won=won["mod-ct"], ot_won=won["mod-ot"])


def _parse_maps(soup: BeautifulSoup, veto: list[VetoStep], team1: str | None, team2: str | None) -> list[MapResult]:
    picked_by_veto: dict[str, int] = {}
    for step in veto:
        if step.action == "pick" and step.team:
            if team1 and step.team.lower() == team1.lower():
                picked_by_veto[step.map_name.lower()] = 1
            elif team2 and step.team.lower() == team2.lower():
                picked_by_veto[step.map_name.lower()] = 2

    maps: list[MapResult] = []
    games = [g for g in soup.select(".vm-stats-game[data-game-id]") if g.get("data-game-id") != "all"]
    for order, game in enumerate(games, start=1):
        header = game.select_one(".vm-stats-game-header")
        if header is None:
            continue
        sides = header.select(".team")
        scores = [_int(_text(s.select_one(".score"))) for s in sides[:2]]
        scores += [None] * (2 - len(scores))
        if scores[0] is None or scores[1] is None:
            continue  # map listed but not played

        name_tag = header.select_one(".map-name")
        picked_by = None
        if name_tag is not None:
            picked_tag = name_tag.select_one(".picked")
            if picked_tag is not None:
                classes = picked_tag.get("class") or []
                picked_by = 1 if "mod-1" in classes else (2 if "mod-2" in classes else None)
                picked_tag.extract()
        map_name = _text(name_tag)
        if picked_by is None and map_name:
            picked_by = picked_by_veto.get(map_name.lower())

        sides = [_parse_team_sides(s) for s in sides[:2]]
        # the two teams must have started on opposite sides, or the markup is not what we think
        if len(sides) == 2 and sides[0] and sides[1] and sides[0].first_side == sides[1].first_side:
            sides = [None, None]
        sides += [None] * (2 - len(sides))

        maps.append(
            MapResult(
                map_order=order, map_name=map_name, team1_score=scores[0], team2_score=scores[1],
                picked_by=picked_by, team1_sides=sides[0], team2_sides=sides[1],
            )
        )
    return maps


def parse_match_detail(html: str, vlr_match_id: int) -> MatchDetail:
    soup = _soup(html)
    detail = MatchDetail(vlr_match_id=vlr_match_id)

    links = soup.select(".match-header-link")
    names, ids = [], []
    for link in links[:2]:
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
        detail.series = _text(event_link.select_one(".match-header-event-series"))
        detail.is_international = is_international_event(detail.event_name)

    for note in soup.select(".match-header-vs-note"):
        text = _text(note) or ""
        bm = _BEST_OF.search(text)
        if bm:
            detail.best_of = int(bm.group(1))
        elif text.lower() in ("final", "upcoming", "live"):
            detail.status = text.lower()

    detail.veto = parse_veto(_text(soup.select_one(".match-header-note")))
    detail.maps = _parse_maps(soup, detail.veto, detail.team1_name, detail.team2_name)
    detail.team1_lineup, detail.team2_lineup = _parse_lineups(soup)
    return detail
