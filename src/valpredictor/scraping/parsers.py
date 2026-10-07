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
_LOWER_LEVEL_EVENT = re.compile(r"qualifier|challengers|ascension|game changers|academy", re.IGNORECASE)


def is_international_event(event_name: str | None) -> bool | None:
    """Masters / Champions / Esports World Cup. Their regional qualifiers and
    Challengers-level events (some of which say "Masters") are not international."""
    if not event_name:
        return None
    if _LOWER_LEVEL_EVENT.search(event_name):
        return False
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
class OddsLine:
    """One bookmaker's decimal odds for the two teams, as vlr.gg lists them."""

    kind: str  # "pre-match" | "live"
    team1_odds: float
    team2_odds: float


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
    odds: list[OddsLine] = field(default_factory=list)
    veto: list[VetoStep] = field(default_factory=list)
    veto_actors: dict[str, int] = field(default_factory=dict)  # veto label ('PRX') -> team 1 or 2
    maps: list[MapResult] = field(default_factory=list)
    team1_lineup: list[tuple[str, int | None]] = field(default_factory=list)
    team2_lineup: list[tuple[str, int | None]] = field(default_factory=list)


def _parse_odds(soup: BeautifulSoup) -> list[OddsLine]:
    lines = []
    for item in soup.select(".match-bet-item"):
        o1 = item.select_one(".match-bet-item-odds.mod-1")
        o2 = item.select_one(".match-bet-item-odds.mod-2")
        note = (_text(item.select_one(".match-bet-item-note")) or "").lower()
        try:
            a, b = float(_text(o1)), float(_text(o2))
        except (TypeError, ValueError):
            continue
        if a > 1.0 and b > 1.0 and note in ("pre-match", "live"):
            lines.append(OddsLine(kind=note, team1_odds=a, team2_odds=b))
    return lines


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


def _parse_tags(soup: BeautifulSoup) -> dict[str, str]:
    """{team name (lowercase): tag} for the two teams, e.g. {'paper rex': 'PRX'}. The veto text
    names teams by tag, the page header by full name; the odds boxes and round tables list both."""
    tags: dict[str, str] = {}
    for half in soup.select(".match-bet-item-half"):
        name = _text(half.select_one(".match-bet-item-team-name"))
        tag = _text(half.select_one(".match-bet-item-team-tag"))
        if name and tag:
            tags[name.lower()] = tag
    for el_ in soup.select(".vlr-rounds-tag[title]"):
        title, tag = el_.get("title"), _text(el_)
        if title and tag:
            tags.setdefault(title.lower(), tag)
    return tags


def resolve_veto_actors(
    veto: list[VetoStep], team1: str | None, team2: str | None, tags: dict[str, str]
) -> dict[str, int]:
    """Map each team label used in the veto text ('PRX', 'LOUD') to team 1 or 2.
    Matches on the full name or the tag; when only one of the two labels resolves,
    the other must be the remaining team."""
    labels = {1: {team1 or "", tags.get((team1 or "").lower(), "")}, 2: {team2 or "", tags.get((team2 or "").lower(), "")}}
    labels = {k: {x.lower() for x in v if x} for k, v in labels.items()}
    actors = list(dict.fromkeys(s.team for s in veto if s.team))
    resolved = {a: team for a in actors for team, names in labels.items() if a.lower() in names}
    unresolved = [a for a in actors if a not in resolved]
    if len(unresolved) == 1 and len(set(resolved.values())) == 1:
        resolved[unresolved[0]] = 3 - next(iter(resolved.values()))
    return resolved


def _parse_maps(soup: BeautifulSoup, veto: list[VetoStep], actors: dict[str, int]) -> list[MapResult]:
    picked_by_veto = {s.map_name.lower(): actors[s.team] for s in veto if s.action == "pick" and s.team in actors}

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
    detail.veto_actors = resolve_veto_actors(detail.veto, detail.team1_name, detail.team2_name, _parse_tags(soup))
    detail.maps = _parse_maps(soup, detail.veto, detail.veto_actors)
    detail.team1_lineup, detail.team2_lineup = _parse_lineups(soup)
    detail.odds = _parse_odds(soup)
    return detail
