"""Glue: scrape -> parse -> upsert into SQLite.

Resumability: a match already in the DB is skipped without any network call,
and match pages are disk-cached, so re-running an interrupted backfill just
continues where it stopped. The (small) results-list pages are always
re-fetched so newly finished matches are never hidden behind a stale cache.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

from tqdm import tqdm

from valpredictor.scraping.client import FetchError, VLRClient
from valpredictor.scraping.match_detail import fetch_match_detail
from valpredictor.scraping.results import iter_results
from valpredictor.storage import db

logger = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    ingested: int = 0
    already_stored: int = 0
    filtered_out: int = 0
    failed: int = 0
    events_included: Counter = field(default_factory=Counter)


def event_matches(name: str | None, include: re.Pattern | None, exclude: re.Pattern | None) -> bool:
    name = name or ""
    if include is not None and not include.search(name):
        return False
    if exclude is not None and exclude.search(name):
        return False
    return True


def _side_columns(m) -> dict:
    """Flatten a parsed map's per-team side data into maps-table columns."""
    s1, s2 = m.team1_sides, m.team2_sides
    if s1 is None or s2 is None:
        return {}
    return {
        "team1_start_side": s1.first_side,
        "team1_atk_won": s1.atk_won, "team1_def_won": s1.def_won, "team1_ot_won": s1.ot_won,
        "team2_atk_won": s2.atk_won, "team2_def_won": s2.def_won, "team2_ot_won": s2.ot_won,
    }


def ingest_match(
    conn: sqlite3.Connection, client: VLRClient, vlr_match_id: int, match_url: str, force: bool = False
) -> int | None:
    """Fetch, parse and store one *completed* match. Returns its local id, or
    None if the match isn't finished yet / couldn't be fetched. With
    `force=True` an already-stored match is re-parsed and overwritten (used to
    backfill newly parsed fields from the page cache)."""
    existing = conn.execute("SELECT id FROM matches WHERE vlr_id = ?", (vlr_match_id,)).fetchone()
    if existing and not force:
        return existing["id"]

    try:
        detail = fetch_match_detail(client, vlr_match_id, match_url)
    except FetchError as exc:
        logger.warning("failed to fetch match %d: %s", vlr_match_id, exc)
        return None

    if detail.status != "final":
        return None

    event_id = db.upsert_event(conn, detail.event_name, detail.event_vlr_id, detail.is_international)
    team1_id = db.upsert_team(conn, detail.team1_name, detail.team1_vlr_id)
    team2_id = db.upsert_team(conn, detail.team2_name, detail.team2_vlr_id)

    team1_maps = sum(1 for m in detail.maps if m.team1_score > m.team2_score)
    team2_maps = sum(1 for m in detail.maps if m.team2_score > m.team1_score)

    match_id = db.upsert_match(
        conn,
        vlr_id=vlr_match_id,
        match_url=match_url,
        event_id=event_id,
        unix_timestamp_ms=detail.unix_timestamp_ms,
        team1_id=team1_id,
        team2_id=team2_id,
        best_of=detail.best_of,
        team1_score=team1_maps if detail.maps else None,
        team2_score=team2_maps if detail.maps else None,
        is_international=detail.is_international,
    )

    pick_team = {1: team1_id, 2: team2_id}
    db.replace_maps(
        conn,
        match_id,
        [
            {
                "map_order": m.map_order,
                "map_name": m.map_name,
                "team1_score": m.team1_score,
                "team2_score": m.team2_score,
                "team1_id": team1_id,
                "team2_id": team2_id,
                "picked_by_team_id": pick_team.get(m.picked_by),
                **_side_columns(m),
            }
            for m in detail.maps
        ],
    )

    for team_id, lineup in ((team1_id, detail.team1_lineup), (team2_id, detail.team2_lineup)):
        if team_id and lineup:
            player_ids = [db.upsert_player(conn, name, pid) for name, pid in lineup]
            db.replace_roster(conn, match_id, team_id, [p for p in player_ids if p])

    conn.commit()
    return match_id


def reparse_from_cache(conn: sqlite3.Connection, client: VLRClient) -> tuple[int, int]:
    """Re-read every stored match from the page cache (no network) and
    overwrite it, to fill fields added to the parser after it was first
    scraped. Returns (matches updated, matches whose page was not cached)."""
    rows = conn.execute("SELECT vlr_id, match_url FROM matches WHERE match_url IS NOT NULL").fetchall()
    updated = missing = 0
    for row in tqdm(rows, desc="reparse"):
        if not client.has_cached(row["match_url"]):
            missing += 1
            continue
        if ingest_match(conn, client, row["vlr_id"], row["match_url"], force=True) is not None:
            updated += 1
    return updated, missing


def scan_events(client: VLRClient, since: dt.date, until: dt.date | None = None) -> Counter:
    """Count matches per event name over a date range (list pages only)."""
    counts: Counter = Counter()
    for row in tqdm(iter_results(client, since, until, refresh=True), desc="list pages"):
        counts[row.event_name or "(unknown)"] += 1
    return counts


def run_backfill(
    conn: sqlite3.Connection,
    client: VLRClient,
    since: dt.date,
    until: dt.date | None = None,
    include: re.Pattern | None = None,
    exclude: re.Pattern | None = None,
    progress=None,
) -> BackfillStats:
    """`progress(stats)` is called after every match that is stored or fails."""
    stats = BackfillStats()
    for row in tqdm(iter_results(client, since, until, refresh=True), desc="matches"):
        if row.status != "completed" or not event_matches(row.event_name, include, exclude):
            stats.filtered_out += 1
            continue
        already = conn.execute("SELECT 1 FROM matches WHERE vlr_id = ?", (row.vlr_match_id,)).fetchone()
        if already:
            stats.already_stored += 1
            continue
        if ingest_match(conn, client, row.vlr_match_id, row.match_url) is None:
            stats.failed += 1
        else:
            stats.ingested += 1
            stats.events_included[row.event_name or "(unknown)"] += 1
        if progress:
            progress(stats)
    return stats
