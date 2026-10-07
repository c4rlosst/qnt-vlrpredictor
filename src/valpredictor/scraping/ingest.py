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


def ingest_match(conn: sqlite3.Connection, client: VLRClient, vlr_match_id: int, match_url: str) -> int | None:
    """Fetch, parse and store one *completed* match. Returns its local id, or
    None if the match isn't finished yet / couldn't be fetched."""
    existing = conn.execute("SELECT id FROM matches WHERE vlr_id = ?", (vlr_match_id,)).fetchone()
    if existing:
        return existing["id"]

    try:
        detail = fetch_match_detail(client, vlr_match_id, match_url)
    except FetchError as exc:
        logger.warning("failed to fetch match %d: %s", vlr_match_id, exc)
        return None

    if detail.status != "final":
        return None

    event_id = db.upsert_event(conn, detail.event_name, detail.event_vlr_id)
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
    )

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
