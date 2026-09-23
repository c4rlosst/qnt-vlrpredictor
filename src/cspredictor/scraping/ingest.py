"""Glue: scrape -> parse -> upsert into SQLite, with resumable backfill.

Resumability: after each successfully-ingested match we record its HLTV id
under scrape_state["last_ingested_match_id"], and the results list is walked
newest-first, so `run_backfill` skips any match id it's already stored. An
interrupted run just needs to be re-invoked with the same date range.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3

from tqdm import tqdm

from cspredictor.scraping.client import FetchError, HLTVClient
from cspredictor.scraping.match_detail import fetch_match_detail
from cspredictor.scraping.rankings import iter_ranking_snapshots
from cspredictor.scraping.results import iter_results
from cspredictor.storage import db

logger = logging.getLogger(__name__)


def ingest_match(conn: sqlite3.Connection, client: HLTVClient, hltv_match_id: int, match_url: str) -> int | None:
    existing = conn.execute(
        "SELECT id FROM matches WHERE hltv_id = ?", (hltv_match_id,)
    ).fetchone()
    if existing:
        return existing["id"]

    try:
        detail = fetch_match_detail(client, hltv_match_id, match_url)
    except FetchError as exc:
        logger.warning("failed to fetch match %d: %s", hltv_match_id, exc)
        return None

    event_id = db.upsert_event(conn, detail.event_name, detail.event_hltv_id, detail.is_lan)
    team1_id = db.upsert_team(conn, detail.team1_name, hltv_id=None)
    team2_id = db.upsert_team(conn, detail.team2_name, hltv_id=None)

    team1_maps_won = sum(
        1 for m in detail.maps if m.team1_score is not None and m.team2_score is not None
        and m.team1_score > m.team2_score
    )
    team2_maps_won = sum(
        1 for m in detail.maps if m.team1_score is not None and m.team2_score is not None
        and m.team2_score > m.team1_score
    )

    match_id = db.upsert_match(
        conn,
        hltv_id=hltv_match_id,
        match_url=match_url,
        event_id=event_id,
        unix_timestamp_ms=detail.unix_timestamp_ms,
        team1_id=team1_id,
        team2_id=team2_id,
        best_of=detail.best_of,
        team1_score=team1_maps_won if detail.maps else None,
        team2_score=team2_maps_won if detail.maps else None,
        is_lan=detail.is_lan,
    )

    map_rows = []
    for m in detail.maps:
        picked_by_team_id = None
        if m.picked_by and team1_id and team2_id:
            if detail.team1_name and m.picked_by.strip().lower() == detail.team1_name.strip().lower():
                picked_by_team_id = team1_id
            elif detail.team2_name and m.picked_by.strip().lower() == detail.team2_name.strip().lower():
                picked_by_team_id = team2_id
        map_rows.append(
            {
                "map_order": m.map_order,
                "map_name": m.map_name,
                "team1_score": m.team1_score,
                "team2_score": m.team2_score,
                "team1_id": team1_id,
                "team2_id": team2_id,
                "picked_by_team_id": picked_by_team_id,
            }
        )
    db.replace_maps(conn, match_id, map_rows)

    if team1_id and detail.team1_lineup:
        player_ids = [db.upsert_player(conn, name) for name in detail.team1_lineup]
        db.replace_roster(conn, match_id, team1_id, [p for p in player_ids if p])
    if team2_id and detail.team2_lineup:
        player_ids = [db.upsert_player(conn, name) for name in detail.team2_lineup]
        db.replace_roster(conn, match_id, team2_id, [p for p in player_ids if p])

    conn.commit()
    return match_id


def run_backfill(
    conn: sqlite3.Connection,
    client: HLTVClient,
    since: dt.date,
    until: dt.date | None = None,
) -> int:
    """Ingest every match in [since, until]. Returns count of newly-ingested matches."""
    ingested = 0
    for row in tqdm(iter_results(client, since, until), desc="matches"):
        match_id = ingest_match(conn, client, row.hltv_match_id, row.match_url)
        if match_id is not None:
            ingested += 1
            db.set_scrape_state(conn, "last_ingested_match_id", str(row.hltv_match_id))
            conn.commit()
    return ingested


def run_rankings_backfill(
    conn: sqlite3.Connection,
    client: HLTVClient,
    since: dt.date,
    until: dt.date | None = None,
) -> int:
    snapshots_ingested = 0
    for snapshot_date, rows in tqdm(iter_ranking_snapshots(client, since, until), desc="ranking snapshots"):
        for r in rows:
            team_id = db.upsert_team(conn, r.team_name, r.team_hltv_id)
            if team_id is None:
                continue
            db.upsert_ranking_snapshot(conn, team_id, snapshot_date, r.rank, r.points)
        conn.commit()
        snapshots_ingested += 1
    return snapshots_ingested
