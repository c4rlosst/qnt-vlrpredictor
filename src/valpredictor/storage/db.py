"""SQLite connection management and upsert helpers.

Upserts are keyed on the vlr.gg numeric id when we have one (reliable, stable)
and fall back to matching on name when we don't. That fallback can create
duplicate rows for a team whose name was written two ways, so prefer fixing
the parser over relying on it.

Databases made by earlier versions carry a few extra columns (sides, picks,
stage). They are unused now and harmless.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from valpredictor.config import load_config, resolve_path

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else resolve_path(load_config()["database"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)  # a long scrape may be writing at the same time
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def upsert_team(conn: sqlite3.Connection, name: str | None, vlr_id: int | None) -> int | None:
    if not name and vlr_id is None:
        return None
    if vlr_id is not None:
        row = conn.execute("SELECT id FROM teams WHERE vlr_id = ?", (vlr_id,)).fetchone()
        if row:
            if name:
                conn.execute("UPDATE teams SET name = ? WHERE id = ?", (name, row["id"]))
            return row["id"]
        cur = conn.execute("INSERT INTO teams (vlr_id, name) VALUES (?, ?)", (vlr_id, name or f"team-{vlr_id}"))
        return cur.lastrowid

    row = conn.execute("SELECT id FROM teams WHERE name = ? AND vlr_id IS NULL", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO teams (vlr_id, name) VALUES (NULL, ?)", (name,))
    return cur.lastrowid


def upsert_player(conn: sqlite3.Connection, name: str | None, vlr_id: int | None = None) -> int | None:
    if not name and vlr_id is None:
        return None
    if vlr_id is not None:
        row = conn.execute("SELECT id FROM players WHERE vlr_id = ?", (vlr_id,)).fetchone()
        if row:
            return row["id"]
        cur = conn.execute("INSERT INTO players (vlr_id, name) VALUES (?, ?)", (vlr_id, name or f"player-{vlr_id}"))
        return cur.lastrowid

    row = conn.execute("SELECT id FROM players WHERE name = ? AND vlr_id IS NULL", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO players (vlr_id, name) VALUES (NULL, ?)", (name,))
    return cur.lastrowid


def upsert_event(conn: sqlite3.Connection, name: str | None, vlr_id: int | None) -> int | None:
    if not name and vlr_id is None:
        return None
    if vlr_id is not None:
        row = conn.execute("SELECT id FROM events WHERE vlr_id = ?", (vlr_id,)).fetchone()
        if row:
            return row["id"]
        return conn.execute("INSERT INTO events (vlr_id, name) VALUES (?, ?)", (vlr_id, name)).lastrowid

    row = conn.execute("SELECT id FROM events WHERE name = ? AND vlr_id IS NULL", (name,)).fetchone()
    if row:
        return row["id"]
    return conn.execute("INSERT INTO events (vlr_id, name) VALUES (NULL, ?)", (name,)).lastrowid


def upsert_match(
    conn: sqlite3.Connection,
    *,
    vlr_id: int,
    match_url: str | None,
    event_id: int | None,
    unix_timestamp_ms: int | None,
    team1_id: int | None,
    team2_id: int | None,
    best_of: int | None,
    team1_score: int | None,
    team2_score: int | None,
) -> int:
    match_date = None
    if unix_timestamp_ms:
        match_date = dt.datetime.fromtimestamp(unix_timestamp_ms / 1000, tz=dt.timezone.utc).date().isoformat()

    winner_team_id = None
    if team1_score is not None and team2_score is not None and team1_score != team2_score:
        winner_team_id = team1_id if team1_score > team2_score else team2_id

    scraped_at = dt.datetime.now(dt.timezone.utc).isoformat()
    values = (match_url, event_id, match_date, unix_timestamp_ms, team1_id, team2_id, best_of,
              team1_score, team2_score, winner_team_id, scraped_at)

    row = conn.execute("SELECT id FROM matches WHERE vlr_id = ?", (vlr_id,)).fetchone()
    if row:
        conn.execute(
            """
            UPDATE matches SET
                match_url = ?, event_id = ?, match_date = ?, unix_timestamp_ms = ?,
                team1_id = ?, team2_id = ?, best_of = ?, team1_score = ?, team2_score = ?,
                winner_team_id = ?, scraped_at = ?
            WHERE id = ?
            """,
            (*values, row["id"]),
        )
        return row["id"]

    return conn.execute(
        """
        INSERT INTO matches (
            match_url, event_id, match_date, unix_timestamp_ms, team1_id, team2_id, best_of,
            team1_score, team2_score, winner_team_id, scraped_at, vlr_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (*values, vlr_id),
    ).lastrowid


def replace_maps(conn: sqlite3.Connection, match_id: int, maps: list[dict]) -> None:
    """`maps` rows: {map_order, map_name, team1_score, team2_score, team1_id, team2_id}."""
    conn.execute("DELETE FROM maps WHERE match_id = ?", (match_id,))
    for m in maps:
        winner_team_id = None
        s1, s2 = m.get("team1_score"), m.get("team2_score")
        if s1 is not None and s2 is not None and s1 != s2:
            winner_team_id = m.get("team1_id") if s1 > s2 else m.get("team2_id")
        conn.execute(
            """
            INSERT INTO maps (match_id, map_order, map_name, team1_score, team2_score, winner_team_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (match_id, m["map_order"], m.get("map_name"), s1, s2, winner_team_id),
        )


def replace_roster(conn: sqlite3.Connection, match_id: int, team_id: int, player_ids: list[int]) -> None:
    conn.execute("DELETE FROM rosters WHERE match_id = ? AND team_id = ?", (match_id, team_id))
    for pid in player_ids:
        conn.execute(
            "INSERT OR IGNORE INTO rosters (match_id, team_id, player_id) VALUES (?, ?, ?)",
            (match_id, team_id, pid),
        )


def find_team_id_by_name(conn: sqlite3.Connection, name: str) -> int | None:
    """Case-insensitive exact match first, then a substring fallback."""
    row = conn.execute("SELECT id FROM teams WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    if row:
        return row["id"]
    row = conn.execute(
        "SELECT id FROM teams WHERE name LIKE ? COLLATE NOCASE ORDER BY LENGTH(name) LIMIT 1",
        (f"%{name}%",),
    ).fetchone()
    return row["id"] if row else None
