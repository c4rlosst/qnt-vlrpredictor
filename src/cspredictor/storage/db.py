"""SQLite connection management and upsert helpers.

All "upsert" helpers here are keyed on the HLTV numeric id when we have one
(reliable, stable) and fall back to matching on name when we don't (e.g. a
selector drift left `hltv_id` as None) — documented as a known limitation:
name-only matching can create duplicate rows for teams whose name we saw
written two different ways. Prefer fixing the parser over relying on the
name fallback for real data.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

from cspredictor.config import load_config, resolve_path

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else resolve_path(load_config()["database"]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def upsert_team(conn: sqlite3.Connection, name: str | None, hltv_id: int | None) -> int | None:
    if not name and hltv_id is None:
        return None
    if hltv_id is not None:
        row = conn.execute("SELECT id FROM teams WHERE hltv_id = ?", (hltv_id,)).fetchone()
        if row:
            if name:
                conn.execute("UPDATE teams SET name = ? WHERE id = ?", (name, row["id"]))
            return row["id"]
        cur = conn.execute(
            "INSERT INTO teams (hltv_id, name) VALUES (?, ?)", (hltv_id, name or f"team-{hltv_id}")
        )
        return cur.lastrowid

    row = conn.execute("SELECT id FROM teams WHERE name = ? AND hltv_id IS NULL", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO teams (hltv_id, name) VALUES (NULL, ?)", (name,))
    return cur.lastrowid


def upsert_player(conn: sqlite3.Connection, name: str | None, hltv_id: int | None = None) -> int | None:
    if not name and hltv_id is None:
        return None
    if hltv_id is not None:
        row = conn.execute("SELECT id FROM players WHERE hltv_id = ?", (hltv_id,)).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            "INSERT INTO players (hltv_id, name) VALUES (?, ?)", (hltv_id, name or f"player-{hltv_id}")
        )
        return cur.lastrowid

    row = conn.execute("SELECT id FROM players WHERE name = ? AND hltv_id IS NULL", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO players (hltv_id, name) VALUES (NULL, ?)", (name,))
    return cur.lastrowid


def upsert_event(
    conn: sqlite3.Connection, name: str | None, hltv_id: int | None, is_lan: bool | None = None
) -> int | None:
    if not name and hltv_id is None:
        return None
    is_lan_int = None if is_lan is None else int(is_lan)
    if hltv_id is not None:
        row = conn.execute("SELECT id FROM events WHERE hltv_id = ?", (hltv_id,)).fetchone()
        if row:
            if is_lan is not None:
                conn.execute("UPDATE events SET is_lan = ? WHERE id = ?", (is_lan_int, row["id"]))
            return row["id"]
        cur = conn.execute(
            "INSERT INTO events (hltv_id, name, is_lan) VALUES (?, ?, ?)",
            (hltv_id, name, is_lan_int),
        )
        return cur.lastrowid

    row = conn.execute("SELECT id FROM events WHERE name = ? AND hltv_id IS NULL", (name,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute("INSERT INTO events (hltv_id, name, is_lan) VALUES (NULL, ?, ?)", (name, is_lan_int))
    return cur.lastrowid


def upsert_match(
    conn: sqlite3.Connection,
    *,
    hltv_id: int,
    match_url: str | None,
    event_id: int | None,
    unix_timestamp_ms: int | None,
    team1_id: int | None,
    team2_id: int | None,
    best_of: int | None,
    team1_score: int | None,
    team2_score: int | None,
    is_lan: bool | None,
) -> int:
    match_date = None
    if unix_timestamp_ms:
        match_date = dt.datetime.fromtimestamp(unix_timestamp_ms / 1000, tz=dt.timezone.utc).date().isoformat()

    winner_team_id = None
    if team1_score is not None and team2_score is not None and team1_score != team2_score:
        winner_team_id = team1_id if team1_score > team2_score else team2_id

    is_lan_int = None if is_lan is None else int(is_lan)
    scraped_at = dt.datetime.now(dt.timezone.utc).isoformat()

    row = conn.execute("SELECT id FROM matches WHERE hltv_id = ?", (hltv_id,)).fetchone()
    if row:
        conn.execute(
            """
            UPDATE matches SET
                match_url = ?, event_id = ?, match_date = ?, unix_timestamp_ms = ?,
                team1_id = ?, team2_id = ?, best_of = ?, team1_score = ?, team2_score = ?,
                winner_team_id = ?, is_lan = ?, scraped_at = ?
            WHERE id = ?
            """,
            (
                match_url, event_id, match_date, unix_timestamp_ms, team1_id, team2_id,
                best_of, team1_score, team2_score, winner_team_id, is_lan_int, scraped_at,
                row["id"],
            ),
        )
        return row["id"]

    cur = conn.execute(
        """
        INSERT INTO matches (
            hltv_id, match_url, event_id, match_date, unix_timestamp_ms,
            team1_id, team2_id, best_of, team1_score, team2_score,
            winner_team_id, is_lan, scraped_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            hltv_id, match_url, event_id, match_date, unix_timestamp_ms,
            team1_id, team2_id, best_of, team1_score, team2_score,
            winner_team_id, is_lan_int, scraped_at,
        ),
    )
    return cur.lastrowid


def replace_maps(conn: sqlite3.Connection, match_id: int, maps: list[dict]) -> None:
    """`maps` rows: {map_order, map_name, team1_score, team2_score, team1_id, team2_id, picked_by_team_id}."""
    conn.execute("DELETE FROM maps WHERE match_id = ?", (match_id,))
    for m in maps:
        winner_team_id = None
        s1, s2 = m.get("team1_score"), m.get("team2_score")
        if s1 is not None and s2 is not None and s1 != s2:
            winner_team_id = m.get("team1_id") if s1 > s2 else m.get("team2_id")
        conn.execute(
            """
            INSERT INTO maps (match_id, map_order, map_name, team1_score, team2_score,
                               winner_team_id, picked_by_team_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id, m["map_order"], m.get("map_name"), s1, s2,
                winner_team_id, m.get("picked_by_team_id"),
            ),
        )


def replace_roster(conn: sqlite3.Connection, match_id: int, team_id: int, player_ids: list[int]) -> None:
    conn.execute("DELETE FROM rosters WHERE match_id = ? AND team_id = ?", (match_id, team_id))
    for pid in player_ids:
        conn.execute(
            "INSERT OR IGNORE INTO rosters (match_id, team_id, player_id) VALUES (?, ?, ?)",
            (match_id, team_id, pid),
        )


def upsert_ranking_snapshot(
    conn: sqlite3.Connection, team_id: int, snapshot_date: dt.date, rank: int | None, points: int | None
) -> None:
    conn.execute(
        """
        INSERT INTO ranking_snapshots (team_id, snapshot_date, rank, points)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(team_id, snapshot_date) DO UPDATE SET rank = excluded.rank, points = excluded.points
        """,
        (team_id, snapshot_date.isoformat(), rank, points),
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


def get_scrape_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM scrape_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_scrape_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO scrape_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
