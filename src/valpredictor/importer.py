"""Import hand-collected match data from a CSV (one row per map played).

Required columns:
    match_id   unique integer you choose (use 9,000,000+ so it never clashes with vlr.gg ids)
    date       2026-10-07 or 2026-10-07T17:00 (UTC unless an offset is given)
    team1, team2, best_of (1/3/5), event
    map_order  1, 2, 3 ... within the match
    map        map name, e.g. Lotus
    score1, score2   rounds won by team1 / team2 on that map
Any other columns are ignored.

The whole file is validated before anything is written, and the import is
all-or-nothing: if any row is invalid nothing is stored.
"""

from __future__ import annotations

import csv
import datetime as dt
import difflib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from valpredictor.storage import db

REQUIRED_COLUMNS = ("match_id", "date", "team1", "team2", "best_of", "event", "map_order", "map", "score1", "score2")
VALID_BEST_OF = (1, 3, 5)


@dataclass
class _Map:
    order: int
    name: str
    score1: int
    score2: int
    line: int


@dataclass
class _Match:
    match_id: int
    when: dt.datetime
    team1: str
    team2: str
    best_of: int
    event: str
    first_line: int
    maps: list[_Map] = field(default_factory=list)


@dataclass
class ImportResult:
    dry_run: bool = False
    matches_imported: int = 0
    maps_imported: int = 0
    matches_skipped: list[int] = field(default_factory=list)
    new_teams: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _int(value: str, label: str, line: int, errors: list[str]) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"line {line}: {label} must be a whole number, got {value!r}")
        return None


def parse_csv(path: Path) -> tuple[list[_Match], list[str]]:
    errors: list[str] = []
    matches: dict[int, _Match] = {}

    with open(path, newline="", encoding="utf-8-sig") as f:  # utf-8-sig tolerates Excel's BOM
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return [], [f"missing required column(s): {', '.join(missing)}"]

        for line, raw in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in raw.items() if k is not None}
            n_before = len(errors)

            match_id = _int(row["match_id"], "match_id", line, errors)
            best_of = _int(row["best_of"], "best_of", line, errors)
            order = _int(row["map_order"], "map_order", line, errors)
            s1 = _int(row["score1"], "score1", line, errors)
            s2 = _int(row["score2"], "score2", line, errors)

            when = None
            try:
                when = dt.datetime.fromisoformat(row["date"])
                if when.tzinfo is None:
                    when = when.replace(tzinfo=dt.timezone.utc)
            except ValueError:
                errors.append(f"line {line}: date must look like 2026-10-07, got {row['date']!r}")

            for col in ("team1", "team2", "event", "map"):
                if not row[col]:
                    errors.append(f"line {line}: {col} is empty")
            if best_of is not None and best_of not in VALID_BEST_OF:
                errors.append(f"line {line}: best_of must be 1, 3 or 5, got {best_of}")
            if order is not None and order < 1:
                errors.append(f"line {line}: map_order must be 1 or higher")
            if s1 is not None and s2 is not None:
                if s1 < 0 or s2 < 0:
                    errors.append(f"line {line}: scores cannot be negative")
                elif s1 == s2:
                    errors.append(f"line {line}: a map cannot end in a tie ({s1}-{s2})")
            if len(errors) > n_before:
                continue

            match = matches.get(match_id)
            if match is None:
                match = matches[match_id] = _Match(match_id, when, row["team1"], row["team2"], best_of, row["event"], line)
            else:
                for label, a, b in (
                    ("team1", match.team1, row["team1"]), ("team2", match.team2, row["team2"]),
                    ("best_of", match.best_of, best_of), ("event", match.event, row["event"]),
                    ("date", match.when, when),
                ):
                    if a != b:
                        errors.append(
                            f"line {line}: match {match_id} has {label}={b!r} here but {a!r} on line {match.first_line}"
                        )
            match.maps.append(_Map(order, row["map"], s1, s2, line))

    for m in matches.values():
        _validate_match(m, errors)
    return list(matches.values()), errors


def _validate_match(m: _Match, errors: list[str]) -> None:
    tag = f"match {m.match_id}"
    if m.team1.lower() == m.team2.lower():
        errors.append(f"{tag}: team1 and team2 are the same team ({m.team1!r})")
    orders = [x.order for x in m.maps]
    if len(set(orders)) != len(orders):
        errors.append(f"{tag}: duplicate map_order values {sorted(orders)}")
        return
    if len(m.maps) > m.best_of:
        errors.append(f"{tag}: {len(m.maps)} maps listed but it is a Bo{m.best_of}")

    needed = m.best_of // 2 + 1
    wins1 = wins2 = 0
    decided = False
    for x in sorted(m.maps, key=lambda x: x.order):
        if decided:
            errors.append(f"line {x.line}: map {x.order} listed after the series was already decided")
        wins1 += x.score1 > x.score2
        wins2 += x.score2 > x.score1
        decided = decided or max(wins1, wins2) == needed
    if not decided:
        errors.append(
            f"{tag}: series incomplete ({wins1}-{wins2} in a Bo{m.best_of}); "
            "list every map played so the winner is known"
        )


def import_csv(
    conn: sqlite3.Connection, path: Path, dry_run: bool = False, replace: bool = False
) -> ImportResult:
    matches, errors = parse_csv(Path(path))
    result = ImportResult(dry_run=dry_run, errors=errors)
    if errors:
        return result

    known = {name.lower(): name for (name,) in conn.execute("SELECT name FROM teams")}

    def check_new_team(name: str) -> None:
        if name.lower() in known:
            return
        close = difflib.get_close_matches(name.lower(), list(known), n=1, cutoff=0.8)
        if close:
            result.warnings.append(f"new team {name!r} looks very similar to existing {known[close[0]]!r} - typo?")
        known[name.lower()] = name
        result.new_teams.append(name)

    def team_id(name: str) -> int:
        row = conn.execute("SELECT id FROM teams WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
        return row["id"] if row else db.upsert_team(conn, name, None)

    for m in matches:
        exists = conn.execute("SELECT 1 FROM matches WHERE vlr_id = ?", (m.match_id,)).fetchone()
        if exists and not replace:
            result.matches_skipped.append(m.match_id)
            continue

        for x in m.maps:
            if max(x.score1, x.score2) < 13:
                result.warnings.append(f"line {x.line}: winner has under 13 rounds ({x.score1}-{x.score2}) - typo?")
        check_new_team(m.team1)
        check_new_team(m.team2)
        result.matches_imported += 1
        result.maps_imported += len(m.maps)
        if dry_run:
            continue

        t1, t2 = team_id(m.team1), team_id(m.team2)
        maps = sorted(m.maps, key=lambda x: x.order)
        wins1 = sum(x.score1 > x.score2 for x in maps)
        match_pk = db.upsert_match(
            conn,
            vlr_id=m.match_id,
            match_url=None,
            event_id=db.upsert_event(conn, m.event, None),
            unix_timestamp_ms=int(m.when.timestamp() * 1000),
            team1_id=t1,
            team2_id=t2,
            best_of=m.best_of,
            team1_score=wins1,
            team2_score=len(maps) - wins1,
        )
        db.replace_maps(
            conn,
            match_pk,
            [
                {
                    "map_order": x.order, "map_name": x.name, "team1_score": x.score1, "team2_score": x.score2,
                    "team1_id": t1, "team2_id": t2,
                }
                for x in maps
            ],
        )

    if not dry_run:
        conn.commit()
    return result
