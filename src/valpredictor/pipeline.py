"""The "refresh everything" pipeline behind the web UI's Refresh button:
scrape newly finished matches -> rebuild features -> retrain -> fresh state.

The CLI exposes the same steps one at a time (backfill-results, build-features,
train); this runs them back to back and reports progress as it goes.
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import pandas as pd

from valpredictor.config import load_config
from valpredictor.features.build_features import ReplayState, replay, symmetrize, to_model_matrix
from valpredictor.models.map_model import chronological_holdout_split, save_model, train_map_model
from valpredictor.scraping.client import VLRClient
from valpredictor.scraping.ingest import run_backfill

MIN_MAPS_TO_TRAIN = 60
OVERLAP_DAYS = 3  # re-check the last few days: a match can finish after its first listing

Progress = Callable[[str, str], None]


@dataclass
class RefreshResult:
    new_matches: int = 0
    total_matches: int = 0
    maps: int = 0
    trained: bool = False
    message: str = ""
    latest_match_date: str | None = None
    model: lgb.Booster | None = None
    state: ReplayState | None = None


def default_since(conn: sqlite3.Connection) -> dt.date:
    """Resume from just before the newest stored match, or from the configured
    history window when the database is empty."""
    latest = conn.execute("SELECT MAX(match_date) FROM matches").fetchone()[0]
    if latest:
        return dt.date.fromisoformat(latest) - dt.timedelta(days=OVERLAP_DAYS)
    months = int(load_config()["backfill"]["months_back"])
    return dt.date.today() - dt.timedelta(days=round(months * 30.4))


def refresh(
    conn: sqlite3.Connection,
    client: VLRClient,
    model_path: Path,
    table_path: Path,
    progress: Progress = lambda step, message: None,
    since: dt.date | None = None,
    config: dict | None = None,
) -> RefreshResult:
    cfg = config or load_config()
    include = re.compile(cfg["backfill"]["event_include"], re.I)
    exclude = re.compile(cfg["backfill"]["event_exclude"], re.I)
    since = since or default_since(conn)

    progress("scrape", f"Checking vlr.gg for matches finished since {since.isoformat()} ...")
    stats = run_backfill(
        conn, client, since, None, include, exclude,
        progress=lambda s: progress("scrape", f"Stored {s.ingested} new matches so far ..."),
    )

    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    latest = conn.execute("SELECT MAX(match_date) FROM matches").fetchone()[0]
    result = RefreshResult(new_matches=stats.ingested, total_matches=total, latest_match_date=latest)

    progress("features", f"Rebuilding features from {total} matches ...")
    state = replay(conn, cfg)  # one chronological pass yields both the training rows and the live state
    table = pd.DataFrame(state.rows)
    result.maps = len(table)
    result.state = state
    if len(table) < MIN_MAPS_TO_TRAIN:
        result.message = (
            f"Stored {stats.ingested} new matches, but only {len(table)} maps in total: "
            f"need at least {MIN_MAPS_TO_TRAIN} to train. Run `valpredictor backfill-results` for the full history."
        )
        return result

    matrix = to_model_matrix(symmetrize(table))
    table_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(table_path)

    progress("train", f"Training on {len(table)} maps ...")
    inner, valid = chronological_holdout_split(matrix)
    model = train_map_model(inner, valid_df=valid if not valid.empty else None, config=cfg)
    save_model(model, model_path)

    result.model = model
    result.trained = True
    result.message = f"Stored {stats.ingested} new matches; retrained on {len(table)} maps (data through {latest})."
    return result
