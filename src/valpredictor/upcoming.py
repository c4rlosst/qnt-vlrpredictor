"""Predict the live/upcoming matches of an event straight from vlr.gg's
schedule page — shared by the `upcoming` CLI command and the web UI.

The prediction needs only the two teams and the series length; the match page
is fetched for the length and for the bookmakers' pre-match line (shown for
reference). Nothing about a game in progress is used.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from valpredictor.features.build_features import ReplayState, replay
from valpredictor.market import implied_team1_probability
from valpredictor.models.elo_model import EloModel
from valpredictor.models.predict import TeamNotFoundError, predict_match
from valpredictor.scraping.client import VLRClient
from valpredictor.scraping.parsers import MatchDetail, ResultRow, parse_match_detail, parse_results_page


@dataclass
class UpcomingPrediction:
    row: ResultRow
    detail: MatchDetail | None = None
    result: dict | None = None
    note: str | None = None  # why there is no prediction (TBD teams, unknown team, ...)
    market: dict | None = None  # bookmaker pre-match line, for reference only


def pre_match_market(detail: MatchDetail) -> dict | None:
    """The bookmakers' pre-match line as a margin-free probability, to compare the model with."""
    line = implied_team1_probability(detail.odds, "pre-match")
    return None if line is None else {"team1": line[0], "team2": 1.0 - line[0], "books": line[1]}


def predict_upcoming(
    conn: sqlite3.Connection,
    client: VLRClient,
    model: EloModel,
    event_pattern: str = "Champions",
    state: ReplayState | None = None,
) -> list[UpcomingPrediction]:
    pattern = re.compile(event_pattern, re.I)
    rows = [
        r
        for r in parse_results_page(client.get("/matches", force_refresh=True))
        if r.status in ("upcoming", "live") and pattern.search(r.event_name or "")
    ]
    state = state or replay(conn)

    out: list[UpcomingPrediction] = []
    for r in rows:
        if not r.team1_name or not r.team2_name or "TBD" in (r.team1_name, r.team2_name):
            out.append(UpcomingPrediction(row=r, note="teams not decided yet"))
            continue
        detail = parse_match_detail(client.get(r.match_url, force_refresh=True), r.vlr_match_id)
        try:
            result = predict_match(conn, model, r.team1_name, r.team2_name, best_of=detail.best_of or 3, state=state)
        except TeamNotFoundError as exc:
            out.append(UpcomingPrediction(row=r, detail=detail, note=str(exc)))
            continue
        out.append(UpcomingPrediction(row=r, detail=detail, result=result, market=pre_match_market(detail)))
    return out
