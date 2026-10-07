"""Predict the live/upcoming matches of an event straight from vlr.gg's
schedule page — shared by the `upcoming` CLI command and the web UI."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import lightgbm as lgb

from valpredictor.features.build_features import ReplayState, replay
from valpredictor.market import implied_team1_probability
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
    veto_posted: bool = False  # True -> predicted from the actual maps and picks, False -> simulated veto


def veto_to_maps(detail: MatchDetail) -> tuple[list[str], dict[str, str]]:
    """Ordered play list + {map: picker's team name} from the veto note. Empty
    when the veto is still in progress (fewer maps than the series length).
    The veto text names teams by tag ('PRX'); those are resolved to full names."""
    names = {1: detail.team1_name, 2: detail.team2_name}
    maps, picks = [], {}
    for step in detail.veto:
        if step.action == "pick":
            maps.append(step.map_name)
            if step.team:
                picks[step.map_name] = names.get(detail.veto_actors.get(step.team)) or step.team
    maps += [s.map_name for s in detail.veto if s.action == "remains"]
    return (maps, picks) if detail.best_of and len(maps) == detail.best_of else ([], {})


def pre_match_market(detail: MatchDetail) -> dict | None:
    """The bookmakers' pre-match line as a margin-free probability, to compare the model with."""
    line = implied_team1_probability(detail.odds, "pre-match")
    return None if line is None else {"team1": line[0], "team2": 1.0 - line[0], "books": line[1]}


def predict_upcoming(
    conn: sqlite3.Connection,
    client: VLRClient,
    model: lgb.Booster,
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
        maps, picks = veto_to_maps(detail)
        try:
            result = predict_match(
                conn, model, r.team1_name, r.team2_name, best_of=detail.best_of or 3,
                maps=maps or None, picks=picks or None, is_international=detail.is_international, state=state,
            )
        except TeamNotFoundError as exc:
            out.append(UpcomingPrediction(row=r, detail=detail, note=str(exc)))
            continue
        out.append(
            UpcomingPrediction(
                row=r, detail=detail, result=result, market=pre_match_market(detail), veto_posted=bool(maps)
            )
        )
    return out
