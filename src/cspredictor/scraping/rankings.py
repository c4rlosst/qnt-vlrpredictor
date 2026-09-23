"""Fetch HLTV world-ranking snapshots over time.

HLTV publishes a new ranking roughly weekly (historically on Mondays). We
step back in weekly increments over the requested window and fetch each
dated snapshot page; missing weeks (holidays, off-weeks) will 404 upstream
and are simply skipped.
"""

from __future__ import annotations

import datetime as dt
import logging

from cspredictor.scraping.client import FetchError, HLTVClient
from cspredictor.scraping.parsers import RankingRow, parse_rankings_page

logger = logging.getLogger(__name__)


def iter_ranking_snapshots(client: HLTVClient, since: dt.date, until: dt.date | None = None):
    """Yield (snapshot_date, list[RankingRow]) for each weekly ranking in range."""
    until = until or dt.date.today()
    # anchor to the most recent Monday on/before `until`
    cursor = until - dt.timedelta(days=until.weekday())

    while cursor >= since:
        path = f"/ranking/teams/{cursor.year}/{cursor.strftime('%B').lower()}/{cursor.day}"
        try:
            html = client.get(path)
        except FetchError as exc:
            logger.info("no ranking snapshot for %s (%s), skipping", cursor, exc)
            cursor -= dt.timedelta(days=7)
            continue

        rows = parse_rankings_page(html)
        if rows:
            yield cursor, rows
        else:
            logger.info("empty ranking parse for %s, skipping", cursor)

        cursor -= dt.timedelta(days=7)
