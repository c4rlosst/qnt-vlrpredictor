"""Scrape the vlr.gg results list, paginated, down to a date cutoff."""

from __future__ import annotations

import datetime as dt
import logging
import re

from valpredictor.scraping.client import VLRClient
from valpredictor.scraping.parsers import ResultRow, parse_results_page

logger = logging.getLogger(__name__)

_DATE_LABEL = re.compile(r"(\w+),\s+(\w+)\s+(\d{1,2}),\s+(\d{4})")
_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}


def parse_date_label(label: str | None) -> dt.date | None:
    """'Wed, October 7, 2026' -> date(2026, 10, 7). None if it doesn't match."""
    if not label:
        return None
    m = _DATE_LABEL.search(label)
    if not m:
        return None
    month = _MONTHS.get(m.group(2).lower())
    if month is None:
        return None
    try:
        return dt.date(int(m.group(4)), month, int(m.group(3)))
    except ValueError:
        return None


def iter_results(client: VLRClient, since: dt.date, until: dt.date | None = None, refresh: bool = False):
    """Yield every ResultRow dated within [since, until], newest first.

    vlr.gg's results list is in descending date order, so pagination stops as
    soon as a whole page is older than `since`. Event filtering is the
    caller's job (see ingest.run_backfill). `refresh=True` bypasses the disk
    cache, which list pages need since new results shift every page.
    """
    until = until or dt.date.today()
    page = 1
    while True:
        html = client.get(f"/matches/results?page={page}", force_refresh=refresh)
        rows = parse_results_page(html)
        if not rows:
            logger.info("no more results at page=%d, stopping", page)
            return

        dates = [parse_date_label(r.date_label) for r in rows]
        for row, row_date in zip(rows, dates):
            if row_date is None or row_date < since or row_date > until:
                continue
            yield row

        known = [d for d in dates if d is not None]
        if known and max(known) < since:
            logger.info("reached results older than %s, stopping", since)
            return
        page += 1
