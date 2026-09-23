"""Scrape the HLTV results list, paginated, down to a date cutoff."""

from __future__ import annotations

import datetime as dt
import logging
import re

from cspredictor.scraping.client import HLTVClient
from cspredictor.scraping.parsers import ResultRow, parse_results_page

logger = logging.getLogger(__name__)

_RESULTS_PER_PAGE = 100  # HLTV's `/results?offset=N` pages 100 at a time

_DATE_LABEL = re.compile(
    r"Results for (?:(\d{1,2})\w{2}\s+(\w+)\s+(\d{4})|today|yesterday)", re.IGNORECASE
)
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
    """Best-effort parse of a results-sublist headline into a date.

    Handles "Results for 22nd September 2026", "Results for today",
    "Results for yesterday". Returns None if the format doesn't match —
    callers should treat that as "unknown date, don't use for cutoff logic".
    """
    if not label:
        return None
    if "today" in label.lower():
        return dt.date.today()
    if "yesterday" in label.lower():
        return dt.date.today() - dt.timedelta(days=1)
    m = _DATE_LABEL.search(label)
    if not m or m.group(1) is None:
        return None
    day, month_name, year = m.group(1), m.group(2), m.group(3)
    month = _MONTHS.get(month_name.lower())
    if month is None:
        return None
    try:
        return dt.date(int(year), month, int(day))
    except ValueError:
        return None


def iter_results(client: HLTVClient, since: dt.date, until: dt.date | None = None):
    """Yield ResultRow objects for matches on/after `since` (and on/before `until` if given).

    Stops paginating once a page's rows are entirely older than `since`,
    since HLTV's results list is in descending date order.
    """
    until = until or dt.date.today()
    offset = 0
    seen_any_in_range = False

    while True:
        html = client.get(f"/results?offset={offset}")
        rows = parse_results_page(html)
        if not rows:
            logger.info("no more results at offset=%d, stopping", offset)
            break

        page_dates = [parse_date_label(r.date_label) for r in rows]
        known_dates = [d for d in page_dates if d is not None]

        any_in_range = False
        for row, row_date in zip(rows, page_dates):
            if row_date is not None and row_date > until:
                continue
            if row_date is not None and row_date < since:
                continue
            any_in_range = True
            seen_any_in_range = True
            yield row

        if known_dates and max(known_dates) < since:
            logger.info("reached results older than %s, stopping pagination", since)
            break
        if not any_in_range and seen_any_in_range:
            # We've moved past the range after having been in it.
            break

        offset += _RESULTS_PER_PAGE
