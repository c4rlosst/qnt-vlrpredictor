"""Fetch (or re-read from cache) a vlr.gg page and pretty-print what the parser
extracted, to spot a drifted selector in src/valpredictor/scraping/parsers.py.

    python scripts/inspect_parse.py /matches/results results      # results or schedule list
    python scripts/inspect_parse.py /matches results
    python scripts/inspect_parse.py /754732/nrg-vs-t1-valorant-champions-2026-ubqf match

A local HTML file path works too:

    python scripts/inspect_parse.py tests/fixtures/match_detail.html match
"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valpredictor.scraping.client import VLRClient  # noqa: E402
from valpredictor.scraping.parsers import parse_match_detail, parse_results_page  # noqa: E402


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("results", "match"):
        print(__doc__)
        raise SystemExit(1)
    source, page_type = sys.argv[1], sys.argv[2]
    local = Path(source)
    html = local.read_text(encoding="utf-8") if local.exists() else VLRClient().get(source)

    if page_type == "results":
        rows = parse_results_page(html)
        print(f"parsed {len(rows)} rows\n")
        for r in rows[:15]:
            print(asdict(r))
    else:
        m = re.match(r"^/?(\d+)/", source)
        detail = parse_match_detail(html, int(m.group(1)) if m else 0)
        for key, value in asdict(detail).items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
