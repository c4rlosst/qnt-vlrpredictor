"""Fetches (or re-reads from cache) a page and pretty-prints what the parser
extracted, so you can eyeball it next to the real page in a browser and fix
any selector in src/cspredictor/scraping/parsers.py that's drifted from
HLTV's current markup.

    python scripts/inspect_parse.py https://www.hltv.org/results results
    python scripts/inspect_parse.py https://www.hltv.org/matches/12345/slug match
    python scripts/inspect_parse.py https://www.hltv.org/ranking/teams/2026/january/5 ranking

You can also point it at a local HTML file instead of a URL:

    python scripts/inspect_parse.py data/raw/html/<hash>.html match --match-id 12345
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cspredictor.scraping.client import HLTVClient  # noqa: E402
from cspredictor.scraping.parsers import (  # noqa: E402
    parse_match_detail,
    parse_rankings_page,
    parse_results_page,
)


def _load_html(source: str) -> str:
    local = Path(source)
    if local.exists():
        return local.read_text(encoding="utf-8")
    return HLTVClient().get(source)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    source, page_type = sys.argv[1], sys.argv[2]
    html = _load_html(source)

    if page_type == "results":
        rows = parse_results_page(html)
        print(f"parsed {len(rows)} result rows\n")
        for r in rows[:10]:
            print(asdict(r))
        if len(rows) > 10:
            print(f"... and {len(rows) - 10} more")

    elif page_type == "match":
        match_id = 0
        if "--match-id" in sys.argv:
            match_id = int(sys.argv[sys.argv.index("--match-id") + 1])
        detail = parse_match_detail(html, hltv_match_id=match_id)
        d = asdict(detail)
        for k, v in d.items():
            print(f"{k}: {v}")

    elif page_type == "ranking":
        rows = parse_rankings_page(html)
        print(f"parsed {len(rows)} ranking rows\n")
        for r in rows[:15]:
            print(asdict(r))

    else:
        print(f"unknown page type {page_type!r} (expected results|match|ranking)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
