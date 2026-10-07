"""Fetch a vlr.gg URL through the same rate-limited, caching client the real
scraper uses, and report where it was saved.

    python scripts/save_page.py /matches/results
    python scripts/save_page.py /754732/nrg-vs-t1-valorant-champions-2026-ubqf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valpredictor.scraping.client import VLRClient, cache_key_for_url  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    client = VLRClient()
    full_url = client.full_url(sys.argv[1])
    html = client.get(sys.argv[1])
    print(f"fetched {full_url}")
    print(f"saved to {client.cache_dir / cache_key_for_url(full_url)} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
