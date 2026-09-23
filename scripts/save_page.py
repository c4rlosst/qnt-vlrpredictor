"""Fetches a URL through the same rate-limited, caching HLTVClient the real
scraper uses, and reports where it landed on disk. Run this on a machine
that can actually reach hltv.org (this sandboxed dev environment can't).

    python scripts/save_page.py https://www.hltv.org/results
    python scripts/save_page.py /matches/12345/some-match-slug
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cspredictor.scraping.client import HLTVClient, cache_key_for_url  # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    url_or_path = sys.argv[1]

    client = HLTVClient()
    full_url = client.full_url(url_or_path)
    html = client.get(url_or_path)
    cache_path = client.cache_dir / cache_key_for_url(full_url)

    print(f"fetched {full_url}")
    print(f"saved to {cache_path} ({len(html)} bytes)")
    print(f"live requests: {client.stats.live_requests}, cache hits: {client.stats.cache_hits}")


if __name__ == "__main__":
    main()
