# Fixtures

`results_page.html`, `match_detail.html`, and `rankings_page.html` in this
directory are **hand-written approximations** of HLTV's page structure, built
from documented conventions since this dev environment has no network access
to hltv.org to save a real page.

They verify the *parser's traversal and typing logic* (nested selectors,
score extraction, veto-line regex, date-label regex, etc.) — not that the
selectors match HLTV's actual current markup. Before trusting scraped data:

1. Run `python scripts/save_page.py <url>` on a machine that can reach
   hltv.org, to save a real page into `data/raw/html/`.
2. Run `python scripts/inspect_parse.py <path-or-url> results|match|ranking`
   to see what the parser extracts from it.
3. If fields come back `None` that shouldn't, fix the selector in
   `src/cspredictor/scraping/parsers.py` and re-run step 2 until it looks
   right. Ideally, also replace these synthetic fixtures with the real saved
   HTML (trimmed to the relevant section) so this test suite starts
   verifying against ground truth.
