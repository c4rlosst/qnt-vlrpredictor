# cspredictor

A machine-learning pipeline that predicts CS2 match outcomes — winner **and**
map score — from historical match data scraped from [HLTV.org](https://www.hltv.org).
Personal/portfolio project, not a production betting system.

## How it works

One model does all the work: a LightGBM binary classifier estimates
**P(team1 wins a given map)** from pre-match features (Elo, per-map Elo,
recent form, head-to-head, HLTV ranking, rest days, roster stability, veto
context, ...). Match-winner probability and the full map-score distribution
(e.g. P(2-0), P(2-1)) are then *derived* from that one model by enumerating
Bo1/Bo3/Bo5 combinatorics over the per-map probabilities
(`src/cspredictor/models/combinatorics.py`) — no separate multiclass model
needed.

Feature engineering (`src/cspredictor/features/`) is a single chronological
replay over match history: every feature for a match is read from Elo/form/
roster trackers *before* that match's own result is folded in, so nothing
ever leaks the future into training data. See the module docstrings in
`build_features.py` for the exact mechanics.

```
src/cspredictor/
  scraping/     HLTV client (rate-limited, disk-cached), parsers, ingest
  storage/      SQLite schema + upsert helpers
  features/     Elo, rolling stats (form/h2h/rest/roster), the replay builder
  models/       the LightGBM map model, combinatorics, backtest, predict
  cli.py        `cspredictor <command>`
```

## ⚠️ Important: the scraper is unverified against the live site

This was built in a sandboxed dev environment with **no network access to
hltv.org** (general internet access worked; hltv.org specifically failed at
the TLS handshake, and the built-in browser's navigation to it was denied —
this looks like a network-policy block on that domain, not a transient
issue). So `src/cspredictor/scraping/parsers.py` was written against HLTV's
long-documented markup conventions, but **no selector in it has been checked
against a real HLTV page.**

Before trusting any scraped data, on a machine that *can* reach hltv.org:

```bash
python scripts/save_page.py https://www.hltv.org/results
python scripts/inspect_parse.py https://www.hltv.org/results results

python scripts/save_page.py https://www.hltv.org/matches/<id>/<slug>
python scripts/inspect_parse.py https://www.hltv.org/matches/<id>/<slug> match --match-id <id>

python scripts/save_page.py https://www.hltv.org/ranking/teams/2026/january/5
python scripts/inspect_parse.py https://www.hltv.org/ranking/teams/2026/january/5 ranking
```

`inspect_parse.py` pretty-prints exactly what the parser extracted. Open the
real page next to it and compare. If a field comes back `None`/wrong, the
selector to fix is in `parsers.py` — each field is its own small extraction
so a fix is usually a one-line change. `tests/test_parsers.py` runs against
hand-written fixture HTML (see `tests/fixtures/README.md`) and will keep
passing regardless — it only proves the parsing *logic* works, not that it
matches HLTV's current markup. Once you've fixed selectors against a real
page, consider swapping the fixtures for real (trimmed) saved HTML so the
test suite starts verifying against ground truth.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -e ".[dev]"
pytest                        # should be all green, no network needed
```

## Running the pipeline

```bash
# 1. Scrape + ingest match history (resumable — safe to Ctrl-C and re-run)
cspredictor backfill-results --since 2024-09-22
cspredictor backfill-rankings --since 2024-09-22

# 2. Replay history into the leakage-safe training table
cspredictor build-features

# 3. Train the model
cspredictor train

# 4. Walk-forward backtest against elo-only and naive-rank baselines
cspredictor backtest

# 5. Predict a matchup
cspredictor predict --team1 "Team A" --team2 "Team B" \
    --best-of 3 --maps Mirage,Inferno,Ancient
```

Every command accepts `--db <path>` (before the subcommand) to point at a
different SQLite file, e.g. `cspredictor --db data/demo.db build-features`.

### Running the full 2-year backfill

`backfill-results --since <2 years ago>` will make roughly one request per
match plus one per results-list page — likely several thousand requests at
the configured 4–7s randomized delay between requests, i.e. **several hours**.
It's deliberately conservative (see `config.yaml` → `scraping`) so as not to
hammer HLTV. It's resumable: every ingested match is committed immediately
and recorded, so if it's interrupted (or you hit `max_requests_per_run`, a
safety cap so a misconfigured run can't run away) just re-run the same
command — already-ingested matches are skipped via the disk cache and a
DB-level check. Run it in the background (`nohup` / a detached terminal /
`screen`) rather than in an interactive session.

### Responsible scraping

- Every fetched page is cached to `data/raw/html/` — re-runs and debugging
  never re-hit the server.
- Rate limiting + exponential backoff on 403/429/5xx is built into
  `HLTVClient` (`src/cspredictor/scraping/client.py`); tune it in
  `config.yaml` if needed, but don't remove the delay.
- This is for personal, non-commercial analysis. Don't redistribute the
  scraped dataset.

## Verifying the pipeline without real data

Since real HLTV data couldn't be pulled from this environment, the
feature/model/backtest/predict pipeline was validated end-to-end against a
**synthetic** match history instead:

```bash
python scripts/generate_synthetic_data.py --out data/synthetic_demo.db
cspredictor --db data/synthetic_demo.db build-features
cspredictor --db data/synthetic_demo.db train
cspredictor --db data/synthetic_demo.db backtest
cspredictor --db data/synthetic_demo.db predict --team1 "Synthetic Team 0" --team2 "Synthetic Team 5"
```

This data is simulated from a known Bradley-Terry skill model (see the
script's docstring) purely so a correctly-implemented pipeline has real
signal to recover — **it is not real CS2 data and the numbers it produces
mean nothing about actual teams.** Its purpose was catching implementation
bugs, and it caught a real one: the first backtest run showed the trained
model losing to a plain Elo-only baseline, traced to `train_map_model` never
being given a validation set, so LightGBM ran all 500 boosting rounds with
no early stopping and overfit badly (confirmed by comparing predicted vs.
true simulated map-win probabilities — predictions were wildly overconfident
in the wrong direction on out-of-sample matchups). Fixed by holding out the
chronologically-last slice of matches for early stopping in both `train` and
`backtest` (`chronological_holdout_split` in `models/map_model.py`); after
the fix, `best_iteration` came back around 27 (of 500 possible) and
predicted probabilities lined up closely with the known-true simulated ones.
The same discipline will matter even more on real data, where the training
set won't be this small relative to the feature count.

## Known limitations / good next steps

- **Roster/stand-in detection** is a simple heuristic (any single-player
  change vs. the immediately preceding match counts as a "stand-in match");
  it doesn't distinguish a one-off illness sub from a permanent roster swap.
- **`is_lan` detection** on the match page is a best-effort text search
  (`LAN` / `Online` anywhere on the page) — verify this against a real page
  before relying on it; the event page may be a more reliable source.
- **Map-specific features can be noisy for low-sample team/map pairs** — a
  team with only a handful of recorded games on a given map will have a
  volatile `map_winrate`/`map_elo`. Worth shrinking these toward the team's
  overall Elo when the per-map sample size is small (e.g. a Bayesian/
  empirical-Bayes prior, or a simple sample-size-weighted blend) — not
  implemented in this version.
- **Odds/betting-market data is out of scope for v1** — HLTV doesn't expose
  it cleanly. Without it there's no ROI/CLV evaluation, only accuracy/
  log-loss/calibration against the naive baselines.
- **Independence assumption in combinatorics.py**: per-map probabilities are
  treated as independent given their features — no within-match momentum
  modeling (e.g. a team that just lost map 1 playing worse/better on map 2
  beyond what the map-level Elo update already captures).
- **Pre-veto prediction** (when the maps aren't known yet) falls back to a
  single map-agnostic probability repeated across all maps — a real map-pool
  model (weighted by each team's likely veto behavior) would be more
  accurate.
