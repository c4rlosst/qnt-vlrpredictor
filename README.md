# valpredictor

Predicts Valorant matches (winner and exact map score) from [vlr.gg](https://www.vlr.gg) data.
Personal project, not betting advice.

**How it works:** a LightGBM model estimates P(team wins a map) from Elo, per-map Elo, recent form,
head-to-head, rest, roster continuity, attack/defence side strength and who picked the map. Recent
matches weigh more, and a team's Elo is pulled back toward average when its lineup changes. Series
odds and exact scores are derived from the per-map probabilities; before the veto, the model is
averaged over the current map pool.

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # git bash: source .venv/Scripts/activate
pip install -e ".[dev]"
```

## Run

```bash
valpredictor backfill-results     # first time: ~12 months of tier-1 matches, about an hour, resumable
valpredictor build-features
valpredictor train
valpredictor serve                # open http://127.0.0.1:8000
```

In the page, **Refresh data** scrapes newly finished matches, retrains and reloads the model (about a
minute). Restart `serve` after changing code.

Other commands: `predict`, `upcoming` (live bracket), `backtest`, `import-csv`, `reparse`, `scan-events`.
Run `valpredictor --help`.

## Your own data

`valpredictor import-csv my_matches.csv --dry-run` validates a CSV (one row per map played) and imports it
all-or-nothing. See `examples/maps_example.csv` for the format.

## Notes

- Tier-1 Valorant is only ~1-2k maps a year, so expect a modest edge over plain Elo at best. Check
  `valpredictor backtest` and `python scripts/compare_variants.py` before trusting the numbers.
- The side a team *starts* on shows no measurable effect (`scripts/analyze_start_side.py`); each map's
  attack/defence balance does.
- Scraping is for personal use: 3-5 s between requests, pages cached, robots.txt respected.
- Tests: `pytest` (no network needed).
