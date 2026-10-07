# valpredictor

Predicts Valorant matches (winner and exact map score) from [vlr.gg](https://www.vlr.gg) data.
Personal project, not betting advice.

## How we compute it

**1. One row per past map.** For every map played we record what each team looked like *before* it
was played. History is replayed in date order, so a row never sees its own result or the future.
- **Elo:** an overall rating plus one per map. The expected score is `1 / (1 + 10^((Rb - Ra) / 400))`.
- **Form:** win rate over the last 5, 10 and 20 maps and on that map, plus head-to-head.
- **Context:** rest days, who picked the map, the event level (international / regional league /
  other) and the stakes (regular season, playoffs, elimination, final).
- **Rosters:** lineup continuity. When a lineup changes, that team's Elo is pulled 10% per new player
  (max 50%) back toward average.
- **Sides:** each team's attack and defence round win rate against the league average, the matchup
  (A's attack vs B's defence and the reverse), and how attacker-friendly the map is.

**2. Learn from them.** A small, heavily regularised LightGBM model predicts P(team 1 wins this map).
Each row is also added with the teams swapped, and recent matches count more (half the weight per 90 days).

**3. Maps to series.** We enumerate every way a Bo1/Bo3/Bo5 can go. With independent maps at 55%, 50%
and 60%, a Bo3 is 2-0 27.5% and 2-1 30%, so a 57.5% match win. Real maps in a series are correlated
(one team's true strength is shared), so each series also gets one common strength shock (sd 0.6),
which makes sweeps a little likelier. The page shows the chance of a deciding map.

**4. The veto.** Once the veto is posted we predict from the real maps and who picked them (the page
watches for it and switches over). Before that we simulate it for both possible first bans (each team
bans its worst map and picks its best) and average the two series. Every prediction is computed from
both teams' sides, so A vs B is exactly 1 minus B vs A. The page also shows the bookmakers' pre-match
line for reference. Predictions are meant to be made before the game starts; the score is never used.

## Checking it

On a full season (about 2,000 maps) the model scores ~55% per map and **statistically ties plain Elo**
(log-loss 0.687 vs 0.685); no extra feature group beat Elo with confidence. Tested on their own:
- starting side: team that started on attack won 48.6% vs 49.8% expected (-1.2% +/- 1.7%)
- stage: favourites beat expectation by +4.1% +/- 1.8% in the regular season but -2.5% +/- 2.0% in
  playoffs; a third Bo3 map happens 42% (regular season) vs 47% (playoffs); 46.7% +/- 2.3% overall
- each map's attack/defence balance varies a lot (attackers win ~55% on Abyss, ~45% on Ascent)

Scripts: `analyze_start_side.py`, `analyze_stage.py`, `compare_variants.py`,
`fit_series_tau.py`, `check_series_calibration.py`.

## Limits

Tier-1 Valorant is only ~1-2k maps a year, so small effects can't be told from noise. The veto
simulation is a simple rational-pick heuristic, and the per-map differences it works from are weak.
No individual player form or patch effects.
