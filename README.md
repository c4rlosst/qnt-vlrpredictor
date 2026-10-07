# valpredictor

Predicts Valorant matches (winner and exact map score) from [vlr.gg](https://www.vlr.gg) data.
Personal project, not betting advice.

## How we compute it

**1. One row per past map.** For every map played we record what each team looked like *before* it
was played. History is replayed in date order, so a row never sees its own result or the future.
- **Elo:** an overall rating plus one per map, updated after every result. The expected score is
  `1 / (1 + 10^((Rb - Ra) / 400))`.
- **Form:** win rate over the last 5, 10 and 20 maps and on that specific map, plus a recency-weighted
  head-to-head record.
- **Context:** rest days, matches in the last 14 days, who picked the map, Masters/Champions or not.
- **Rosters:** lineup continuity (how much of today's lineup appeared in the last 10). When a lineup
  changes, that team's Elo is pulled 10% per new player (max 50%) back toward average.
- **Sides:** each team's attack and defence round win rate against the league average on that side,
  the matchup (A's attack vs B's defence and the reverse), and how attacker-friendly the map is.

**2. Learn from them.** A LightGBM model predicts P(team 1 wins this map). Each row is also added with
the teams swapped, so list order can't bias it. Recent matches count more (half the weight for every
90 days of age), and the newest slice of matches is held out to decide when to stop training.

**3. Maps to series.** We enumerate every way a Bo1/Bo3/Bo5 can go. With map probabilities of 55%,
50% and 60% for a Bo3: 2-0 = 0.55 x 0.50 = 27.5%, 2-1 = 0.165 + 0.135 = 30%, so the team wins the
match 57.5% of the time. The 1-2 and 0-2 lines work the same way.

**4. Before the veto.** The maps aren't known yet, so the model is run on every map in the current pool
and averaged, weighted by how often each map was played in the last 120 days. Every prediction is
computed from both teams' sides and averaged, so A vs B is exactly 1 minus B vs A.

## Checking it

A walk-forward backtest trains on the past and tests on the next slice, scored against plain Elo.
Tier-1 Valorant is only ~1-2k maps a year, so expect a modest edge at best. The side a team merely
*starts* on has no measurable effect (-1.2% +/- 1.7% over 829 maps); each map's attack/defence
balance does.

## Limits

Maps in a series are treated as independent. No agent comps, patches or individual player form, and
the pre-veto estimate ignores veto strategy.
