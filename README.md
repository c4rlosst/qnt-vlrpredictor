# valpredictor

Predicts Valorant matches (winner and exact map score) from [vlr.gg](https://www.vlr.gg) data.
Personal project, not betting advice.

## How we compute it

**1. One Elo rating per team.** History is replayed in date order (a series win moves a team's
rating; a loss moves it down) so a prediction never sees its own result or the future. When a
team's lineup changes, its rating is pulled back toward average — 10% per new player, capped at
50% — since its past results were earned by other players.

**2. Elo gap to a per-map win chance.** `P(team 1 wins a map) = sigmoid(slope * elo_diff / 100)`,
with no home-team term (A vs B is exactly 1 minus B vs A). `slope` is fit on past maps by weighted
logistic regression, recent maps counted more (half weight every 90 days). The plain textbook Elo
scale is `slope = 0.576`; the fitted value is lower (~0.41 on the 2025–26 season) — Elo gaps mean a
bit less in tier-1 Valorant than the textbook formula assumes.

**3. Maps to a series.** With one rating per team, every map in a series has the same win chance, so
we enumerate every way a Bo1/Bo3/Bo5 can go from that one number — which maps actually get played
(the veto) can't change the odds. The page shows the full score distribution and the chance of a
deciding map.

Predictions are always made from each team's rating *before* the match — never from an in-progress
score. For a match already underway, the page keeps showing the pre-game estimate and marks it as
such, alongside the real series score.

## Checking it

On a full season (~2,000 maps), calibrated Elo scores 55.4% accuracy and 0.6846 log-loss — barely
better than the textbook Elo scale (0.6850) and both comfortably ahead of a coin flip (0.6931).
Richer models were tried and dropped because they didn't hold up out of sample:
- a gradient-boosted model on form, head-to-head, rest days, rosters and sides tied or lost to Elo
- scaling by event stage/tier (regular season vs playoffs vs final) was significantly *worse*
  (+0.007 ± 0.003 log-loss)
- agent composition and starting side both moved accuracy by less than their own noise band
- correlating the maps within one series (a shared "which team overperforms today" shock) only
  bordered on significant (p ≈ 0.05–0.07) and isn't worth the extra machinery

So the model stays Elo-only: it's the simplest thing that isn't beaten by anything more complex.

## Limits

Tier-1 Valorant is only ~1–2k maps a year, so small effects are hard to tell from noise. Elo also
isn't cross-region aware outside international events — two teams that have never shared a bracket
can have ratings that aren't really comparable, so predictions for those matchups can diverge a lot
from the bookmakers' line. No player-level form, patch effects, or map-specific skill.
