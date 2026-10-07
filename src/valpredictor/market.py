"""Turn the bookmaker odds vlr.gg shows on a match page into probabilities,
purely as a reference point to compare the model against (not betting advice).

Decimal odds o give an implied probability 1/o; the two sides of one line add
up to more than 100% (the bookmaker's margin), so each line is normalised
before averaging across bookmakers.
"""

from __future__ import annotations

from valpredictor.scraping.parsers import OddsLine


def implied_team1_probability(lines: list[OddsLine], kind: str) -> tuple[float, int] | None:
    """(average margin-free P(team 1 wins), number of bookmakers) for 'pre-match' or 'live' lines."""
    probs = []
    for line in lines:
        if line.kind != kind:
            continue
        a, b = 1.0 / line.team1_odds, 1.0 / line.team2_odds
        probs.append(a / (a + b))
    return (sum(probs) / len(probs), len(probs)) if probs else None
