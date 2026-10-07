"""Simulate the map veto when it hasn't happened yet.

Pre-veto, averaging the whole map pool hides the thing that decides a lot of
series: each team gets to remove its worst maps and play its best ones, so
"A is strong on Lotus, B on Haven" makes a split series more likely than an
average suggests. We play out the standard VCT veto with a simple rational
policy: a team BANS the map where its own win probability is lowest and PICKS
the map where it is highest. Which team vetoes first isn't known in advance,
so callers evaluate both orders and average them.

    Bo1: six alternating bans            -> the last map is played
    Bo3: ban, ban, pick, pick, ban, ban  -> picks are maps 1-2, the last map is the decider
    Bo5: ban, ban, pick, pick, pick, pick -> picks are maps 1-4, the last map is the decider

The standard pool is seven maps, so the pool is trimmed to the seven most played.
"""

from __future__ import annotations

POOL_SIZE = 7


def veto_sequence(best_of: int, first_team: int) -> list[tuple[str, int]]:
    """[(action, team)] with action 'ban' | 'pick' and team 1 | 2."""
    f, o = first_team, 3 - first_team
    if best_of == 1:
        return [("ban", f), ("ban", o)] * 3
    if best_of == 3:
        return [("ban", f), ("ban", o), ("pick", f), ("pick", o), ("ban", f), ("ban", o)]
    return [("ban", f), ("ban", o), ("pick", f), ("pick", o), ("pick", f), ("pick", o)]


def trim_pool(weights: dict[str, float], size: int = POOL_SIZE) -> dict[str, float]:
    top = sorted(weights, key=lambda m: -weights[m])[:size]
    return {m: weights[m] for m in top}


def simulate_veto(
    p_team1: dict[str, float], best_of: int, first_team: int
) -> tuple[list[str], dict[str, int]]:
    """Returns (maps in play order, {map: team that picked it}); the decider is not in the dict.

    `p_team1[map]` is P(team 1 wins that map) before any pick effect.
    """
    remaining = dict(p_team1)
    order: list[str] = []
    picked_by: dict[str, int] = {}
    for action, team in veto_sequence(best_of, first_team):
        if len(remaining) <= 1:
            break
        own = (lambda m: remaining[m]) if team == 1 else (lambda m: 1.0 - remaining[m])
        if action == "ban":
            chosen = min(remaining, key=own)  # remove the map where this team is weakest
        else:
            chosen = max(remaining, key=own)  # play the map where this team is strongest
            picked_by[chosen] = team
            order.append(chosen)
        del remaining[chosen]
    # whatever is left is the decider (Bo1: the single map that survives all the bans)
    if remaining:
        order.append(next(iter(remaining)))
    return order[:best_of], picked_by
