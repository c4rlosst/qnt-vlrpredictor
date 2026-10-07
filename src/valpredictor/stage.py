"""What kind of match is it? Two coarse labels, from the event name and the
stage text vlr.gg shows on every match ("Playoffs: Lower Round 2",
"Group Stage: Week 3", ...).

event tier   2 = international (Masters / Champions / Esports World Cup)
             1 = regional VCT league (Kickoff, Stage 1/2)
             0 = qualifiers, Challengers, Ascension, anything else
stakes       0 = regular season / group stage opener
             1 = playoff or bracket match with a life in hand (upper bracket, quarter/semifinals, winners' match)
             2 = elimination: lower bracket, group-stage decider / elimination match
             3 = a final (grand, upper, lower, middle)
Unknown -> None (the model treats it as missing).
"""

from __future__ import annotations

import re

STAGE_CHOICES = {"regular": 0, "playoffs": 1, "elimination": 2, "final": 3}

_LOW_TIER = re.compile(r"qualifier|challengers|ascension|game changers|academy", re.I)
_INTERNATIONAL = re.compile(r"\bmasters\b|\bvalorant champions\b|\besports world cup\b|lock//in", re.I)
_LEAGUE = re.compile(r"\bvct\b|champions tour", re.I)

_FINAL = re.compile(r"\b(grand|upper|lower|middle)\s+final\b", re.I)
_ELIMINATION = re.compile(r"\blower\b|\bdecider\b|\belimination\b", re.I)
_BRACKET = re.compile(
    r"playoffs|\bupper\b|\bmiddle\b|quarterfinal|semifinal|winner'?s|main event|play-ins|swiss|\bstage\s*\d", re.I
)
_REGULAR = re.compile(r"group stage|week|seeding|opening", re.I)


def classify_event_tier(event_name: str | None) -> int | None:
    if not event_name:
        return None
    if _LOW_TIER.search(event_name):
        return 0
    if _INTERNATIONAL.search(event_name):
        return 2
    if _LEAGUE.search(event_name):
        return 1
    return 0


def classify_stakes(series: str | None) -> int | None:
    if not series:
        return None
    if _FINAL.search(series):
        return 3
    if _ELIMINATION.search(series):
        return 2
    if _BRACKET.search(series):
        return 1
    if _REGULAR.search(series):
        return 0
    return None
