import pytest

from valpredictor.stage import classify_event_tier, classify_stakes


@pytest.mark.parametrize("event, tier", [
    ("Valorant Champions 2026", 2),
    ("Valorant Masters London 2026", 2),
    ("Esports World Cup 2026", 2),
    ("VCT 2026: EMEA Stage 2", 1),
    ("VCT 2026: Americas Kickoff", 1),
    ("Esports World Cup 2026: Pacific Qualifier", 0),
    ("Challengers 2026: LATAM North ACE Masters", 0),  # "Masters" in the name, but a Challengers event
    ("VCT 2025: Pacific Ascension", 0),
    ("Some Local Cup", 0),
    (None, None),
])
def test_event_tier(event, tier):
    assert classify_event_tier(event) == tier


@pytest.mark.parametrize("series, stakes", [
    # regular season / group openers
    ("Group Stage: Week 3", 0),
    ("Group Stage: Seeding", 0),
    ("Group Stage: Opening (A)", 0),
    # a life in hand: bracket / playoff matches
    ("Playoffs: Upper Quarterfinals", 1),
    ("Playoffs: Upper Semifinals", 1),
    ("Group Stage: Winner's (B)", 1),
    ("Main Event: Middle Round 2", 1),
    ("Swiss Stage: Round 2 (1-0)", 1),
    ("Stage 2: Upper Quarterfinals", 1),
    # elimination
    ("Playoffs: Lower Round 2", 2),
    ("Group Stage: Decider (A)", 2),
    ("Group Stage: Elimination (C)", 2),
    ("Play-Ins: Lower Round 1", 2),
    # finals (including the ones that also say "lower"/"upper")
    ("Playoffs: Grand Final", 3),
    ("Playoffs: Upper Final", 3),
    ("Playoffs: Lower Final", 3),
    ("Main Event: Middle Final", 3),
    # unknown
    (None, None),
    ("", None),
    ("Showmatch", None),
])
def test_stakes(series, stakes):
    assert classify_stakes(series) == stakes


def test_semifinals_are_not_finals():
    assert classify_stakes("Playoffs: Semifinals") == 1
