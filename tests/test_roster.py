import datetime as dt

import pytest

from valpredictor.features.roster import RosterTracker


def test_roster_continuity_and_players_changed():
    rt = RosterTracker(history=3)
    core = frozenset({1, 2, 3, 4, 5})
    assert rt.continuity(10, core) is None  # no history yet
    assert rt.players_changed(10, core) is None
    for day in (1, 2, 3):
        rt.update(10, core, dt.date(2026, 1, day))
    assert rt.continuity(10, core) == pytest.approx(1.0)
    assert rt.players_changed(10, core) == 0

    rebuilt = frozenset({1, 2, 3, 6, 7})  # two new players
    assert rt.players_changed(10, rebuilt) == 2
    assert rt.continuity(10, rebuilt) == pytest.approx(3 / 5)

    for day in (4, 5, 6):  # the new lineup plays enough that history is all new
        rt.update(10, rebuilt, dt.date(2026, 1, day))
    assert rt.continuity(10, rebuilt) == pytest.approx(1.0)
    assert rt.continuity(10, core) == pytest.approx(3 / 5)


def test_first_sighting_is_unknown():
    rt = RosterTracker()
    assert rt.last_roster(10) == frozenset()
    assert rt.stability_days(10, dt.date(2026, 1, 1)) is None


def test_stability_days_counts_from_the_last_change():
    rt = RosterTracker()
    rt.update(10, frozenset({1, 2, 3, 4, 5}), dt.date(2026, 1, 1))
    rt.update(10, frozenset({1, 2, 3, 4, 6}), dt.date(2026, 1, 8))  # one stand-in
    rt.update(10, frozenset({1, 2, 3, 4, 6}), dt.date(2026, 1, 9))  # same lineup again: not a change
    assert rt.stability_days(10, dt.date(2026, 1, 15)) == 7


def test_an_empty_roster_is_ignored():
    rt = RosterTracker()
    rt.update(10, frozenset({1, 2, 3, 4, 5}), dt.date(2026, 1, 1))
    rt.update(10, frozenset(), dt.date(2026, 1, 8))  # lineup data missing for that match
    assert rt.last_roster(10) == frozenset({1, 2, 3, 4, 5})
    assert rt.players_changed(10, frozenset()) is None
