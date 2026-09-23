import datetime as dt

import pytest

from cspredictor.features.rolling import (
    FormTracker,
    H2HTracker,
    RestAndCongestionTracker,
    RosterTracker,
)


def test_form_tracker_no_history_is_none():
    ft = FormTracker(windows=[5, 10])
    rates = ft.win_rates(1)
    assert rates[5] is None
    assert rates[10] is None


def test_form_tracker_rolling_window():
    ft = FormTracker(windows=[3])
    for won in [True, True, False, True, False]:
        ft.update(1, won)
    # only the last 3 results should count: False, True, False -> 1/3
    assert ft.win_rates(1)[3] == 1 / 3


def test_form_tracker_map_specific_independent_of_overall():
    ft = FormTracker(windows=[5])
    ft.update(1, True, map_name="Mirage")
    ft.update(1, False, map_name="Inferno")
    assert ft.map_win_rate(1, "Mirage") == 1.0
    assert ft.map_win_rate(1, "Inferno") == 0.0
    assert ft.map_win_rate(1, "Nuke") is None


def test_h2h_no_history():
    h2h = H2HTracker()
    rate, n = h2h.win_rate(1, 2)
    assert rate is None
    assert n == 0


def test_h2h_symmetric_complement():
    h2h = H2HTracker()
    h2h.update(1, 2, a_won=True)
    h2h.update(1, 2, a_won=True)
    h2h.update(1, 2, a_won=False)
    rate_1v2, n = h2h.win_rate(1, 2)
    rate_2v1, _ = h2h.win_rate(2, 1)
    assert n == 3
    assert rate_1v2 == pytest.approx(1.0 - rate_2v1)


def test_h2h_recency_weighting_favors_recent_result():
    h2h_recent_win = H2HTracker(decay=0.5)
    h2h_recent_win.update(1, 2, a_won=False)
    h2h_recent_win.update(1, 2, a_won=False)
    h2h_recent_win.update(1, 2, a_won=True)  # most recent: team 1 won

    h2h_recent_loss = H2HTracker(decay=0.5)
    h2h_recent_loss.update(1, 2, a_won=True)
    h2h_recent_loss.update(1, 2, a_won=True)
    h2h_recent_loss.update(1, 2, a_won=False)  # most recent: team 1 lost

    rate_a, _ = h2h_recent_win.win_rate(1, 2)
    rate_b, _ = h2h_recent_loss.win_rate(1, 2)
    # both have 2-1 records overall, but recency weighting should separate them
    assert rate_a > rate_b


def test_rest_and_congestion_first_match_is_none_and_zero():
    tracker = RestAndCongestionTracker(congestion_window_days=14)
    today = dt.date(2026, 1, 15)
    assert tracker.rest_days(1, today) is None
    assert tracker.congestion(1, today) == 0


def test_rest_and_congestion_after_updates():
    tracker = RestAndCongestionTracker(congestion_window_days=14)
    tracker.update(1, dt.date(2026, 1, 1))
    tracker.update(1, dt.date(2026, 1, 5))
    current = dt.date(2026, 1, 10)
    assert tracker.rest_days(1, current) == 5  # since last match on Jan 5
    assert tracker.congestion(1, current) == 2  # both prior matches within 14 days


def test_roster_tracker_first_sighting_is_unknown():
    rt = RosterTracker()
    roster = frozenset({1, 2, 3, 4, 5})
    assert rt.is_standin_match(10, roster) is None
    assert rt.stability_days(10, dt.date(2026, 1, 1)) is None


def test_roster_tracker_detects_change():
    rt = RosterTracker()
    roster_a = frozenset({1, 2, 3, 4, 5})
    roster_b = frozenset({1, 2, 3, 4, 6})  # one stand-in
    rt.update(10, roster_a, dt.date(2026, 1, 1))
    assert rt.is_standin_match(10, roster_a) is False
    assert rt.is_standin_match(10, roster_b) is True
    rt.update(10, roster_b, dt.date(2026, 1, 8))
    assert rt.stability_days(10, dt.date(2026, 1, 15)) == 7
