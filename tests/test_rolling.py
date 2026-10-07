import datetime as dt

import pytest

from valpredictor.features.rolling import (
    FormTracker,
    H2HTracker,
    RestAndCongestionTracker,
    RosterTracker,
    SideTracker,
    side_rounds,
)


def test_side_rounds_regulation_and_early_finish():
    assert side_rounds(13, 11, "def") == (12, 12, 12, 12)  # 24 rounds: 12 on each side
    assert side_rounds(13, 7, "def") == (8, 12, 12, 8)     # 20 rounds: only 8 after the swap
    assert side_rounds(13, 7, "atk") == (12, 8, 8, 12)
    assert side_rounds(15, 13, "atk") == (12, 12, 12, 12)  # overtime rounds are ignored


def test_side_tracker_rejects_inconsistent_counts():
    st = SideTracker()
    # NRG-T1 Lotus: NRG started defence (5 def / 8 atk), T1 started attack (7 atk / 4 def)
    assert st.update(1, 2, "Lotus", 13, 11, t1_atk_won=8, t1_def_won=5, t2_atk_won=7, t2_def_won=4, team1_first_side="def")
    assert not st.update(1, 2, "Lotus", 13, 11, t1_atk_won=9, t1_def_won=5, t2_atk_won=7, t2_def_won=4, team1_first_side="def")


def test_side_tracker_edges_and_map_bias():
    st = SideTracker(prior_rounds=48.0)
    assert st.edges(1) == (0.0, 0.0)          # unseen team = league average
    assert st.league_atk_rate == 0.5
    st.update(1, 2, "Lotus", 13, 11, 8, 5, 7, 4, "def")
    assert st.league_atk_rate == pytest.approx(15 / 24)
    atk, dfn = st.edges(1)
    assert atk == pytest.approx((8 + 48 * 15 / 24) / 60 - 15 / 24)   # shrunk toward the league rate
    assert dfn == pytest.approx((5 + 48 * 9 / 24) / 60 - 9 / 24)
    assert st.map_atk_bias("Lotus") > 0 > st.map_atk_bias("Lotus") - 0.2   # attack-friendly so far
    assert st.map_atk_bias("Haven") == pytest.approx(15 / 24 - 0.5)        # unseen map falls back to the league


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
