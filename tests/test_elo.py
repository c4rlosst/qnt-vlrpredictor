from valpredictor.features.elo import EloTracker, expected_score


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) > 0.5
    assert abs(expected_score(1600, 1400) + expected_score(1400, 1600) - 1.0) < 1e-9


def test_equal_ratings_start_at_initial():
    tracker = EloTracker(initial_rating=1500.0)
    assert tracker.rating(1) == 1500.0
    assert tracker.rating(2) == 1500.0


def test_winner_gains_loser_loses():
    tracker = EloTracker(initial_rating=1500.0, k_factor=32.0)
    r1_before, r2_before = tracker.rating(1), tracker.rating(2)
    new1, new2 = tracker.update(1, 2, a_won=True)
    assert new1 > r1_before
    assert new2 < r2_before


def test_zero_sum_update():
    tracker = EloTracker(initial_rating=1500.0, k_factor=32.0)
    r1_before, r2_before = tracker.rating(1), tracker.rating(2)
    new1, new2 = tracker.update(1, 2, a_won=True)
    delta1 = new1 - r1_before
    delta2 = new2 - r2_before
    assert abs(delta1 + delta2) < 1e-9


def test_upset_moves_rating_more_than_expected_win():
    strong = EloTracker(initial_rating=1500.0, k_factor=32.0)
    weak = EloTracker(initial_rating=1500.0, k_factor=32.0)
    # engineer a rating gap: team 1 strong (1800), team 2 weak (1200)
    for t in (strong, weak):
        t._ratings[1] = 1800.0
        t._ratings[2] = 1200.0

    # scenario A: favorite (1) wins as expected
    new1_a, _ = strong.update(1, 2, a_won=True)
    gain_expected_win = new1_a - 1800.0

    # scenario B: underdog (2) wins — a bigger upset
    _, new2_b = weak.update(1, 2, a_won=False)
    gain_upset = new2_b - 1200.0

    assert gain_upset > gain_expected_win > 0


def test_regress_pulls_rating_toward_the_start():
    tracker = EloTracker(initial_rating=1500.0)
    tracker._ratings[1] = 1700.0
    tracker._ratings[2] = 1300.0
    assert tracker.regress(1, 0.25) == 1650.0
    assert tracker.regress(2, 0.5) == 1400.0
    assert tracker.regress(1, 0.0) == 1650.0   # no change
    assert tracker.regress(1, 5.0) == 1500.0   # clamped to a full reset
    assert tracker.regress(99, 0.5) == 1500.0  # unseen team stays at the start
