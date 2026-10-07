import numpy as np
import pandas as pd
import pytest

from valpredictor.models.elo_model import (
    RAW_ELO_SLOPE,
    EloModel,
    fit_elo_model,
    fit_slope,
    raw_elo_prob,
    recency_weights,
)
from valpredictor.models.evaluate import COIN_FLIP_LOG_LOSS, score, walk_forward_backtest

CONFIG = {"model": {"recency_half_life_days": 90}}


def _simulated_maps(n=1500, true_slope=0.35, seed=3) -> pd.DataFrame:
    """Maps whose winner really follows sigmoid(true_slope * gap / 100)."""
    rng = np.random.default_rng(seed)
    gap = rng.normal(0, 120, n)
    won = (rng.random(n) < 1 / (1 + np.exp(-true_slope * gap / 100))).astype(int)
    dates = pd.date_range("2026-01-01", periods=n // 5, freq="D").repeat(5)[:n].strftime("%Y-%m-%d")
    return pd.DataFrame({"match_date": dates, "elo_diff": gap, "team1_won_map": won})


def test_raw_elo_slope_is_the_textbook_scale():
    assert RAW_ELO_SLOPE == pytest.approx(0.5756, abs=1e-3)
    assert raw_elo_prob(0) == pytest.approx(0.5)
    assert raw_elo_prob(400) == pytest.approx(10 / 11)  # 400 points = 10:1 odds


def test_fit_slope_recovers_the_true_scale():
    rows = _simulated_maps()
    assert fit_slope(rows["elo_diff"].to_numpy(), rows["team1_won_map"].to_numpy()) == pytest.approx(0.35, abs=0.06)


def test_fit_slope_with_nothing_to_learn_falls_back_to_textbook():
    assert fit_slope(np.array([0.0, 0.0]), np.array([1, 0])) == RAW_ELO_SLOPE
    assert fit_slope(np.array([50.0]), np.array([1])) == RAW_ELO_SLOPE


def test_model_is_antisymmetric_and_has_no_home_advantage():
    model = EloModel(slope=0.4)
    assert model.map_prob(0.0) == pytest.approx(0.5)
    assert model.map_prob(80.0) + model.map_prob(-80.0) == pytest.approx(1.0)
    assert model.map_prob(80.0) > 0.5 > model.map_prob(-80.0)


def test_model_round_trips_through_json(tmp_path):
    model = fit_elo_model(_simulated_maps(), half_life_days=90)
    model.save(tmp_path / "sub" / "model.json")
    loaded = EloModel.load(tmp_path / "sub" / "model.json")
    assert loaded == model and loaded.trained_on_maps == 1500


def test_recency_weights_decay_with_age():
    dates = pd.Series(["2026-10-07", "2026-07-09", "2026-01-01"])
    w = recency_weights(dates, half_life_days=90)
    assert w[0] == pytest.approx(1.0)                  # newest row = full weight
    assert w[1] == pytest.approx(0.5, rel=0.02)        # ~90 days older = half
    assert w[2] < w[1] < w[0]
    assert recency_weights(dates, None) is None        # disabled


def test_score_metrics():
    won = np.array([1, 0, 1, 1])
    perfect = score(won, np.array([0.999, 0.001, 0.999, 0.999]))
    assert perfect["accuracy"] == 1.0 and perfect["log_loss"] < 0.01
    coin = score(won, np.full(4, 0.5))
    assert coin["log_loss"] == pytest.approx(COIN_FLIP_LOG_LOSS) and coin["brier"] == pytest.approx(0.25)
    assert score(np.array([]), np.array([]))["n"] == 0


def test_backtest_beats_a_coin_flip_and_keeps_row_labels():
    rows = _simulated_maps()
    shuffled = rows.sample(frac=1.0, random_state=1)  # callers' labels must survive the internal sort
    result = walk_forward_backtest(shuffled, CONFIG, n_folds=4)

    assert result["overall"]["calibrated_elo"]["log_loss"] < COIN_FLIP_LOG_LOSS
    # the data truly follows slope 0.35, so the fitted scale beats the over-confident textbook one
    assert result["overall"]["calibrated_elo"]["log_loss"] < result["overall"]["raw_elo"]["log_loss"]
    oof = result["oof"]
    assert oof.index.isin(shuffled.index).all() and not oof.index.has_duplicates
    # every out-of-fold probability sits on the row it was predicted for
    gap = shuffled.loc[oof.index, "elo_diff"].to_numpy()
    assert np.allclose(oof["p_raw"].to_numpy(), raw_elo_prob(gap))


def test_backtest_only_trains_on_the_past():
    result = walk_forward_backtest(_simulated_maps(), CONFIG, n_folds=3)
    starts = [f["test_start"] for f in result["folds"]]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_backtest_needs_enough_dates():
    tiny = pd.DataFrame({"match_date": ["2026-01-01"] * 5, "elo_diff": [1.0] * 5, "team1_won_map": [1] * 5})
    with pytest.raises(ValueError):
        walk_forward_backtest(tiny, CONFIG)
