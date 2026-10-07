import json

import pytest
from conftest import make_league_db

from valpredictor.features import build_features as bf
from valpredictor.models.elo_model import fit_elo_model
from valpredictor.models.predict import TeamNotFoundError, format_prediction, predict_match, result_to_json

CONFIG = {
    "features": {"elo": {"initial_rating": 1500.0, "k_factor": 32.0}},
    "model": {"recency_half_life_days": 90},
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    conn = make_league_db(tmp_path_factory.mktemp("db") / "t.db")
    state = bf.replay(conn, CONFIG)
    import pandas as pd

    return conn, fit_elo_model(pd.DataFrame(state.rows), 90), state


def test_prediction_is_exactly_antisymmetric_and_favours_the_stronger_team(trained):
    conn, model, state = trained
    ab = predict_match(conn, model, "Team 0", "Team 4", config=CONFIG, state=state)
    ba = predict_match(conn, model, "Team 4", "Team 0", config=CONFIG, state=state)
    assert ab["team1_win_prob"] + ba["team1_win_prob"] == pytest.approx(1.0)
    assert ab["p_map_team1"] + ba["p_map_team1"] == pytest.approx(1.0)
    assert ab["team1_win_prob"] > 0.5  # Team 0 was simulated much stronger than Team 4
    assert sum(ab["score_distribution"].values()) == pytest.approx(1.0)
    for (x, y), p in ab["score_distribution"].items():
        assert ba["score_distribution"][(y, x)] == pytest.approx(p)


def test_a_longer_series_favours_the_better_team_more(trained):
    conn, model, state = trained
    bo1 = predict_match(conn, model, "Team 0", "Team 4", best_of=1, config=CONFIG, state=state)
    bo3 = predict_match(conn, model, "Team 0", "Team 4", best_of=3, config=CONFIG, state=state)
    bo5 = predict_match(conn, model, "Team 0", "Team 4", best_of=5, config=CONFIG, state=state)
    assert bo1["team1_win_prob"] < bo3["team1_win_prob"] < bo5["team1_win_prob"]
    assert bo1["p_distance"] is None and bo3["p_distance"] is not None


def test_unknown_team_raises(trained):
    conn, model, state = trained
    with pytest.raises(TeamNotFoundError):
        predict_match(conn, model, "No Such Team", "Team 1", config=CONFIG, state=state)


def test_result_to_json_is_serialisable(trained):
    conn, model, state = trained
    res = predict_match(conn, model, "Team 0", "Team 1", config=CONFIG, state=state)
    payload = json.loads(json.dumps(result_to_json(res)))
    assert set(payload["score_distribution"]) == {"2-0", "2-1", "1-2", "0-2"}
    assert {"elo", "roster_continuity", "days_since_lineup_change"} <= set(payload["context"]["team1"])


def test_format_prediction_mentions_both_teams_and_the_scores(trained):
    conn, model, state = trained
    text = format_prediction(predict_match(conn, model, "Team 0", "Team 1", config=CONFIG, state=state))
    assert "Team 0 vs Team 1  (Bo3)" in text and "Team 0 2-0 Team 1" in text and "deciding map" in text
