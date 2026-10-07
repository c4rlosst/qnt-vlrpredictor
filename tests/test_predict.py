import datetime as dt
import random

import pytest

from valpredictor.features import build_features as bf
from valpredictor.models.map_model import train_map_model, chronological_holdout_split
from valpredictor.models.predict import (
    TeamNotFoundError,
    active_map_pool,
    predict_match,
    result_to_json,
)
from valpredictor.scraping.parsers import MatchDetail, VetoStep
from valpredictor.storage import db
from valpredictor.upcoming import veto_to_maps

MAPS = ["Ascent", "Bind", "Haven", "Lotus"]
CONFIG = {
    "features": {
        "elo": {"initial_rating": 1500.0, "k_factor": 32.0, "map_k_factor": 24.0},
        "rolling_windows": [5, 10, 20],
        "recent_days_congestion": 14,
    },
    "model": {
        "lightgbm_params": {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.1,
                            "num_leaves": 7, "min_data_in_leaf": 5, "verbose": -1},
        "num_boost_round": 40,
        "early_stopping_rounds": 10,
    },
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    conn = db.get_connection(tmp_path_factory.mktemp("db") / "t.db")
    rng = random.Random(7)
    skills = [1.2, 0.6, 0.0, -0.6, -1.2, 0.3]
    teams = [db.upsert_team(conn, f"Team {i}", vlr_id=100 + i) for i in range(len(skills))]
    start = dt.date.today() - dt.timedelta(days=90)
    for n in range(120):
        a, b = rng.sample(range(len(teams)), 2)
        day = start + dt.timedelta(days=n * 3 // 4)
        mid = db.upsert_match(
            conn, vlr_id=n + 1, match_url=f"/{n + 1}/x", event_id=None,
            unix_timestamp_ms=int(dt.datetime.combine(day, dt.time(12), tzinfo=dt.timezone.utc).timestamp() * 1000),
            team1_id=teams[a], team2_id=teams[b], best_of=3, team1_score=None, team2_score=None,
            is_international=False,
        )
        rows, wa, wb = [], 0, 0
        for order, name in enumerate(rng.sample(MAPS, 3), start=1):
            if 2 in (wa, wb):
                break
            a_wins = rng.random() < 1 / (1 + 2.718 ** -(skills[a] - skills[b]))
            rows.append({"map_order": order, "map_name": name, "team1_score": 13 if a_wins else 8,
                         "team2_score": 8 if a_wins else 13, "team1_id": teams[a], "team2_id": teams[b],
                         "picked_by_team_id": teams[a] if order == 1 else None})
            wa, wb = wa + a_wins, wb + (not a_wins)
        db.replace_maps(conn, mid, rows)
    conn.commit()

    matrix = bf.to_model_matrix(bf.symmetrize(bf.build_map_training_table(conn, CONFIG)))
    train, valid = chronological_holdout_split(matrix)
    model = train_map_model(train, valid_df=valid, config=CONFIG)
    return conn, model, bf.replay(conn, CONFIG)


def test_recency_weights_decay_with_age():
    import pandas as pd
    from valpredictor.models.map_model import recency_weights

    dates = pd.Series(["2026-10-07", "2026-07-09", "2026-01-01"])
    w = recency_weights(dates, half_life_days=90)
    assert w[0] == pytest.approx(1.0)                  # newest row = full weight
    assert w[1] == pytest.approx(0.5, rel=0.02)        # ~90 days older = half
    assert w[2] < w[1] < w[0]
    assert recency_weights(dates, None) is None        # disabled


def test_prediction_is_exactly_antisymmetric(trained):
    conn, model, state = trained
    ab = predict_match(conn, model, "Team 0", "Team 4", maps=["Ascent", "Bind", "Haven"], config=CONFIG, state=state)
    ba = predict_match(conn, model, "Team 4", "Team 0", maps=["Ascent", "Bind", "Haven"], config=CONFIG, state=state)
    assert ab["team1_win_prob"] + ba["team1_win_prob"] == pytest.approx(1.0)
    assert ab["team1_win_prob"] > 0.5  # the stronger team is favoured


def test_pre_veto_uses_map_pool_and_is_antisymmetric(trained):
    conn, model, state = trained
    ab = predict_match(conn, model, "Team 1", "Team 3", config=CONFIG, state=state)
    ba = predict_match(conn, model, "Team 3", "Team 1", config=CONFIG, state=state)
    assert ab["mode"].startswith("pre-veto")
    assert set(ab["pool"]) == set(MAPS)
    assert sum(v[0] for v in ab["pool"].values()) == pytest.approx(1.0)
    assert ab["team1_win_prob"] + ba["team1_win_prob"] == pytest.approx(1.0)
    assert sum(ab["score_distribution"].values()) == pytest.approx(1.0)


def test_unknown_team_raises(trained):
    conn, model, state = trained
    with pytest.raises(TeamNotFoundError):
        predict_match(conn, model, "No Such Team", "Team 1", config=CONFIG, state=state)


def test_result_to_json_is_serialisable(trained):
    import json

    conn, model, state = trained
    res = predict_match(conn, model, "Team 0", "Team 1", config=CONFIG, state=state)
    payload = json.loads(json.dumps(result_to_json(res)))
    assert set(payload["score_distribution"]) == {"2-0", "2-1", "1-2", "0-2"}


def test_active_map_pool_weights_sum_to_one(trained):
    conn, _, _ = trained
    pool = active_map_pool(conn)
    assert pool and sum(pool.values()) == pytest.approx(1.0)


def _detail(veto, best_of):
    return MatchDetail(vlr_match_id=1, best_of=best_of, veto=veto)


def test_veto_to_maps_complete_bo3():
    veto = [
        VetoStep("T1", "ban", "Split"), VetoStep("NRG", "ban", "Sunset"),
        VetoStep("T1", "pick", "Lotus"), VetoStep("NRG", "pick", "Summit"),
        VetoStep("T1", "ban", "Haven"), VetoStep("NRG", "ban", "Ascent"),
        VetoStep(None, "remains", "Abyss"),
    ]
    maps, picks = veto_to_maps(_detail(veto, 3))
    assert maps == ["Lotus", "Summit", "Abyss"]
    assert picks == {"Lotus": "T1", "Summit": "NRG"}


def test_veto_to_maps_incomplete_or_missing_veto():
    assert veto_to_maps(_detail([VetoStep("T1", "ban", "Split")], 3)) == ([], {})
    assert veto_to_maps(_detail([], 3)) == ([], {})
