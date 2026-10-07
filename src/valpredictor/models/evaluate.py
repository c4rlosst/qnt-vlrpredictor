"""Walk-forward backtest: fit on matches up to time T, score the next
chronological slice, roll T forward, repeat. Never a random shuffle split;
a model that peeked at the future would look artificially good.

Scores the fitted (calibrated) Elo model against the textbook Elo expectation,
per map. A coin flip scores a log-loss of 0.6931.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from valpredictor.config import load_config
from valpredictor.models.elo_model import fit_elo_model, raw_elo_prob, sigmoid

COIN_FLIP_LOG_LOSS = float(np.log(2))


def score(won: np.ndarray, prob: np.ndarray) -> dict:
    won = np.asarray(won, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    if len(won) == 0:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None}
    return {
        "n": int(len(won)),
        "accuracy": float(((p >= 0.5) == (won == 1)).mean()),
        "log_loss": float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
        "brier": float(((p - won) ** 2).mean()),
    }


def walk_forward_backtest(
    rows: pd.DataFrame, config: dict | None = None, n_folds: int = 5, min_train_frac: float = 0.4
) -> dict:
    """`rows` is the map table from `build_map_table`. Returns per-fold and pooled scores plus
    the out-of-fold probabilities (`oof`, indexed by the labels of `rows`, so they join back exactly)."""
    cfg = config or load_config()
    half_life = cfg["model"].get("recency_half_life_days")
    df = rows.sort_values("match_date", kind="stable")  # keeps the caller's row labels
    dates = sorted(df["match_date"].unique())
    if len(dates) < 10:
        raise ValueError("not enough distinct match dates to backtest meaningfully")

    remaining = dates[max(1, int(len(dates) * min_train_frac)):]
    if not remaining:
        raise ValueError("min_train_frac leaves no data to evaluate on")

    folds, oof_parts = [], []
    for i, edge in enumerate(np.array_split(np.array(remaining, dtype=object), max(1, n_folds))):
        if len(edge) == 0:
            continue
        start, end = edge[0], edge[-1]
        train = df[df["match_date"] < start]
        test = df[(df["match_date"] >= start) & (df["match_date"] <= end)]
        if train.empty or test.empty:
            continue
        model = fit_elo_model(train, half_life)
        p_cal = sigmoid(model.slope * test["elo_diff"].to_numpy() / 100.0)
        p_raw = raw_elo_prob(test["elo_diff"].to_numpy())
        won = test["team1_won_map"].to_numpy()
        folds.append(
            {
                "fold": i, "test_start": start, "test_end": end, "slope": model.slope,
                "calibrated_elo": score(won, p_cal), "raw_elo": score(won, p_raw),
            }
        )
        oof_parts.append(pd.DataFrame({"p_calibrated": p_cal, "p_raw": p_raw}, index=test.index))

    oof = pd.concat(oof_parts)
    won = df.loc[oof.index, "team1_won_map"].to_numpy()
    overall = {
        "calibrated_elo": score(won, oof["p_calibrated"].to_numpy()),
        "raw_elo": score(won, oof["p_raw"].to_numpy()),
    }
    return {"folds": folds, "overall": overall, "oof": oof}
