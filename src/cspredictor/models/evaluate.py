"""Walk-forward backtesting: train on matches up to time T, evaluate on the
next chronological slice, roll T forward, repeat. Never a random shuffle
split — respecting match order is the whole point, since a model that peeked
at the future would look artificially good.

Reports the main LightGBM model against two baselines:
  - naive-rank: always favors the team with the better (lower) HLTV rank
  - elo-only: a logistic function of match-level Elo difference alone
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from cspredictor.config import load_config
from cspredictor.models.map_model import chronological_holdout_split, predict_proba, train_map_model

_NAIVE_CLIP = 0.9  # avoid infinite log-loss when the naive baseline is simply wrong


def elo_only_proba(elo_diff: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.power(10.0, -elo_diff / 400.0))


def naive_rank_proba(rank_diff: pd.Series) -> pd.Series:
    # rank_diff > 0 means team1 has the better (lower) rank number
    return rank_diff.apply(lambda d: _NAIVE_CLIP if pd.notna(d) and d > 0 else (1 - _NAIVE_CLIP))


def _score(y_true: pd.Series, y_prob: pd.Series) -> dict:
    mask = y_prob.notna()
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0 or y_true.nunique() < 2:
        return {"n": int(len(y_true)), "accuracy": None, "log_loss": None, "brier": None}
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
    }


def walk_forward_backtest(
    model_df: pd.DataFrame, config: dict | None = None, n_folds: int = 5, min_train_frac: float = 0.4
) -> dict:
    cfg = config or load_config()
    df = model_df.sort_values("match_date").reset_index(drop=True)
    dates = sorted(df["match_date"].unique())
    if len(dates) < 10:
        raise ValueError("not enough distinct match dates to backtest meaningfully")

    initial_idx = max(1, int(len(dates) * min_train_frac))
    remaining = dates[initial_idx:]
    if not remaining:
        raise ValueError("min_train_frac leaves no data to evaluate on")
    fold_edges = np.array_split(np.array(remaining, dtype=object), max(1, n_folds))

    fold_results = []
    oof_model, oof_elo, oof_naive, oof_y = [], [], [], []

    for i, edge in enumerate(fold_edges):
        if len(edge) == 0:
            continue
        test_start, test_end = edge[0], edge[-1]
        train_mask = df["match_date"] < test_start
        test_mask = (df["match_date"] >= test_start) & (df["match_date"] <= test_end)
        train_df, test_df = df[train_mask], df[test_mask]
        if train_df.empty or test_df.empty:
            continue

        inner_train, inner_valid = chronological_holdout_split(train_df)
        model = train_map_model(inner_train, valid_df=inner_valid if not inner_valid.empty else None, config=cfg)
        model_proba = predict_proba(model, test_df)
        elo_proba = elo_only_proba(test_df["elo_diff"])
        naive_proba = naive_rank_proba(test_df["rank_diff"])
        y = test_df["target"]

        fold_results.append(
            {
                "fold": i,
                "test_start": test_start,
                "test_end": test_end,
                "model": _score(y, model_proba),
                "elo_only": _score(y, elo_proba),
                "naive_rank": _score(y, naive_proba),
            }
        )
        oof_model.append(model_proba)
        oof_elo.append(elo_proba)
        oof_naive.append(naive_proba)
        oof_y.append(y)

    overall = {
        "model": _score(pd.concat(oof_y), pd.concat(oof_model)),
        "elo_only": _score(pd.concat(oof_y), pd.concat(oof_elo)),
        "naive_rank": _score(pd.concat(oof_y), pd.concat(oof_naive)),
    }

    return {"folds": fold_results, "overall": overall}


def calibration_table(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    mask = y_prob.notna()
    y_true, y_prob = y_true[mask], y_prob[mask]
    bins = pd.qcut(y_prob, q=min(n_bins, y_prob.nunique()), duplicates="drop")
    grouped = pd.DataFrame({"y": y_true, "p": y_prob, "bin": bins}).groupby("bin", observed=True)
    return grouped.agg(mean_predicted=("p", "mean"), actual_rate=("y", "mean"), n=("y", "size")).reset_index()
