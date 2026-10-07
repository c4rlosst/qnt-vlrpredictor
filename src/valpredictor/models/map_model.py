"""LightGBM binary classifier: P(team1 wins this map | pre-match features).

This is the single model the whole system is built around — match winner and
map-score predictions are both derived from it via `models.combinatorics`.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from valpredictor.config import load_config

FEATURE_COLUMNS = [
    "map_name",
    "best_of",
    "is_international",
    "elo_diff",
    "map_elo_diff",
    "form_5_diff",
    "form_10_diff",
    "form_20_diff",
    "map_winrate_diff",
    "rest_days_diff",
    "congestion_diff",
    "roster_stability_diff",
    "roster_continuity_diff",
    "atk1_vs_def2",
    "def1_vs_atk2",
    "round_edge_diff",
    "map_atk_bias",
    "standin_diff",
    "h2h_team1_rate",
    "h2h_n",
    "team1_pick",
]
CATEGORICAL_COLUMNS = ["map_name"]
TARGET_COLUMN = "target"


def recency_weights(match_dates: pd.Series, half_life_days: float | None) -> np.ndarray | None:
    """Exponential-decay sample weights: a row `half_life_days` older than the
    newest row counts half as much. None disables weighting (all rows equal)."""
    if not half_life_days:
        return None
    dates = pd.to_datetime(match_dates)
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age_days / float(half_life_days))


def chronological_holdout_split(df: pd.DataFrame, valid_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carves the chronologically-last `valid_frac` of match dates off `df`
    for early-stopping validation. Never a random split — a random holdout
    would leak future information back into "training" via rows from the
    same date range the model is meant to be evaluated as not having seen."""
    dates = df.sort_values("match_date")["match_date"].unique()
    n_valid_dates = max(1, int(len(dates) * valid_frac))
    if n_valid_dates >= len(dates):
        return df, df.iloc[0:0]
    valid_dates = set(dates[-n_valid_dates:])
    valid_mask = df["match_date"].isin(valid_dates)
    return df[~valid_mask], df[valid_mask]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    for c in CATEGORICAL_COLUMNS:
        X[c] = X[c].astype("category")
    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    # explicit cast: a single-row prediction frame built from a dict with None
    # values infers as object dtype, which LightGBM rejects outright — force
    # float64 (NaN-safe, and LightGBM treats NaN as "missing" natively).
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return X


def train_map_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> lgb.Booster:
    cfg = (config or load_config())["model"]
    X_train = _prep(train_df)
    y_train = train_df[TARGET_COLUMN]
    weights = recency_weights(train_df["match_date"], cfg.get("recency_half_life_days"))
    train_set = lgb.Dataset(X_train, label=y_train, weight=weights, categorical_feature=CATEGORICAL_COLUMNS)

    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = []
    if valid_df is not None and not valid_df.empty:
        X_valid = _prep(valid_df)
        y_valid = valid_df[TARGET_COLUMN]
        valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
        valid_sets.append(valid_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False))

    booster = lgb.train(
        cfg["lightgbm_params"],
        train_set,
        num_boost_round=cfg["num_boost_round"],
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def predict_proba(model: lgb.Booster, df: pd.DataFrame) -> pd.Series:
    X = _prep(df)
    preds = model.predict(X, num_iteration=getattr(model, "best_iteration", None) or None)
    return pd.Series(preds, index=df.index)


def save_model(model: lgb.Booster, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path | str) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
