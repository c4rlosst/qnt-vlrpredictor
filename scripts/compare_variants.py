"""Compare model / feature variants on identical walk-forward folds.

Reports, for each variant, pooled out-of-fold accuracy and log-loss, and the
paired difference in log-loss against plain Elo with a 1-s.e. error bar
(negative = better than Elo). With a few thousand maps, differences smaller
than ~2 s.e. are noise: don't tune to them.

    python scripts/compare_variants.py
    python scripts/compare_variants.py --db data/other.db --folds 6
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from valpredictor.config import load_config  # noqa: E402
from valpredictor.features.build_features import build_map_training_table, symmetrize, to_model_matrix  # noqa: E402
from valpredictor.models import map_model  # noqa: E402
from valpredictor.models.evaluate import elo_only_proba, walk_forward_backtest  # noqa: E402
from valpredictor.storage.db import get_connection  # noqa: E402

LR_FEATURES = [
    "elo_diff", "map_elo_diff", "form_10_diff", "map_winrate_diff", "round_edge_diff",
    "atk1_vs_def2", "def1_vs_atk2", "map_atk_bias", "roster_continuity_diff", "team1_pick",
]


def logistic_predictor(half_life: float | None, C: float):
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        def prep(df):
            X = df[LR_FEATURES].copy()
            X["h2h"] = df["h2h_team1_rate"].fillna(0.5) - 0.5
            return X.fillna(0.0).to_numpy(dtype=float)

        scaler = StandardScaler().fit(prep(train))
        weights = map_model.recency_weights(train["match_date"], half_life)
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(scaler.transform(prep(train)), train["target"], sample_weight=weights)
        return model.predict_proba(scaler.transform(prep(test)))[:, 1]

    return predict


def gbm_predictor(cfg: dict):
    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        inner, valid = map_model.chronological_holdout_split(train)
        model = map_model.train_map_model(inner, valid_df=valid if not valid.empty else None, config=cfg)
        return map_model.predict_proba(model, test).to_numpy()

    return predict


def with_model_params(cfg: dict, **overrides) -> dict:
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key in out["model"]["lightgbm_params"]:
            out["model"]["lightgbm_params"][key] = value
        else:
            out["model"][key] = value
    return out


def logloss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def run(table: pd.DataFrame, cfg: dict, folds: int, label: str) -> list[dict]:
    variants = {
        "GBM (config)": gbm_predictor(cfg),
        "GBM no recency weights": gbm_predictor(with_model_params(cfg, recency_half_life_days=None)),
        "GBM regularised": gbm_predictor(with_model_params(
            cfg, num_leaves=7, min_data_in_leaf=40, learning_rate=0.03, feature_fraction=0.7, lambda_l2=10.0)),
        "Logistic (C=0.1)": logistic_predictor(cfg["model"].get("recency_half_life_days"), 0.1),
        "Logistic (C=1.0)": logistic_predictor(cfg["model"].get("recency_half_life_days"), 1.0),
    }
    rows = []
    for name, predictor in variants.items():
        captured: dict[str, pd.Series] = {}

        def capture(train, test, _p=predictor):
            preds = pd.Series(_p(train, test), index=test.index)
            captured.setdefault("p", []).append(preds)
            return preds

        walk_forward_backtest(table, cfg, n_folds=folds, predictor=capture)
        oof = pd.concat(captured["p"])
        test_rows = table.loc[oof.index]
        # one row per map (the symmetrised mirror carries no new information)
        keep = test_rows["team1_id"] < test_rows["team2_id"]
        y = test_rows.loc[keep, "target"].to_numpy(dtype=float)
        p = oof[keep].to_numpy(dtype=float)
        p_elo = elo_only_proba(test_rows.loc[keep, "elo_diff"]).to_numpy(dtype=float)
        diff = logloss_rows(y, p) - logloss_rows(y, p_elo)
        rows.append({
            "table": label, "variant": name, "maps": len(y),
            "acc": float(((p >= 0.5) == (y == 1)).mean()),
            "logloss": float(logloss_rows(y, p).mean()),
            "vs_elo": float(diff.mean()), "se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
        })
    rows.append({"table": label, "variant": "Elo only", "maps": len(y),
                 "acc": float(((p_elo >= 0.5) == (y == 1)).mean()),
                 "logloss": float(logloss_rows(y, p_elo).mean()), "vs_elo": 0.0, "se": 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    conn = get_connection(args.db)
    base = load_config()
    results = []
    for shrink in (0.0, base["features"]["elo"].get("roster_change_shrink_per_player", 0.1)):
        cfg = copy.deepcopy(base)
        cfg["features"]["elo"]["roster_change_shrink_per_player"] = shrink
        table = to_model_matrix(symmetrize(build_map_training_table(conn, cfg)))
        label = f"roster shrink {shrink:.2f}"
        print(f"running variants on table '{label}' ({len(table) // 2} maps) ...", flush=True)
        results += run(table, cfg, args.folds, label)

    out = pd.DataFrame(results)
    out["vs_elo"] = out.apply(lambda r: f"{r['vs_elo']:+.4f} +/- {r['se']:.4f}", axis=1)
    print()
    print(out[["table", "variant", "maps", "acc", "logloss", "vs_elo"]].to_string(
        index=False, formatters={"acc": "{:.3f}".format, "logloss": "{:.4f}".format}))
    print("\nvs_elo = paired log-loss difference against Elo only (negative = better, +/- 1 s.e.).")


if __name__ == "__main__":
    main()
