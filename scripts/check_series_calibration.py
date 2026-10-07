"""Do our series predictions match how often real Bo3s go to a deciding map?

A Bo3 needs a third map exactly when maps 1 and 2 are split, so from the
out-of-fold map probabilities p1, p2:  P(decider) = p1(1-p2) + (1-p1)p2.
This compares that with how often real Bo3s in the database went to three maps.
If the model is overconfident, it predicts too few deciders (too many sweeps).

    python scripts/check_series_calibration.py [--db data/x.db]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from valpredictor.config import load_config  # noqa: E402
from valpredictor.features.build_features import build_map_training_table, symmetrize, to_model_matrix  # noqa: E402
from valpredictor.models import map_model  # noqa: E402
from valpredictor.models.evaluate import walk_forward_backtest  # noqa: E402
from valpredictor.storage.db import get_connection  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    conn = get_connection(args.db)
    cfg = load_config()

    table = to_model_matrix(symmetrize(build_map_training_table(conn, cfg)))
    captured: list[pd.Series] = []

    def predictor(train: pd.DataFrame, test: pd.DataFrame):
        inner, valid = map_model.chronological_holdout_split(train)
        model = map_model.train_map_model(inner, valid_df=valid if not valid.empty else None, config=cfg)
        preds = map_model.predict_proba(model, test)
        captured.append(preds)
        return preds

    walk_forward_backtest(table, cfg, n_folds=args.folds, predictor=predictor)
    oof = pd.concat(captured)
    rows = table.loc[oof.index].assign(p=oof)
    rows = rows[(rows["team1_id"] < rows["team2_id"]) & (rows["best_of"] == 3)]  # one orientation per map

    maps_per_match = pd.read_sql_query("SELECT match_id, COUNT(*) AS n FROM maps GROUP BY match_id", conn).set_index("match_id")["n"]
    first = rows[rows["map_order"] == 1].set_index("match_id")["p"]
    second = rows[rows["map_order"] == 2].set_index("match_id")["p"]
    both = pd.concat([first.rename("p1"), second.rename("p2")], axis=1, join="inner")
    both["went_3"] = (maps_per_match.reindex(both.index) == 3).astype(float)
    both["pred"] = both["p1"] * (1 - both["p2"]) + (1 - both["p1"]) * both["p2"]

    n = len(both)
    actual, predicted = both["went_3"].mean(), both["pred"].mean()
    se = both["went_3"].std(ddof=1) / math.sqrt(n)
    print(f"Bo3 series in the out-of-fold period: {n}")
    print(f"real series that needed a 3rd map:    {actual:.1%}  (+/- {se:.1%})")
    print(f"model-implied probability of that:    {predicted:.1%}")
    print(f"a coin-flip model would imply:        50.0%")
    gap = predicted - actual
    if abs(gap) < 2 * se:
        print(f"=> consistent (gap {gap:+.1%} is within noise): sweeps are not over-predicted")
    else:
        print(f"=> gap {gap:+.1%}: the model {'under' if gap < 0 else 'over'}-predicts 3-map series")

    p = both["pred"].clip(1e-6, 1 - 1e-6)
    y = both["went_3"]
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    print(f"log-loss predicting 'goes to a 3rd map': model {ll:.4f} vs coin flip {math.log(2):.4f}")


if __name__ == "__main__":
    main()
