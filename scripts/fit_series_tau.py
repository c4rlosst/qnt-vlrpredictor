"""Fit how strongly the maps of a series are correlated (the shared-strength shock `tau`).

Maps in a series share the teams' true strength, so a team that wins map 1 is
more likely to be the stronger side. The model's per-map probabilities are
shifted by one common logit shock ~ N(0, tau^2) per series (marginals kept
exactly as predicted); this picks tau by maximum likelihood over real series,
using out-of-fold predictions only, and checks how often Bo3s reach a third map.

    python scripts/fit_series_tau.py [--db data/x.db]
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
from valpredictor.models.combinatorics import _NODES, _WEIGHTS, _marginal_preserving_logits, _sigmoid  # noqa: E402
from valpredictor.models.evaluate import walk_forward_backtest  # noqa: E402
from valpredictor.storage.db import get_connection  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    conn, cfg = get_connection(args.db), load_config()

    table = to_model_matrix(symmetrize(build_map_training_table(conn, cfg)))
    captured: list[pd.Series] = []

    def predictor(train, test):
        inner, valid = map_model.chronological_holdout_split(train)
        model = map_model.train_map_model(inner, valid_df=valid if not valid.empty else None, config=cfg)
        preds = map_model.predict_proba(model, test)
        captured.append(preds)
        return preds

    walk_forward_backtest(table, cfg, n_folds=args.folds, predictor=predictor)
    oof = pd.concat(captured)
    rows = table.loc[oof.index].assign(p=oof)  # rows keep the table's labels, so this join is exact
    rows = rows[rows["best_of"].isin([3, 5])]
    # one orientation per series, chosen by match id (NOT by team id, which correlates with strength)
    rows = rows[(rows["team1_id"] < rows["team2_id"]) == (rows["match_id"] % 2 == 0)]
    series = [(g["p"].to_numpy(), g["target"].to_numpy()) for _, g in rows.sort_values("map_order").groupby("match_id")]
    print(f"out-of-fold series: {len(series)}  ({len(rows)} maps); map win rate of the listed team: {rows['target'].mean():.3f}")

    all_p = np.concatenate([s[0] for s in series])
    splits = np.cumsum([len(s[0]) for s in series])[:-1]

    def loglik(tau: float) -> float:
        centres = np.split(_marginal_preserving_logits(all_p, tau) if tau > 0 else np.log(np.clip(all_p, 1e-4, 1 - 1e-4) / (1 - np.clip(all_p, 1e-4, 1 - 1e-4))), splits)
        total = 0.0
        for (p, y), c in zip(series, centres):
            pp = _sigmoid(c[None, :] + tau * _NODES[:, None])
            lik = np.prod(np.where(y[None, :] == 1, pp, 1 - pp), axis=1)
            total += math.log(float(_WEIGHTS @ lik))
        return total

    base = loglik(0.0)
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    results = [(t, loglik(t)) for t in grid]
    print(f"\n{'tau':>5s} {'log-lik':>10s} {'vs tau=0':>9s}")
    for t, ll in results:
        print(f"{t:5.2f} {ll:10.2f} {ll - base:+9.2f}")
    best_tau, best_ll = max(results, key=lambda r: r[1])
    print(f"\nbest tau ~ {best_tau}; log-likelihood gain over independent maps {best_ll - base:+.2f} "
          f"(likelihood-ratio statistic {2 * (best_ll - base):.1f}; > 2.7 is p < 0.05 one-sided)")

    # how often did real Bo3s need a 3rd map, vs what each setting implies?
    bo3 = rows[rows["best_of"] == 3]
    n_maps = pd.read_sql_query("SELECT match_id, COUNT(*) AS n FROM maps GROUP BY match_id", conn).set_index("match_id")["n"]
    p1 = bo3[bo3["map_order"] == 1].set_index("match_id")["p"]
    p2 = bo3[bo3["map_order"] == 2].set_index("match_id")["p"]
    both = pd.concat([p1.rename("p1"), p2.rename("p2")], axis=1, join="inner")
    went3 = (n_maps.reindex(both.index) == 3).astype(float)
    se = went3.std(ddof=1) / math.sqrt(len(went3))
    print(f"\nBo3 series: {len(both)}; real share that needed a 3rd map: {went3.mean():.1%} (+/- {se:.1%})")
    for tau in sorted({0.0, 0.4, 0.6, best_tau}):
        c1 = _marginal_preserving_logits(both["p1"].to_numpy(), tau) if tau > 0 else np.log(both["p1"] / (1 - both["p1"])).to_numpy()
        c2 = _marginal_preserving_logits(both["p2"].to_numpy(), tau) if tau > 0 else np.log(both["p2"] / (1 - both["p2"])).to_numpy()
        a, b = _sigmoid(c1[:, None] + tau * _NODES[None, :]), _sigmoid(c2[:, None] + tau * _NODES[None, :])
        implied = ((a * (1 - b) + (1 - a) * b) @ _WEIGHTS).mean()
        print(f"  tau={tau:.1f}: model implies {implied:.1%} need a 3rd map")


if __name__ == "__main__":
    main()
