"""Do the event tier and the stakes of a match change how it plays out?

For every stage bucket this reports how the favourite (higher Elo going in)
did against what a calibrated Elo model expected, and how often Bo3s went to
a third map. A mean residual near 0 means that stage behaves like any other;
clearly positive means favourites do better than expected there (chalk),
negative means more upsets. Differences under ~2 standard errors are noise.

    python scripts/analyze_stage.py [--db data/x.db]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from valpredictor.features.build_features import build_map_training_table  # noqa: E402
from valpredictor.storage.db import get_connection  # noqa: E402

STAKES = {0: "regular season", 1: "playoffs (life in hand)", 2: "elimination", 3: "final"}
TIERS = {0: "other / qualifier", 1: "regional league", 2: "international"}


def _se(x: pd.Series) -> float:
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def report(df: pd.DataFrame, by: str, names: dict[int, str]) -> None:
    print(f"\n{'':26s} {'maps':>5s} {'fav win':>8s} {'expected':>9s} {'residual':>16s} {'Bo3 deciders':>14s}")
    for key, name in names.items():
        sub = df[df[by] == key]
        if sub.empty:
            continue
        resid = sub["fav_won"] - sub["fav_p"]
        bo3 = sub[(sub["best_of"] == 3) & (sub["map_order"] == 1)]
        deciders = bo3["went_3"].mean() if len(bo3) else float("nan")
        print(
            f"{name:26s} {len(sub):5d} {sub['fav_won'].mean():8.1%} {sub['fav_p'].mean():9.1%} "
            f"{resid.mean():+8.1%} +/- {_se(resid):4.1%} {deciders:9.1%} (n={len(bo3)})"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    conn = get_connection(args.db)

    maps = build_map_training_table(conn)
    meta = pd.read_sql_query("SELECT id AS match_id, stakes, event_tier FROM matches", conn)
    count = pd.read_sql_query("SELECT match_id, COUNT(*) AS n FROM maps GROUP BY match_id", conn).set_index("match_id")["n"]
    df = maps.drop(columns=["event_tier", "stakes"], errors="ignore").merge(meta, on="match_id")
    df["went_3"] = (df["match_id"].map(count) == 3).astype(float)

    # calibrated Elo: P(team1 wins) = sigmoid(a + b * elo_diff), fitted on all maps
    diff = (df["team1_elo"] - df["team2_elo"]).to_numpy()[:, None] / 100.0
    lr = LogisticRegression(C=1e6).fit(np.vstack([diff, -diff]), np.concatenate([df["team1_won_map"], 1 - df["team1_won_map"]]))
    p1 = lr.predict_proba(diff)[:, 1]
    fav_is_1 = p1 >= 0.5
    df["fav_p"] = np.where(fav_is_1, p1, 1 - p1)
    df["fav_won"] = np.where(fav_is_1, df["team1_won_map"], 1 - df["team1_won_map"])
    print(f"maps: {len(df)}; calibrated Elo: P(win) = sigmoid({lr.intercept_[0]:+.3f} + {lr.coef_[0][0]:.3f} * eloDiff/100)")
    print(f"favourite wins {df['fav_won'].mean():.1%} of maps (calibrated expectation {df['fav_p'].mean():.1%})")

    print("\n== by stakes ==")
    report(df, "stakes", STAKES)
    print("\n== by event tier ==")
    report(df, "event_tier", TIERS)
    print("\nresidual = favourite's actual map win rate minus calibrated-Elo expectation (+ = chalk, - = upsets)")


if __name__ == "__main__":
    main()
