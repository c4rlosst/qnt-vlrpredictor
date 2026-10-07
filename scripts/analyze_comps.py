"""Do agent compositions predict who wins a map, beyond how strong the teams are?

Compositions are only known once agent select happens on the map, so this is
an UPPER BOUND on what comps could add to a pre-match prediction: if even the
comps actually played carry no signal, a guess of them can't either.

Two comp strengths are built strictly from earlier maps (no leakage):
  agent effect  - for each (agent, map): how often teams that played that agent
                  on that map won, shrunk toward 50%; a comp's score is the mean
                  effect of its five agents
  exact comp    - how often the identical five-agent comp won on that map,
                  shrunk toward 50% (only meaningful when comps repeat)
Each is then added to calibrated Elo and scored on the same walk-forward folds.

    python scripts/analyze_comps.py [--db data/x.db]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from valpredictor.config import load_config  # noqa: E402
from valpredictor.features.build_features import build_map_training_table, symmetrize, to_model_matrix  # noqa: E402
from valpredictor.models.evaluate import walk_forward_backtest  # noqa: E402
from valpredictor.storage.db import get_connection  # noqa: E402

PRIOR = 20.0  # pseudo-observations at 50%: heavy shrinkage, since cells are small


def comp_features(conn) -> pd.DataFrame:
    """(match_id, map_order) -> agent_edge, exact_edge for team 1 minus team 2, built from earlier maps only."""
    maps = pd.read_sql_query(
        """
        SELECT m.match_id, m.map_order, m.map_name, m.team1_score, m.team2_score, m.team1_comp, m.team2_comp
        FROM maps m JOIN matches mt ON mt.id = m.match_id
        WHERE m.team1_comp IS NOT NULL AND m.team2_comp IS NOT NULL AND m.team1_score IS NOT NULL
        ORDER BY COALESCE(mt.unix_timestamp_ms, 0), mt.vlr_id, m.map_order
        """,
        conn,
    )
    agent_stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])  # (agent, map) -> [wins, plays]
    comp_stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    rows = []

    def agent_score(comp: str, map_name: str) -> float:
        effects = []
        for agent in comp.split(","):
            wins, plays = agent_stats[(agent, map_name)]
            effects.append((wins + PRIOR * 0.5) / (plays + PRIOR) - 0.5)
        return float(np.mean(effects))

    def exact_score(comp: str, map_name: str) -> float:
        wins, plays = comp_stats[(comp, map_name)]
        return (wins + PRIOR * 0.5) / (plays + PRIOR) - 0.5

    for r in maps.itertuples():
        a1, a2 = agent_score(r.team1_comp, r.map_name), agent_score(r.team2_comp, r.map_name)
        e1, e2 = exact_score(r.team1_comp, r.map_name), exact_score(r.team2_comp, r.map_name)
        rows.append((r.match_id, r.map_order, a1 - a2, e1 - e2, comp_stats[(r.team1_comp, r.map_name)][1] > 0))
        t1_won = r.team1_score > r.team2_score
        for comp, won in ((r.team1_comp, t1_won), (r.team2_comp, not t1_won)):
            comp_stats[(comp, r.map_name)][0] += won
            comp_stats[(comp, r.map_name)][1] += 1
            for agent in comp.split(","):
                agent_stats[(agent, r.map_name)][0] += won
                agent_stats[(agent, r.map_name)][1] += 1
    out = pd.DataFrame(rows, columns=["match_id", "map_order", "agent_edge", "exact_edge", "comp_seen_before"])
    print(f"maps with both comps: {len(out)}; distinct comps: {len(set(maps.team1_comp) | set(maps.team2_comp))}; "
          f"team-1 comp seen before on that map: {out['comp_seen_before'].mean():.0%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    conn, cfg = get_connection(args.db), load_config()

    feats = comp_features(conn)
    table = to_model_matrix(symmetrize(build_map_training_table(conn, cfg)))
    keys = table[["match_id", "map_order"]].merge(feats, on=["match_id", "map_order"], how="left")
    keys.index = table.index
    # a mirrored row sees the same comps from the other team's side: flip the sign
    mirrored = table["team1_id"] > table["team2_id"]
    # orientation of the feature rows is team1 as stored: identify rows whose team1 is the stored team1
    stored = pd.read_sql_query(
        "SELECT m.match_id, m.map_order, mt.team1_id AS stored_team1 FROM maps m JOIN matches mt ON mt.id = m.match_id", conn)
    keys = keys.merge(stored, on=["match_id", "map_order"], how="left")
    keys.index = table.index
    sign = np.where(table["team1_id"].to_numpy() == keys["stored_team1"].to_numpy(), 1.0, -1.0)
    table["agent_edge"] = keys["agent_edge"].to_numpy() * sign
    table["exact_edge"] = keys["exact_edge"].to_numpy() * sign
    table = table[table["agent_edge"].notna()]
    print(f"rows with comps: {len(table)} ({len(table) // 2} maps)")

    half_life = cfg["model"].get("recency_half_life_days")

    def lr(cols):
        def predict(train, test):
            from valpredictor.models.map_model import recency_weights

            X = lambda d: d[cols].fillna(0.0).to_numpy(dtype=float)
            mu, sd = X(train).mean(0), X(train).std(0) + 1e-9
            model = LogisticRegression(C=1.0, max_iter=1000).fit(
                (X(train) - mu) / sd, train["target"], sample_weight=recency_weights(train["match_date"], half_life))
            return model.predict_proba((X(test) - mu) / sd)[:, 1]
        return predict

    results = []
    for name, cols in (("Elo", ["elo_diff"]), ("Elo + agent comp", ["elo_diff", "agent_edge"]),
                       ("Elo + exact comp", ["elo_diff", "exact_edge"]),
                       ("Elo + both", ["elo_diff", "agent_edge", "exact_edge"])):
        captured: list[pd.Series] = []
        predictor = lr(cols)
        walk_forward_backtest(table, cfg, n_folds=5, predictor=lambda tr, te, _p=predictor: captured.append(pd.Series(_p(tr, te), index=te.index)) or captured[-1])
        oof = pd.concat(captured)
        rows = table.loc[oof.index]
        keep = (rows["team1_id"] < rows["team2_id"]).to_numpy()
        y, p = rows["target"].to_numpy(dtype=float)[keep], np.clip(oof.to_numpy()[keep], 1e-6, 1 - 1e-6)
        results.append((name, -(y * np.log(p) + (1 - y) * np.log(1 - p)), float(((p >= 0.5) == (y == 1)).mean())))

    base = results[0][1]
    print(f"\n{'model':22s} {'maps':>5s} {'acc':>6s} {'log-loss':>9s}  vs Elo")
    for name, ll, acc in results:
        diff = ll - base
        se = diff.std(ddof=1) / math.sqrt(len(diff)) if name != "Elo" else 0.0
        print(f"{name:22s} {len(ll):5d} {acc:6.3f} {ll.mean():9.4f}  {diff.mean():+.4f} +/- {se:.4f}")
    print("\n(negative = better than Elo alone; under ~2 s.e. is noise)")


if __name__ == "__main__":
    main()
