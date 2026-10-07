"""Does the side a team STARTS a map on (attack / defence) change who wins it?

Compares each map's result with what Elo alone predicted for the team that
started on attack. A mean residual near 0 means the starting side carries no
information beyond team strength; clearly positive means starting on attack
helps (negative: starting on defence helps). Also reports side-choice
behaviour on picked maps (the picker's opponent chooses the side), per-map
attack-round win rates, and a data-consistency check on the scraped side counts.

    python scripts/analyze_start_side.py                  # default database
    python scripts/analyze_start_side.py --db data/x.db
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from valpredictor.features.build_features import build_map_training_table  # noqa: E402
from valpredictor.features.rolling import side_rounds  # noqa: E402
from valpredictor.storage.db import get_connection  # noqa: E402


def _mean_se(values: pd.Series) -> tuple[float, float, int]:
    n = len(values)
    if n < 2:
        return float("nan"), float("nan"), n
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(n)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    conn = get_connection(args.db)

    maps = pd.read_sql_query(
        """
        SELECT m.match_id, m.map_order, m.map_name, m.team1_score, m.team2_score, m.team1_start_side,
               m.team1_atk_won, m.team1_def_won, m.team2_atk_won, m.team2_def_won,
               m.picked_by_team_id, mt.team1_id, mt.team2_id
        FROM maps m JOIN matches mt ON mt.id = m.match_id
        WHERE m.team1_score IS NOT NULL
        """,
        conn,
    )
    with_sides = maps[maps["team1_start_side"].notna()].copy()
    print(f"maps: {len(maps)}   with side data: {len(with_sides)}")
    if with_sides.empty:
        print("no side data yet - run `valpredictor reparse` first")
        return

    # ---- consistency: attackers' wins + defenders' wins must equal rounds played on that side
    def consistent(r) -> bool:
        a1, d1, a2, d2 = side_rounds(int(r.team1_score), int(r.team2_score), r.team1_start_side)
        return r.team1_atk_won + r.team2_def_won == a1 and r.team1_def_won + r.team2_atk_won == d1

    with_sides["ok"] = with_sides.apply(consistent, axis=1)
    print(f"side counts internally consistent: {int(with_sides['ok'].sum())} / {len(with_sides)}")
    with_sides = with_sides[with_sides["ok"]].copy()

    # ---- league-wide attack round win rate, per map (regulation rounds only)
    rows = []
    for r in with_sides.itertuples():
        a1, d1, a2, d2 = side_rounds(int(r.team1_score), int(r.team2_score), r.team1_start_side)
        rows.append((r.map_name, r.team1_atk_won + r.team2_atk_won, a1 + a2))
    per_map = pd.DataFrame(rows, columns=["map", "atk_won", "atk_played"]).groupby("map").sum()
    per_map["atk_rate"] = per_map["atk_won"] / per_map["atk_played"]
    per_map["maps"] = (per_map["atk_played"] / 24).round().astype(int)
    total_rate = per_map["atk_won"].sum() / per_map["atk_played"].sum()
    print(f"\nattack-side round win rate: {total_rate:.1%} overall")
    print(per_map.sort_values("atk_rate", ascending=False)[["maps", "atk_rate"]]
          .to_string(formatters={"atk_rate": "{:.1%}".format}))

    # ---- does the starting side matter beyond team strength?
    feats = build_map_training_table(conn)[["match_id", "map_order", "team1_elo", "team2_elo", "team1_won_map"]]
    df = with_sides.merge(feats, on=["match_id", "map_order"])
    p1 = 1.0 / (1.0 + 10 ** (-(df["team1_elo"] - df["team2_elo"]) / 400.0))
    t1_atk = df["team1_start_side"] == "atk"
    # residual from the point of view of the team that started on attack
    df["atk_starter_won"] = df["team1_won_map"].where(t1_atk, 1 - df["team1_won_map"])
    df["atk_starter_expected"] = p1.where(t1_atk, 1 - p1)
    df["resid"] = df["atk_starter_won"] - df["atk_starter_expected"]

    raw = df["atk_starter_won"].mean()
    mean, se, n = _mean_se(df["resid"])
    print(f"\nteam that STARTED on attack won {raw:.1%} of {n} maps "
          f"(Elo alone expected {df['atk_starter_expected'].mean():.1%})")
    print(f"starting on attack vs Elo expectation: {mean:+.1%} +/- {se:.1%} (1 s.e.)")
    verdict = "no clear effect" if abs(mean) < 2 * se else ("attack start helps" if mean > 0 else "defence start helps")
    print(f"=> {verdict} at this sample size (needs |effect| > 2 s.e. to call it)")

    # ---- side choice on picked maps: the picker's opponent chooses the side
    picked = df[df["picked_by_team_id"].notna()].copy()
    if len(picked) >= 10:
        chooser_is_t1 = picked["picked_by_team_id"] == picked["team2_id"]
        chooser_atk = (chooser_is_t1 & (picked["team1_start_side"] == "atk")) | (
            ~chooser_is_t1 & (picked["team1_start_side"] == "def"))
        print(f"\npicked maps: {len(picked)}; the side-choosing team chose attack {chooser_atk.mean():.1%} of the time")
        p_ch = p1.loc[picked.index].where(chooser_is_t1, 1 - p1.loc[picked.index])
        won_ch = picked["team1_won_map"].where(chooser_is_t1, 1 - picked["team1_won_map"])
        for label, mask in (("chose attack", chooser_atk), ("chose defence", ~chooser_atk)):
            m, s, k = _mean_se((won_ch - p_ch)[mask])
            print(f"  chooser {label:13s}: {k:4d} maps, result vs Elo expectation {m:+.1%} +/- {s:.1%}")


if __name__ == "__main__":
    main()
