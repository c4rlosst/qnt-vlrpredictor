"""The prediction model: a team's Elo gap -> its chance of winning a map.

P(team 1 wins a map) = sigmoid(slope * elo_diff / 100), with the single
`slope` fitted on past maps (recent matches weigh more). There is no
intercept: team 1 vs team 2 is an arbitrary labelling, so P(A beats B) must
equal 1 - P(B beats A). A plain Elo expectation corresponds to slope 0.576;
the fitted slope is lower (~0.44 on the 2025-26 season), i.e. Elo gaps in
tier-1 Valorant mean a bit less than the textbook scale says.

Why this and nothing more: on a full season (~2,000 maps) every richer model
(gradient boosting on form, per-map Elo, sides, head-to-head, rosters, ...)
tied this or did worse out of sample; see README.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_ELO_SLOPE = math.log(10) / 4.0  # logit per 100 Elo points on the standard 400-point scale (0.5756)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def raw_elo_prob(elo_diff):
    """The textbook Elo expectation, for comparison."""
    return sigmoid(RAW_ELO_SLOPE * np.asarray(elo_diff, dtype=float) / 100.0)


def recency_weights(match_dates: pd.Series, half_life_days: float | None) -> np.ndarray | None:
    """Exponential-decay sample weights: a row `half_life_days` older than the
    newest row counts half as much. None disables weighting."""
    if not half_life_days:
        return None
    dates = pd.to_datetime(match_dates)
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age_days / float(half_life_days))


def fit_slope(elo_diff: np.ndarray, won: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted logistic regression through the origin (Newton's method)."""
    x = np.asarray(elo_diff, dtype=float) / 100.0
    y = np.asarray(won, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    if len(x) < 2 or not np.any(x):
        return RAW_ELO_SLOPE
    b = 0.0
    for _ in range(50):
        p = sigmoid(b * x)
        grad = float(np.sum(w * x * (y - p)))
        hess = float(np.sum(w * x * x * p * (1.0 - p)))
        if hess < 1e-12:
            break
        step = grad / hess
        b += step
        if abs(step) < 1e-10:
            break
    return b


@dataclass
class EloModel:
    slope: float
    trained_on_maps: int = 0

    def map_prob(self, elo_diff: float) -> float:
        """P(team 1 wins a map) for a team-1-minus-team-2 Elo gap."""
        return float(sigmoid(self.slope * elo_diff / 100.0))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "EloModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_elo_model(rows: pd.DataFrame, half_life_days: float | None = None) -> EloModel:
    """Fit on a table of past maps with columns elo_diff, team1_won_map, match_date."""
    weights = recency_weights(rows["match_date"], half_life_days) if len(rows) else None
    slope = fit_slope(rows["elo_diff"].to_numpy(), rows["team1_won_map"].to_numpy(), weights)
    return EloModel(slope=slope, trained_on_maps=len(rows))
