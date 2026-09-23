"""cspredictor command-line entrypoint.

    cspredictor backfill-results --since 2024-09-22
    cspredictor backfill-rankings --since 2024-09-22
    cspredictor build-features
    cspredictor train
    cspredictor backtest
    cspredictor predict --team1 "Team A" --team2 "Team B" --maps Mirage,Inferno,Ancient

Every command accepts `--db PATH` (before the subcommand) to operate on a
database other than the default from config.yaml, e.g.:

    cspredictor --db data/demo.db build-features
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import click
import pandas as pd

from cspredictor.config import load_config, resolve_path
from cspredictor.features.build_features import build_map_training_table, symmetrize, to_model_matrix
from cspredictor.models.evaluate import walk_forward_backtest
from cspredictor.models.map_model import (
    chronological_holdout_split,
    load_model,
    predict_proba,
    save_model,
    train_map_model,
)
from cspredictor.models.predict import TeamNotFoundError, format_prediction, predict_match
from cspredictor.scraping.client import HLTVClient
from cspredictor.scraping.ingest import run_backfill, run_rankings_backfill
from cspredictor.storage.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _db_path(ctx) -> Path:
    return ctx.obj["db_path"] or resolve_path(load_config()["database"]["path"])


def _default_training_table_path(ctx) -> Path:
    return _db_path(ctx).parent / "training_table.parquet"


def _default_model_path(ctx) -> Path:
    return _db_path(ctx).parent / "map_model.txt"


@click.group()
@click.option("--db", "db_path", default=None, help="SQLite DB path (defaults to config.yaml's database.path)")
@click.pass_context
def cli(ctx, db_path: str | None):
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = resolve_path(db_path) if db_path else None


@cli.command("backfill-results")
@click.option("--since", required=True, help="YYYY-MM-DD, earliest match date to ingest")
@click.option("--until", default=None, help="YYYY-MM-DD, defaults to today")
@click.pass_context
def backfill_results(ctx, since: str, until: str | None):
    """Scrape + ingest match results from HLTV into SQLite. Resumable — safe
    to re-run after an interruption; already-ingested matches are skipped."""
    since_d = dt.date.fromisoformat(since)
    until_d = dt.date.fromisoformat(until) if until else None
    conn = get_connection(_db_path(ctx))
    client = HLTVClient()
    count = run_backfill(conn, client, since_d, until_d)
    click.echo(
        f"ingested {count} new matches "
        f"({client.stats.live_requests} live requests, {client.stats.cache_hits} cache hits)"
    )


@cli.command("backfill-rankings")
@click.option("--since", required=True, help="YYYY-MM-DD")
@click.option("--until", default=None, help="YYYY-MM-DD, defaults to today")
@click.pass_context
def backfill_rankings(ctx, since: str, until: str | None):
    """Scrape + ingest weekly HLTV world-ranking snapshots into SQLite."""
    since_d = dt.date.fromisoformat(since)
    until_d = dt.date.fromisoformat(until) if until else None
    conn = get_connection(_db_path(ctx))
    client = HLTVClient()
    count = run_rankings_backfill(conn, client, since_d, until_d)
    click.echo(
        f"ingested {count} ranking snapshots "
        f"({client.stats.live_requests} live requests, {client.stats.cache_hits} cache hits)"
    )


@cli.command("build-features")
@click.option("--out", default=None, help="output parquet path")
@click.pass_context
def build_features_cmd(ctx, out: str | None):
    """Replay match history chronologically into the leakage-safe training table."""
    conn = get_connection(_db_path(ctx))
    df = build_map_training_table(conn)
    if df.empty:
        click.echo("no played maps found in the database yet — run backfill-results first", err=True)
        raise SystemExit(1)
    df = symmetrize(df)
    matrix = to_model_matrix(df)
    out_path = resolve_path(out) if out else _default_training_table_path(ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_path)
    click.echo(f"wrote {len(matrix)} rows ({len(matrix) // 2} maps, symmetrized) to {out_path}")


@cli.command("train")
@click.option("--in", "in_path", default=None, help="input parquet path (from build-features)")
@click.option("--out", default=None, help="output model path")
@click.pass_context
def train_cmd(ctx, in_path: str | None, out: str | None):
    """Train the map-level LightGBM model on the full training table, holding
    out the chronologically-last slice of matches for early stopping."""
    src = resolve_path(in_path) if in_path else _default_training_table_path(ctx)
    df = pd.read_parquet(src)
    inner_train, inner_valid = chronological_holdout_split(df)
    model = train_map_model(inner_train, valid_df=inner_valid if not inner_valid.empty else None)
    out_path = resolve_path(out) if out else _default_model_path(ctx)
    save_model(model, out_path)
    best_iter = getattr(model, "best_iteration", None)
    click.echo(
        f"trained on {len(inner_train)} rows (held out {len(inner_valid)} for early stopping, "
        f"best_iteration={best_iter}), saved model to {out_path}"
    )


@cli.command("backtest")
@click.option("--in", "in_path", default=None, help="input parquet path (from build-features)")
@click.option("--folds", default=5, help="number of walk-forward folds")
@click.pass_context
def backtest_cmd(ctx, in_path: str | None, folds: int):
    """Walk-forward backtest: main model vs elo-only vs naive-rank baselines."""
    src = resolve_path(in_path) if in_path else _default_training_table_path(ctx)
    df = pd.read_parquet(src)
    results = walk_forward_backtest(df, n_folds=folds)

    click.echo("Per-fold accuracy (model / elo-only / naive-rank):")
    for fold in results["folds"]:
        m, e, n = fold["model"], fold["elo_only"], fold["naive_rank"]
        click.echo(
            f"  fold {fold['fold']} [{fold['test_start']}..{fold['test_end']}] n={m['n']}: "
            f"{_fmt(m['accuracy'])} / {_fmt(e['accuracy'])} / {_fmt(n['accuracy'])}"
        )

    click.echo("\nOverall (pooled out-of-fold):")
    for name, res in results["overall"].items():
        click.echo(
            f"  {name:12s} n={res['n']:<6} accuracy={_fmt(res['accuracy'])} "
            f"log_loss={_fmt(res['log_loss'])} brier={_fmt(res['brier'])}"
        )


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "n/a"


@cli.command("predict")
@click.option("--team1", required=True)
@click.option("--team2", required=True)
@click.option("--best-of", default=3, type=int)
@click.option("--maps", default=None, help="comma-separated map list if the veto is known, e.g. Mirage,Inferno,Ancient")
@click.option(
    "--picks", default=None,
    help='who picked which known map, e.g. "Mirage:Team A,Inferno:Team B" (unlisted maps are treated as deciders)',
)
@click.option("--model", "model_path", default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.pass_context
def predict_cmd(
    ctx, team1: str, team2: str, best_of: int, maps: str | None, picks: str | None,
    model_path: str | None, as_json: bool,
):
    """Predict a matchup: match-winner probability and map-score distribution."""
    conn = get_connection(_db_path(ctx))
    path = resolve_path(model_path) if model_path else _default_model_path(ctx)
    if not path.exists():
        click.echo(f"no trained model at {path} — run `train` first", err=True)
        raise SystemExit(1)
    model = load_model(path)
    map_list = [m.strip() for m in maps.split(",")] if maps else None
    picks_dict = None
    if picks:
        picks_dict = {}
        for pair in picks.split(","):
            map_name, _, picker = pair.partition(":")
            if picker:
                picks_dict[map_name.strip()] = picker.strip()

    try:
        result = predict_match(conn, model, team1, team2, best_of=best_of, maps=map_list, picks=picks_dict)
    except TeamNotFoundError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)

    if as_json:
        printable = dict(result)
        printable["score_distribution"] = {f"{a}-{b}": p for (a, b), p in result["score_distribution"].items()}
        click.echo(json.dumps(printable, indent=2))
    else:
        click.echo(format_prediction(result))


if __name__ == "__main__":
    cli()
