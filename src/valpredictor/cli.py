"""valpredictor command-line entrypoint.

    valpredictor scan-events --since 2025-10-01       # what events exist? (list pages only)
    valpredictor backfill-results --since 2025-10-01  # scrape + store tier-1 matches
    valpredictor build-features
    valpredictor train
    valpredictor backtest
    valpredictor predict --team1 "NRG" --team2 "T1" --maps Lotus,Summit,Abyss
    valpredictor upcoming --event Champions           # predict the live bracket

Every command accepts `--db PATH` (before the subcommand) to use a database
other than the default from config.yaml.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from pathlib import Path

import click
import pandas as pd

from valpredictor.config import load_config, resolve_path
from valpredictor.features.build_features import build_map_training_table, symmetrize, to_model_matrix
from valpredictor.importer import import_csv
from valpredictor.models.evaluate import walk_forward_backtest
from valpredictor.models.map_model import (
    chronological_holdout_split,
    load_model,
    save_model,
    train_map_model,
)
from valpredictor.models.predict import TeamNotFoundError, format_prediction, predict_match, result_to_json
from valpredictor.scraping.client import VLRClient
from valpredictor.scraping.ingest import reparse_from_cache, run_backfill, scan_events
from valpredictor.storage.db import get_connection
from valpredictor.upcoming import predict_upcoming

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _db_path(ctx) -> Path:
    return ctx.obj["db_path"] or resolve_path(load_config()["database"]["path"])


def _default_training_table_path(ctx) -> Path:
    p = _db_path(ctx)
    return p.parent / f"{p.stem}_training_table.parquet"


def _default_model_path(ctx) -> Path:
    p = _db_path(ctx)
    return p.parent / f"{p.stem}_map_model.txt"


def _default_since() -> dt.date:
    months = int(load_config()["backfill"]["months_back"])
    return dt.date.today() - dt.timedelta(days=round(months * 30.4))


@click.group()
@click.option("--db", "db_path", default=None, help="SQLite DB path (defaults to config.yaml's database.path)")
@click.pass_context
def cli(ctx, db_path: str | None):
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = resolve_path(db_path) if db_path else None


@cli.command("scan-events")
@click.option("--since", default=None, help="YYYY-MM-DD (default: config backfill.months_back ago)")
@click.option("--until", default=None, help="YYYY-MM-DD, defaults to today")
def scan_events_cmd(since: str | None, until: str | None):
    """List event names (with match counts) found on the results list, and
    whether the configured include/exclude filter would keep them. Costs one
    request per ~50 matches; fetches no match pages."""
    cfg = load_config()["backfill"]
    include, exclude = re.compile(cfg["event_include"], re.I), re.compile(cfg["event_exclude"], re.I)
    since_d = dt.date.fromisoformat(since) if since else _default_since()
    counts = scan_events(VLRClient(), since_d, dt.date.fromisoformat(until) if until else None)
    kept = 0
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        keep = bool(include.search(name)) and not exclude.search(name)
        kept += n if keep else 0
        click.echo(f"{'KEEP' if keep else 'skip'}  {n:5d}  {name}")
    click.echo(f"\n{kept} of {sum(counts.values())} matches pass the filter")


@cli.command("backfill-results")
@click.option("--since", default=None, help="YYYY-MM-DD, earliest match date (default: config months_back)")
@click.option("--until", default=None, help="YYYY-MM-DD, defaults to today")
@click.option("--include", default=None, help="regex override for config backfill.event_include")
@click.option("--exclude", default=None, help="regex override for config backfill.event_exclude")
@click.pass_context
def backfill_results(ctx, since: str | None, until: str | None, include: str | None, exclude: str | None):
    """Scrape + store completed matches from vlr.gg. Resumable: stored matches
    are skipped without a request, so just re-run after an interruption."""
    cfg = load_config()["backfill"]
    inc = re.compile(include or cfg["event_include"], re.I)
    exc = re.compile(exclude or cfg["event_exclude"], re.I)
    since_d = dt.date.fromisoformat(since) if since else _default_since()
    until_d = dt.date.fromisoformat(until) if until else None

    conn = get_connection(_db_path(ctx))
    client = VLRClient()
    stats = run_backfill(conn, client, since_d, until_d, inc, exc)
    click.echo(
        f"stored {stats.ingested} new matches ({stats.already_stored} already stored, "
        f"{stats.filtered_out} filtered out, {stats.failed} failed; "
        f"{client.stats.live_requests} live requests, {client.stats.cache_hits} cache hits)"
    )
    for name, n in stats.events_included.most_common():
        click.echo(f"  {n:4d}  {name}")


@cli.command("reparse")
@click.pass_context
def reparse_cmd(ctx):
    """Re-read all stored matches from the local page cache (no network
    requests) to fill in newly parsed fields, e.g. per-side round data."""
    updated, missing = reparse_from_cache(get_connection(_db_path(ctx)), VLRClient())
    click.echo(f"re-parsed {updated} matches from the cache ({missing} had no cached page)")


@cli.command("import-csv")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="validate and report, but write nothing")
@click.option("--replace", is_flag=True, help="overwrite matches whose match_id already exists (default: skip them)")
@click.pass_context
def import_csv_cmd(ctx, path: Path, dry_run: bool, replace: bool):
    """Import hand-collected matches from a CSV (one row per map played).
    Validated up front and all-or-nothing; see examples/maps_example.csv."""
    result = import_csv(get_connection(_db_path(ctx)), path, dry_run=dry_run, replace=replace)

    if result.errors:
        shown = result.errors[:25]
        click.echo(f"{len(result.errors)} problem(s) found, nothing was imported:", err=True)
        for e in shown:
            click.echo(f"  - {e}", err=True)
        if len(result.errors) > len(shown):
            click.echo(f"  ... and {len(result.errors) - len(shown)} more", err=True)
        raise SystemExit(1)

    verb = "would import" if dry_run else "imported"
    click.echo(f"{verb} {result.matches_imported} matches ({result.maps_imported} maps)")
    if result.matches_skipped:
        click.echo(f"skipped {len(result.matches_skipped)} already-stored match_id(s) (use --replace to overwrite)")
    if result.new_teams:
        click.echo(f"new teams: {', '.join(result.new_teams)}")
    for w in result.warnings:
        click.echo(f"warning: {w}")
    if not dry_run and result.matches_imported:
        click.echo("next: valpredictor build-features && valpredictor train")


@cli.command("build-features")
@click.option("--out", default=None, help="output parquet path")
@click.pass_context
def build_features_cmd(ctx, out: str | None):
    """Replay match history chronologically into the leakage-safe training table."""
    conn = get_connection(_db_path(ctx))
    df = build_map_training_table(conn)
    if df.empty:
        click.echo("no played maps in the database yet — run backfill-results first", err=True)
        raise SystemExit(1)
    matrix = to_model_matrix(symmetrize(df))
    out_path = resolve_path(out) if out else _default_training_table_path(ctx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_path)
    click.echo(f"wrote {len(matrix)} rows ({len(matrix) // 2} maps, symmetrized) to {out_path}")


@cli.command("train")
@click.option("--in", "in_path", default=None, help="input parquet path (from build-features)")
@click.option("--out", default=None, help="output model path")
@click.pass_context
def train_cmd(ctx, in_path: str | None, out: str | None):
    """Train the map-level LightGBM model, holding out the chronologically
    last slice of matches for early stopping."""
    df = pd.read_parquet(resolve_path(in_path) if in_path else _default_training_table_path(ctx))
    inner_train, inner_valid = chronological_holdout_split(df)
    model = train_map_model(inner_train, valid_df=inner_valid if not inner_valid.empty else None)
    out_path = resolve_path(out) if out else _default_model_path(ctx)
    save_model(model, out_path)
    click.echo(
        f"trained on {len(inner_train)} rows (held out {len(inner_valid)} for early stopping, "
        f"best_iteration={getattr(model, 'best_iteration', None)}), saved model to {out_path}"
    )


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "n/a"


@cli.command("backtest")
@click.option("--in", "in_path", default=None, help="input parquet path (from build-features)")
@click.option("--folds", default=5, help="number of walk-forward folds")
@click.pass_context
def backtest_cmd(ctx, in_path: str | None, folds: int):
    """Walk-forward backtest: main model vs elo-only vs naive-form baselines."""
    df = pd.read_parquet(resolve_path(in_path) if in_path else _default_training_table_path(ctx))
    results = walk_forward_backtest(df, n_folds=folds)

    click.echo("Per-fold map accuracy (model / elo-only / naive-form):")
    for fold in results["folds"]:
        m, e, n = fold["model"], fold["elo_only"], fold["naive_form"]
        click.echo(
            f"  fold {fold['fold']} [{fold['test_start']}..{fold['test_end']}] n={m['n']}: "
            f"{_fmt(m['accuracy'])} / {_fmt(e['accuracy'])} / {_fmt(n['accuracy'])}"
        )
    click.echo("\nOverall (pooled out-of-fold, per map):")
    for name, res in results["overall"].items():
        click.echo(
            f"  {name:12s} n={res['n']:<6} accuracy={_fmt(res['accuracy'])} "
            f"log_loss={_fmt(res['log_loss'])} brier={_fmt(res['brier'])}"
        )


def _load_model_or_exit(ctx, model_path: str | None):
    path = resolve_path(model_path) if model_path else _default_model_path(ctx)
    if not path.exists():
        click.echo(f"no trained model at {path} — run `train` first", err=True)
        raise SystemExit(1)
    return load_model(path)


@cli.command("predict")
@click.option("--team1", required=True)
@click.option("--team2", required=True)
@click.option("--best-of", default=3, type=int)
@click.option("--maps", default=None, help="comma-separated maps in play order if the veto is known, e.g. Lotus,Summit,Abyss")
@click.option("--picks", default=None, help='who picked which map, e.g. "Lotus:T1,Summit:NRG" (unlisted = decider)')
@click.option("--international", is_flag=True, default=False, help="treat as a Masters/Champions-level event")
@click.option("--model", "model_path", default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.pass_context
def predict_cmd(ctx, team1, team2, best_of, maps, picks, international, model_path, as_json):
    """Predict one matchup: winner probability and map-score distribution."""
    conn = get_connection(_db_path(ctx))
    model = _load_model_or_exit(ctx, model_path)
    map_list = [m.strip() for m in maps.split(",")] if maps else None
    picks_dict = None
    if picks:
        picks_dict = {}
        for pair in picks.split(","):
            map_name, _, picker = pair.partition(":")
            if picker:
                picks_dict[map_name.strip()] = picker.strip()
    try:
        result = predict_match(
            conn, model, team1, team2, best_of=best_of, maps=map_list, picks=picks_dict,
            is_international=international or None,
        )
    except TeamNotFoundError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1)

    if as_json:
        click.echo(json.dumps(result_to_json(result), indent=2))
    else:
        click.echo(format_prediction(result))


@cli.command("upcoming")
@click.option("--event", default="Champions", help="regex on event name (default: Champions)")
@click.option("--model", "model_path", default=None)
@click.pass_context
def upcoming_cmd(ctx, event: str, model_path: str | None):
    """Fetch the live schedule from vlr.gg and predict every upcoming/live
    match of matching events whose teams are already known."""
    conn = get_connection(_db_path(ctx))
    model = _load_model_or_exit(ctx, model_path)
    items = predict_upcoming(conn, VLRClient(), model, event)
    if not items:
        click.echo(f"no upcoming/live matches found for event pattern {event!r}")
        return
    for item in items:
        r = item.row
        header = f"[{r.date_label} {r.time_label}] {r.series}"
        if item.result is None:
            click.echo(f"\n{header}\n  {r.team1_name} vs {r.team2_name}: {item.note}")
            continue
        live = "  (LIVE — prediction ignores the current score)\n" if r.status == "live" else ""
        click.echo(f"\n{header}\n{live}{format_prediction(item.result)}")


@cli.command("serve")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.option("--model", "model_path", default=None)
@click.pass_context
def serve_cmd(ctx, host: str, port: int, model_path: str | None):
    """Run the local web UI (open http://127.0.0.1:8000 in a browser)."""
    from valpredictor.web.server import serve

    path = resolve_path(model_path) if model_path else _default_model_path(ctx)
    model = load_model(path) if path.exists() else None
    if model is None:
        click.echo("no trained model yet - open the page and click \"Refresh data\" to scrape matches and train one")
    serve(_db_path(ctx), model, host, port, model_path=path)

if __name__ == "__main__":
    cli()
