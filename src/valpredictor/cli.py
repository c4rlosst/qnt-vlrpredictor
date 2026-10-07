"""valpredictor command-line entrypoint.

    valpredictor scan-events --since 2025-10-01       # what events exist? (list pages only)
    valpredictor backfill-results --since 2025-10-01  # scrape + store tier-1 matches
    valpredictor train                                 # replay history into Elo and fit the model
    valpredictor backtest                              # walk-forward check against textbook Elo
    valpredictor predict --team1 "NRG" --team2 "T1"
    valpredictor upcoming --event Champions           # predict the upcoming bracket
    valpredictor serve                                 # the browser UI

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

from valpredictor.config import load_config, resolve_path
from valpredictor.features.build_features import build_map_table
from valpredictor.importer import import_csv
from valpredictor.models.elo_model import EloModel, fit_elo_model
from valpredictor.models.evaluate import COIN_FLIP_LOG_LOSS, walk_forward_backtest
from valpredictor.models.predict import TeamNotFoundError, format_prediction, predict_match, result_to_json
from valpredictor.scraping.client import VLRClient
from valpredictor.scraping.ingest import run_backfill, scan_events
from valpredictor.storage.db import get_connection
from valpredictor.upcoming import predict_upcoming

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _db_path(ctx) -> Path:
    return ctx.obj["db_path"] or resolve_path(load_config()["database"]["path"])


def _default_model_path(ctx) -> Path:
    p = _db_path(ctx)
    return p.parent / f"{p.stem}_elo_model.json"


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
        click.echo("next: valpredictor train")


@cli.command("train")
@click.option("--out", default=None, help="output model path (JSON)")
@click.pass_context
def train_cmd(ctx, out: str | None):
    """Replay all stored matches into Elo ratings and fit the model: the one
    number that turns an Elo gap into a per-map win chance."""
    cfg = load_config()
    table = build_map_table(get_connection(_db_path(ctx)), cfg)
    if table.empty:
        click.echo("no played maps in the database yet — run backfill-results first", err=True)
        raise SystemExit(1)
    model = fit_elo_model(table, cfg["model"].get("recency_half_life_days"))
    out_path = resolve_path(out) if out else _default_model_path(ctx)
    model.save(out_path)
    click.echo(
        f"fitted on {model.trained_on_maps} maps: {model.slope:.3f} logit per 100 Elo points "
        f"(textbook Elo is 0.576), saved to {out_path}"
    )


def _fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else "n/a"


@cli.command("backtest")
@click.option("--folds", default=5, help="number of walk-forward folds")
@click.pass_context
def backtest_cmd(ctx, folds: int):
    """Walk-forward backtest of the fitted Elo model against textbook Elo."""
    cfg = load_config()
    results = walk_forward_backtest(build_map_table(get_connection(_db_path(ctx)), cfg), cfg, n_folds=folds)

    click.echo("Per-fold map accuracy (fitted Elo / textbook Elo):")
    for f in results["folds"]:
        c, r = f["calibrated_elo"], f["raw_elo"]
        click.echo(
            f"  fold {f['fold']} [{f['test_start']}..{f['test_end']}] n={c['n']}: "
            f"{_fmt(c['accuracy'])} / {_fmt(r['accuracy'])}   (slope {f['slope']:.3f})"
        )
    click.echo("\nOverall (pooled out-of-fold, per map):")
    for name, res in results["overall"].items():
        click.echo(
            f"  {name:15s} n={res['n']:<6} accuracy={_fmt(res['accuracy'])} "
            f"log_loss={_fmt(res['log_loss'])} brier={_fmt(res['brier'])}"
        )
    click.echo(f"  (a coin flip has log_loss={COIN_FLIP_LOG_LOSS:.4f})")


def _load_model_or_exit(ctx, model_path: str | None) -> EloModel:
    path = resolve_path(model_path) if model_path else _default_model_path(ctx)
    if not path.exists():
        click.echo(f"no model at {path} — run `train` first", err=True)
        raise SystemExit(1)
    return EloModel.load(path)


@cli.command("predict")
@click.option("--team1", required=True)
@click.option("--team2", required=True)
@click.option("--best-of", default=3, type=int)
@click.option("--model", "model_path", default=None)
@click.option("--json", "as_json", is_flag=True, default=False)
@click.pass_context
def predict_cmd(ctx, team1, team2, best_of, model_path, as_json):
    """Predict one matchup: winner probability and map-score distribution."""
    conn = get_connection(_db_path(ctx))
    model = _load_model_or_exit(ctx, model_path)
    try:
        result = predict_match(conn, model, team1, team2, best_of=best_of)
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
        started = (
            f"  (STARTED, {r.team1_score}-{r.team2_score} in maps: this is the pre-game estimate, not updated)\n"
            if r.status == "live" else ""
        )
        market = ""
        if item.market:
            market = (f"\nBookmaker pre-match line: {r.team1_name} {item.market['team1']:.1%} / "
                      f"{r.team2_name} {item.market['team2']:.1%}")
        click.echo(f"\n{header}\n{started}{format_prediction(item.result)}{market}")


@cli.command("serve")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.option("--model", "model_path", default=None)
@click.pass_context
def serve_cmd(ctx, host: str, port: int, model_path: str | None):
    """Run the local web UI (open http://127.0.0.1:8000 in a browser)."""
    from valpredictor.web.server import serve

    path = resolve_path(model_path) if model_path else _default_model_path(ctx)
    model = EloModel.load(path) if path.exists() else None
    if model is None:
        click.echo("no model yet - open the page and click \"Refresh data\" to scrape matches and fit one")
    serve(_db_path(ctx), model, host, port, model_path=path)


if __name__ == "__main__":
    cli()
