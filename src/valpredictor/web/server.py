"""Tiny local web UI: team inputs -> prediction, the live bracket, and a
"Refresh data" button that scrapes, rebuilds and retrains in the background.

Standard library only (http.server). Binds to 127.0.0.1 by default; there is
no auth, so don't expose it publicly.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import lightgbm as lgb

from valpredictor.features.build_features import ReplayState, replay
from valpredictor.models.predict import (
    TeamNotFoundError,
    active_map_pool,
    predict_match,
    result_to_json,
)
from valpredictor.pipeline import refresh
from valpredictor.scraping.client import VLRClient
from valpredictor.stage import STAGE_CHOICES
from valpredictor.storage.db import get_connection
from valpredictor.upcoming import predict_upcoming

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).with_name("index.html")
UPCOMING_TTL_SECONDS = 90
NO_MODEL_MESSAGE = "No trained model yet - click \"Refresh data\" to scrape matches and train one."


def _iso(ts: float | None) -> str | None:
    return None if ts is None else dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


class App:
    def __init__(
        self,
        db_path: Path,
        model: lgb.Booster | None,
        model_path: Path,
        client_factory=VLRClient,
    ):
        self.db_path = Path(db_path)
        self.model = model
        self.model_path = Path(model_path)
        self.table_path = self.db_path.parent / f"{self.db_path.stem}_training_table.parquet"
        self.client_factory = client_factory
        self.state: ReplayState = replay(get_connection(self.db_path))
        # The schedule fetch (network, ~20s) is serialised on its own client.
        # Predictions only read the shared model/state, so they never wait on it.
        self.upcoming_lock = threading.Lock()
        self.client = client_factory()
        self._upcoming_cache: dict[str, tuple[float, list]] = {}
        # background "Refresh data" job: one at a time
        self._refresh_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._refresh = {"state": "idle", "step": "", "message": "", "started_at": None, "finished_at": None, "new_matches": 0}

    # ---------------------------------------------------------------- refresh job
    def start_refresh(self) -> bool:
        """Start a refresh in the background. False if one is already running."""
        if not self._refresh_lock.acquire(blocking=False):
            return False
        with self._status_lock:
            self._refresh = {"state": "running", "step": "scrape", "message": "Starting ...",
                             "started_at": time.time(), "finished_at": None, "new_matches": 0}
        threading.Thread(target=self._refresh_worker, name="refresh", daemon=True).start()
        return True

    def _progress(self, step: str, message: str) -> None:
        with self._status_lock:
            self._refresh.update(step=step, message=message)

    def _refresh_worker(self) -> None:
        try:
            result = refresh(
                get_connection(self.db_path), self.client_factory(), self.model_path, self.table_path,
                progress=self._progress,
            )
            self.state = result.state  # swap in the fresh state/model; readers just see the new objects
            if result.model is not None:
                self.model = result.model
            self._upcoming_cache.clear()
            self._finish("done", result.message, result.new_matches)
        except Exception as exc:  # surfaced in the UI; the server keeps running
            logger.exception("refresh failed")
            self._finish("error", f"{type(exc).__name__}: {exc}", 0)
        finally:
            self._refresh_lock.release()

    def _finish(self, state: str, message: str, new_matches: int) -> None:
        with self._status_lock:
            self._refresh.update(state=state, step="", message=message, finished_at=time.time(), new_matches=new_matches)

    def status(self) -> dict:
        conn = get_connection(self.db_path)
        matches, latest = conn.execute("SELECT COUNT(*), MAX(match_date) FROM matches").fetchone()
        maps = conn.execute("SELECT COUNT(*) FROM maps WHERE team1_score IS NOT NULL").fetchone()[0]
        with self._status_lock:
            refresh_state = dict(self._refresh)
        refresh_state["started_at"] = _iso(refresh_state["started_at"])
        refresh_state["finished_at"] = _iso(refresh_state["finished_at"])
        trained_at = _iso(self.model_path.stat().st_mtime) if self.model_path.exists() else None
        return {
            "matches": matches,
            "maps": maps,
            "latest_match_date": latest,
            "has_model": self.model is not None,
            "model_trained_at": trained_at,
            "refresh": refresh_state,
        }

    # ---------------------------------------------------------------- data for the page
    def _require_model(self) -> lgb.Booster:
        if self.model is None:
            raise ValueError(NO_MODEL_MESSAGE)
        return self.model

    def teams(self) -> list[dict]:
        conn = get_connection(self.db_path)
        state = self.state
        rows = conn.execute(
            """
            SELECT t.id, t.name, COUNT(m.id) AS matches, MAX(m.match_date) AS last_played
            FROM teams t JOIN matches m ON t.id IN (m.team1_id, m.team2_id)
            GROUP BY t.id ORDER BY MAX(m.match_date) DESC
            """
        ).fetchall()
        out = [
            {
                "name": r["name"],
                "matches": r["matches"],
                "last_played": r["last_played"],
                "elo": round(state.elo.rating(r["id"])),
            }
            for r in rows
        ]
        return sorted(out, key=lambda t: -t["elo"])

    def predict(self, params: dict[str, str]) -> dict:
        model, state = self._require_model(), self.state
        conn = get_connection(self.db_path)
        maps = [m.strip() for m in params.get("maps", "").split(",") if m.strip()] or None
        best_of = int(params.get("best_of", "3"))
        international = params.get("international") == "1"
        result = predict_match(
            conn, model, params["team1"], params["team2"], best_of=best_of, maps=maps,
            is_international=True if international else None, state=state,
            event_tier=2 if international else 1, stakes=STAGE_CHOICES.get(params.get("stage", "")),
        )
        return result_to_json(result)

    def upcoming(self, event: str, force: bool = False) -> list[dict]:
        model = self._require_model()
        cached = self._upcoming_cache.get(event)
        if cached and not force and time.time() - cached[0] < UPCOMING_TTL_SECONDS:
            return cached[1]
        conn = get_connection(self.db_path)
        with self.upcoming_lock:
            cached = self._upcoming_cache.get(event)  # another request may have just refreshed it
            if cached and not force and time.time() - cached[0] < UPCOMING_TTL_SECONDS:
                return cached[1]
            items = predict_upcoming(conn, self.client, model, event, state=self.state)
        data = []
        for it in items:
            r = it.row
            data.append(
                {
                    "match_id": r.vlr_match_id,
                    "url": f"https://www.vlr.gg{r.match_url}",
                    "date": r.date_label,
                    "time": r.time_label,
                    "series": r.series,
                    "status": r.status,
                    "team1": r.team1_name,
                    "team2": r.team2_name,
                    "note": it.note,
                    "veto_posted": it.veto_posted,
                    "market": it.market,
                    "prediction": result_to_json(it.result) if it.result else None,
                }
            )
        self._upcoming_cache[event] = (time.time(), data)
        return data


def _make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_HEAD(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def do_GET(self):  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/teams":
                    self._json(app.teams())
                elif url.path == "/api/status":
                    self._json(app.status())
                elif url.path == "/api/maps":
                    self._json(active_map_pool(get_connection(app.db_path)))
                elif url.path == "/api/predict":
                    self._json(app.predict(params))
                elif url.path == "/api/upcoming":
                    self._json(app.upcoming(params.get("event", "Champions"), force=params.get("refresh") == "1"))
                else:
                    self._json({"error": "not found"}, 404)
            except (TeamNotFoundError, KeyError, ValueError) as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:  # keep the server alive; surface the message in the UI
                logger.exception("request failed: %s", self.path)
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            # A custom header can't be sent cross-origin without a CORS preflight (which we never
            # grant), so another website open in your browser can't trigger a scrape on this server.
            if self.headers.get("X-Requested-With") != "valpredictor":
                self._json({"error": "forbidden"}, 403)
            elif url.path == "/api/refresh":
                self._json({"started": app.start_refresh()}, 202)
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, fmt, *args):
            logger.info("%s %s", self.address_string(), fmt % args)

    return Handler


def serve(
    db_path: Path, model: lgb.Booster | None, host: str = "127.0.0.1", port: int = 8000,
    model_path: Path | None = None,
) -> None:
    db_path = Path(db_path)
    model_path = Path(model_path) if model_path else db_path.parent / f"{db_path.stem}_map_model.txt"
    app = App(db_path, model, model_path)
    server = ThreadingHTTPServer((host, port), _make_handler(app))
    print(f"valpredictor web UI on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
