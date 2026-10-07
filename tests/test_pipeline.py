import datetime as dt
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from conftest import make_league_db

from valpredictor import pipeline
from valpredictor.storage import db
from valpredictor.web.server import App, _make_handler

FIXTURES = Path(__file__).parent / "fixtures"
EMPTY_PAGE = "<html><body></body></html>"
SINCE = dt.date(2026, 10, 1)


class FakeVLR:
    """Serves fixture pages: the results list on page 1 (empty afterwards) and
    one match page for any match URL. No network."""

    def __init__(self, results_html: str = EMPTY_PAGE, match_html: str = EMPTY_PAGE):
        self.results_html, self.match_html = results_html, match_html
        self.requests: list[str] = []

    def get(self, path: str, force_refresh: bool = False) -> str:
        self.requests.append(path)
        if path.startswith("/matches/results"):
            return self.results_html if path.endswith("page=1") else EMPTY_PAGE
        return self.match_html


def test_refresh_retrains_and_returns_fresh_state(tmp_path):
    conn = make_league_db(tmp_path / "t.db")
    stages = []
    result = pipeline.refresh(
        conn, FakeVLR(), tmp_path / "model.json",
        progress=lambda step, msg: stages.append(step), since=SINCE,
    )
    assert result.trained and result.model is not None and result.state is not None
    assert result.new_matches == 0 and result.maps >= pipeline.MIN_MAPS_TO_TRAIN
    assert (tmp_path / "model.json").exists()
    assert result.latest_match_date
    assert {"scrape", "train"} <= set(stages)


def test_refresh_stores_new_matches_but_declines_to_train_on_too_little_data(tmp_path):
    conn = db.get_connection(tmp_path / "empty.db")
    client = FakeVLR(
        results_html=(FIXTURES / "results_page.html").read_text(encoding="utf-8"),
        match_html=(FIXTURES / "match_detail.html").read_text(encoding="utf-8"),
    )
    result = pipeline.refresh(conn, client, tmp_path / "m.json", since=SINCE)
    assert result.new_matches == 3 and result.total_matches == 3
    assert not result.trained and result.model is None
    assert "need at least" in result.message
    assert not (tmp_path / "m.json").exists()


def test_default_since_resumes_just_before_the_newest_match(tmp_path):
    conn = make_league_db(tmp_path / "t.db")
    newest = dt.date.fromisoformat(conn.execute("SELECT MAX(match_date) FROM matches").fetchone()[0])
    assert pipeline.default_since(conn) == newest - dt.timedelta(days=pipeline.OVERLAP_DAYS)
    assert pipeline.default_since(db.get_connection(tmp_path / "empty.db")) < dt.date.today() - dt.timedelta(days=300)


class BlockingClient(FakeVLR):
    """Holds the results request until released, so a refresh can be observed mid-flight."""

    gate = threading.Event()

    def get(self, path, force_refresh=False):
        if path.startswith("/matches/results"):
            self.gate.wait(15)
        return super().get(path, force_refresh)


def _wait_for(predicate, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_app_refresh_runs_in_background_one_at_a_time_and_swaps_in_the_model(tmp_path):
    BlockingClient.gate = threading.Event()
    make_league_db(tmp_path / "t.db")
    app = App(tmp_path / "t.db", model=None, model_path=tmp_path / "t_elo_model.json", client_factory=BlockingClient)

    assert app.status()["has_model"] is False and app.status()["refresh"]["state"] == "idle"
    assert app.start_refresh() is True
    assert _wait_for(lambda: app.status()["refresh"]["state"] == "running")
    assert app.start_refresh() is False  # a second click while one is running does nothing

    BlockingClient.gate.set()
    assert _wait_for(lambda: app.status()["refresh"]["state"] == "done")
    status = app.status()
    assert status["has_model"] is True and app.model is not None
    assert status["model_trained_at"] is not None and status["maps"] >= pipeline.MIN_MAPS_TO_TRAIN
    assert "refit" in status["refresh"]["message"]
    assert app.start_refresh() is True  # free again afterwards
    assert _wait_for(lambda: app.status()["refresh"]["state"] in ("done", "error"))


def test_predict_without_a_model_explains_what_to_do(tmp_path):
    make_league_db(tmp_path / "t.db")
    app = App(tmp_path / "t.db", model=None, model_path=tmp_path / "m.json", client_factory=FakeVLR)
    with pytest.raises(ValueError, match="Refresh data"):
        app.predict({"team1": "Team 0", "team2": "Team 1"})


def test_http_endpoints_and_cross_site_post_protection(tmp_path):
    make_league_db(tmp_path / "t.db")
    app = App(tmp_path / "t.db", model=None, model_path=tmp_path / "m.json", client_factory=FakeVLR)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status = json.load(urllib.request.urlopen(f"{base}/api/status"))
        assert status["matches"] > 0 and status["refresh"]["state"] == "idle"
        assert b"Refresh data" in urllib.request.urlopen(f"{base}/").read()

        bare = urllib.request.Request(f"{base}/api/refresh", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bare)
        assert exc.value.code == 403  # no X-Requested-With header -> refused

        ok = urllib.request.Request(f"{base}/api/refresh", method="POST", headers={"X-Requested-With": "valpredictor"})
        resp = urllib.request.urlopen(ok)
        assert resp.status == 202 and json.load(resp)["started"] is True
        assert _wait_for(lambda: app.status()["refresh"]["state"] in ("done", "error"))
    finally:
        server.shutdown()
        server.server_close()
