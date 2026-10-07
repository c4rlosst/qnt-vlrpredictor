"""Rate-limited, disk-cached HTTP client for fetching vlr.gg pages.

  - Never re-fetch a URL already on disk (unless force_refresh=True).
  - Never hammer the server: a randomized delay between every live request,
    exponential backoff on 403/429/5xx, and a hard cap on requests per run.
  - vlr.gg's robots.txt only disallows /search/auto and /rr/; this client
    never touches either.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass

import requests

from valpredictor.config import load_config, resolve_path

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a URL could not be fetched after all retries."""


def cache_key_for_url(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return f"{digest}.html"


@dataclass
class ClientStats:
    live_requests: int = 0
    cache_hits: int = 0


class VLRClient:
    """Fetches vlr.gg pages with disk caching and conservative rate limiting."""

    def __init__(self, config: dict | None = None):
        cfg = (config or load_config())["scraping"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.min_delay = float(cfg["min_delay_seconds"])
        self.max_delay = float(cfg["max_delay_seconds"])
        self.max_retries = int(cfg["max_retries"])
        self.backoff_base = float(cfg["backoff_base_seconds"])
        self.user_agent = cfg["user_agent"]
        self.max_requests_per_run = int(cfg["max_requests_per_run"])
        self.cache_dir = resolve_path(cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stats = ClientStats()
        self._last_request_ts: float | None = None
        self._session = self._build_session()

    def _build_session(self):
        session = requests.Session()
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _throttle(self) -> None:
        if self._last_request_ts is not None:
            elapsed = time.monotonic() - self._last_request_ts
            target = random.uniform(self.min_delay, self.max_delay)
            remaining = target - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_ts = time.monotonic()

    def full_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http"):
            return path_or_url
        return f"{self.base_url}/{path_or_url.lstrip('/')}"

    def get(self, path_or_url: str, force_refresh: bool = False) -> str:
        """Return the HTML body for a URL, using the disk cache when possible."""
        url = self.full_url(path_or_url)
        cache_path = self.cache_dir / cache_key_for_url(url)

        if not force_refresh and cache_path.exists():
            self.stats.cache_hits += 1
            return cache_path.read_text(encoding="utf-8")

        html = self._fetch_live(url)
        cache_path.write_text(html, encoding="utf-8")
        return html

    def _fetch_live(self, url: str) -> str:
        if self.stats.live_requests >= self.max_requests_per_run:
            raise FetchError(
                f"Hit max_requests_per_run={self.max_requests_per_run} safety cap. "
                "Re-run the command to continue (already-cached pages will be skipped)."
            )

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=30)
                self.stats.live_requests += 1
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("request error on %s (attempt %d): %s", url, attempt, exc)
            else:
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code in (403, 429) or resp.status_code >= 500:
                    last_exc = FetchError(f"HTTP {resp.status_code} for {url}")
                    logger.warning(
                        "got HTTP %s for %s (attempt %d/%d)",
                        resp.status_code, url, attempt, self.max_retries,
                    )
                else:
                    raise FetchError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]!r}")

            sleep_for = self.backoff_base * (2 ** (attempt - 1)) * random.uniform(0.8, 1.2)
            logger.info("backing off %.1fs before retry", sleep_for)
            time.sleep(sleep_for)

        raise FetchError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_exc
