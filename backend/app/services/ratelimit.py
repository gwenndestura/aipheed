"""
app/services/ratelimit.py
--------------------------
Per-client request limits on the endpoints that can be abused.

Three routes needed this for different reasons:

    POST /api/v1/auth/login   Argon2 makes each guess slow, but nothing stopped
                              a sustained run. Strict limit, and a failed
                              attempt costs the same as a successful one so the
                              limit cannot be probed.
    GET  /api/v1/report       Each call runs a SHAP pass per province and
                              renders a PDF -- seconds of CPU. A handful of
                              concurrent requests saturates the process, so
                              this is a denial-of-service surface, not a
                              correctness one.
    POST /api/v1/feedback     Public, unauthenticated, and writes PII. Without
                              a limit anyone can flood the survey, which
                              corrupts a research result rather than just
                              filling a table.

Deliberately dependency-free. slowapi would be the conventional choice, but its
default backend is also in-process, so it would add a dependency without
changing the guarantee.

KNOWN LIMITS, stated rather than hidden:

* State is per-process. Behind more than one worker each holds its own
  counters, so the effective limit multiplies by the worker count. For a
  single-process deployment -- which is what this runs as -- it is exact.
  A shared Redis counter is the fix if this is ever scaled out.
* The client key is the socket address. Behind a reverse proxy every request
  appears to come from the proxy, so run uvicorn with --proxy-headers and set
  RATE_LIMIT_TRUST_FORWARDED=1, which switches the key to the first hop in
  X-Forwarded-For. It is off by default because trusting that header when the
  app is directly exposed lets a caller forge its own identity.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import Request

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimited(Exception):
    """429 — the caller has spent its allowance for this window."""

    def __init__(self, message: str, retry_after: int, limit: int, window: int):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
        self.window = window


class SlidingWindow:
    """
    Sliding-window counter: timestamps per key, trimmed on each check.

    A fixed window would let a caller spend the whole allowance at the end of
    one window and again at the start of the next -- twice the intended rate
    across the boundary. Sliding avoids that, and at these volumes the memory
    cost of keeping timestamps is trivial.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def check(self, key: str, limit: int, window: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._sweep(now, window)
            hits = self._hits[key]
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(hits[0] + window - now) + 1
                raise RateLimited(
                    f"Too many requests. Limit is {limit} per "
                    f"{window // 60 or 1} minute(s); try again in {retry}s.",
                    retry_after=retry, limit=limit, window=window,
                )
            hits.append(now)

    def _sweep(self, now: float, window: int) -> None:
        """Drop keys that have gone quiet, so the map cannot grow unbounded."""
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        stale = [k for k, v in self._hits.items() if not v or v[-1] <= now - window]
        for k in stale:
            del self._hits[k]
        if stale:
            logger.debug("ratelimit: swept %d idle keys", len(stale))


_window = SlidingWindow()


def client_key(request: Request) -> str:
    """
    Identify the caller.

    X-Forwarded-For is honoured only when explicitly trusted: if the app is
    directly reachable, a caller can set that header freely and mint a fresh
    identity per request, which would make the limit decorative.
    """
    if settings.RATE_LIMIT_TRUST_FORWARDED:
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def limit(request: Request, bucket: str, per_window: int, window_seconds: int) -> None:
    """
    Consume one unit of the caller's allowance for `bucket`.

    Buckets are named rather than derived from the path so that a limit covers
    what it is meant to cover even if a route is later renamed.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return
    key = f"{bucket}:{client_key(request)}"
    _window.check(key, per_window, window_seconds)


# ---------------------------------------------------------------------------
# Route guards
# ---------------------------------------------------------------------------

def login_guard(request: Request) -> None:
    """Counts every attempt, successful or not -- otherwise it is a guess oracle."""
    limit(request, "auth_login",
          settings.RATE_LIMIT_LOGIN_PER_15MIN, 15 * 60)


def report_guard(request: Request) -> None:
    """Seconds of CPU per call; the tightest limit of the three."""
    limit(request, "report",
          settings.RATE_LIMIT_REPORT_PER_HOUR, 60 * 60)


def feedback_guard(request: Request) -> None:
    """Public write of PII into a research dataset."""
    limit(request, "feedback",
          settings.RATE_LIMIT_FEEDBACK_PER_HOUR, 60 * 60)


def reset() -> None:
    """Clear all counters. For tests."""
    with _window._lock:
        _window._hits.clear()
