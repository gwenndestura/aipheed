"""
scripts/rest_then_lgu.py
-------------------------
Hands-off driver for the last Google News lever: the per-LGU query set.

That set trips Google's bot detection within minutes at the normal gentle
rate, so this driver:
  1. Rests the IP with ZERO traffic for REST_HOURS (what actually clears a
     block — pausing mid-run does not).
  2. Probes once every 20 min until 3 CONSECUTIVE clean 200s (a single 200
     proved meaningless before).
  3. Runs scripts/gnews_fetch_only.py --profile lgu at an even slower rate
     (6s delay, 5-min window rest) so the sustained request cadence stays
     under the trigger. Per-year checkpoints mean a re-block costs minutes.

Safe to leave running unattended. Re-runnable: completed LGU years
(gnews_lgu_<year>.parquet) are skipped on restart.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

BACKEND = Path(__file__).resolve().parents[1]
REST_HOURS = 4
PROBE_EVERY = 1200          # 20 min
NEED_STREAK = 3

PROBE_URL = ("https://news.google.com/rss/search"
             "?q=Batangas%20food%20price+before:2024-04-01+after:2024-01-01"
             "&hl=en-PH&gl=PH&ceid=PH:en")
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0.0.0 Safari/537.36"),
      "Accept-Language": "en-PH,en;q=0.9"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main() -> None:
    log(f"Resting IP with zero traffic for {REST_HOURS}h before any probe...")
    time.sleep(REST_HOURS * 3600)

    streak = 0
    while streak < NEED_STREAK:
        try:
            r = requests.get(PROBE_URL, headers=UA, timeout=20)
            ok = r.status_code == 200 and b"<item>" in r.content
            streak = streak + 1 if ok else 0
            log(f"probe status={r.status_code} clean_streak={streak}/{NEED_STREAK}")
        except Exception as exc:
            streak = 0
            log(f"probe error: {exc}")
        if streak < NEED_STREAK:
            time.sleep(PROBE_EVERY)

    log("IP stable — launching per-LGU fetch at slow rate.")
    env = dict(os.environ)
    env["AIPHEED_GNEWS_WORKERS"] = "1"
    env["AIPHEED_GNEWS_DELAY"] = "6.0"        # slower than the 3s that blocked
    env["AIPHEED_GNEWS_WINDOW_REST"] = "300"  # 5-min rest between windows
    proc = subprocess.run(
        [sys.executable, "scripts/gnews_fetch_only.py",
         "--start", "2020", "--end", "2025", "--profile", "lgu"],
        cwd=str(BACKEND), env=env,
    )
    log(f"LGU fetch exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
