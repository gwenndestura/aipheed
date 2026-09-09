"""
scripts/refresh_all_sources.py
-------------------------------
Re-fetch every source, rebuild the feature matrix, and report what each series
actually covers.

WHY THIS EXISTS
---------------
Before 2026-09-09 there was no single way to advance the data. Five "fetchers"
were hand-typed literal tables that ended at 2025-Q4, the model window was a
hardcoded 2020-2025 range, and the news window was a separate hardcoded
2020-Q1..2025-Q4 pair. Nothing announced that it had stopped, so the feature
matrix sat at 2025-Q4 while the production panel had already run to 2026-Q2.

This script is the answer to "how do I pull the latest data". Run it, read the
coverage table at the end, and act on anything marked STALE.

WHAT IS LIVE AND WHAT IS NOT
----------------------------
LIVE -- extends itself every run:
    PSA OpenStat        production panel, rice wholesale prices, CPI
    NOAA CPC            ENSO (Oceanic Nino Index)
    NOAA IBTrACS        tropical cyclone best track
    FRED                Brent crude spot
    ECB / Frankfurter   USD/PHP
    NASA POWER          province rainfall

CURATED -- cannot be fetched; each raises CuratedCoverageError past its range:
    DOE     pump prices     doe.gov.ph WAF times out after the first request
    BSP     OFW remittances www.bsp.gov.ph returns 403 to programmatic clients
    SWS     hunger survey   press-release HTML, layout varies per release

Those three are run with strict=False here so a refresh completes end to end,
and every degraded series is listed in the closing report.

USAGE
-----
    cd backend
    python scripts/refresh_all_sources.py
    python scripts/refresh_all_sources.py --skip-panel   # skip the slow PSA pull
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("refresh")

PROCESSED = Path("data/processed")


def _run_fetcher(name: str, fn, dest: str, **kwargs) -> tuple[str, str]:
    """Run one fetcher, write its parquet, return (status, detail)."""
    t0 = time.time()
    try:
        df = fn(**kwargs)
    except Exception as exc:                       # noqa: BLE001
        log.error("%-22s FAILED: %s", name, str(exc)[:160])
        return "FAILED", str(exc)[:160]
    path = PROCESSED / dest
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    span = (f"{df['quarter'].min()} .. {df['quarter'].max()}"
            if "quarter" in df.columns else f"{len(df)} rows")
    log.info("%-22s ok  %-24s %5.1fs  -> %s", name, span, time.time() - t0, dest)
    return "ok", span


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh every aiPHeed data source")
    ap.add_argument("--skip-panel", action="store_true",
                    help="skip the PSA production panel (the slowest pull)")
    args = ap.parse_args()

    from app.ml.corpus.bsp_macro_fetcher import fetch_bsp_macro
    from app.ml.corpus.cpi_full_fetcher import fetch_cpi_full
    from app.ml.corpus.oil_price_fetcher import fetch_oil_prices
    from app.ml.corpus.pagasa_climate_fetcher import fetch_pagasa_climate
    from app.ml.corpus.psa_rice_fetcher import fetch_psa_rice_prices
    from app.ml.corpus.sws_hunger_fetcher import fetch_sws_hunger
    from app.ml.features.feature_matrix import MODEL_QUARTERS

    window_end = MODEL_QUARTERS[-1]
    log.info("target window: %s .. %s", MODEL_QUARTERS[0], window_end)

    results: dict[str, tuple[str, str]] = {}

    log.info("--- live sources ---")
    results["climate (NOAA)"] = _run_fetcher(
        "pagasa_climate", fetch_pagasa_climate, "pagasa_climate.parquet",
        start_year=2021)
    results["rice (PSA)"] = _run_fetcher(
        "psa_rice", fetch_psa_rice_prices, "psa_rice_prices.parquet",
        start_year=2021)
    results["cpi (PSA)"] = _run_fetcher(
        "cpi_full", fetch_cpi_full, "cpi_full.parquet",
        start_year=2021, end_year=int(window_end[:4]))

    # strict=False: these three are curated and will not reach the window edge.
    # The closing report names every one that fell short.
    log.info("--- part-curated sources (degraded output permitted) ---")
    results["fuel (FRED + DOE)"] = _run_fetcher(
        "oil_prices", fetch_oil_prices, "oil_prices.parquet",
        start_year=2021, strict=False)
    results["macro (ECB + BSP)"] = _run_fetcher(
        "bsp_macro", fetch_bsp_macro, "bsp_macro.parquet",
        start_year=2021, strict=False)
    results["hunger (SWS)"] = _run_fetcher(
        "sws_hunger", fetch_sws_hunger, "sws_hunger.parquet",
        start_year=2021, strict=False)

    if not args.skip_panel:
        log.info("--- PSA production panel (slow) ---")
        rc = subprocess.run(
            [sys.executable, "scripts/build_food_availability_panel.py"],
            check=False).returncode
        results["panel (PSA)"] = (("ok", "rebuilt") if rc == 0
                                  else ("FAILED", f"exit {rc}"))
    else:
        log.info("--- PSA production panel SKIPPED (--skip-panel) ---")

    log.info("--- rainfall (NASA POWER) ---")
    rc = subprocess.run([sys.executable, "scripts/build_province_rainfall.py"],
                        check=False).returncode
    results["rainfall (NASA)"] = (("ok", "rebuilt") if rc == 0
                                  else ("FAILED", f"exit {rc}"))

    log.info("--- FSSI chain + feature matrix ---")
    rc = subprocess.run([sys.executable, "scripts/rebuild_fssi_chain.py"],
                        check=False).returncode
    results["FSSI + matrix"] = (("ok", "rebuilt") if rc == 0
                                else ("FAILED", f"exit {rc}"))

    # ── coverage report ───────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"COVERAGE REPORT   target window {MODEL_QUARTERS[0]} .. {window_end}")
    print("=" * 78)
    print(f"{'source':22s} {'status':8s} {'span':24s} verdict")
    print("-" * 78)
    stale = []
    for name, (status, detail) in results.items():
        verdict = ""
        if status == "ok" and ".." in detail:
            end = detail.split("..")[-1].strip()
            if end < window_end:
                verdict = f"STALE -- {end} < {window_end}"
                stale.append((name, end))
            else:
                verdict = "current"
        elif status == "FAILED":
            verdict = "FAILED"
            stale.append((name, "failed"))
        print(f"{name:22s} {status:8s} {detail[:24]:24s} {verdict}")
    print("=" * 78)

    prov = PROCESSED / "feature_provenance.json"
    if prov.exists():
        print(f"\nImputation record: {prov}")
        print("  Any feature listed there is carried forward, not observed. "
              "Do not report those cells as measured.")

    if stale:
        print("\nACTION NEEDED — these did not reach the window edge:")
        for name, end in stale:
            print(f"  - {name}: {end}")
        print("\n  DOE pump prices, BSP remittances and SWS hunger are curated "
              "by hand;\n  extend them in their fetcher modules. Everything "
              "else should be reachable\n  and a shortfall there means an "
              "upstream API changed.")
    else:
        print("\nAll sources reached the window edge.")

    failed = [n for n, (s, _) in results.items() if s == "FAILED"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
