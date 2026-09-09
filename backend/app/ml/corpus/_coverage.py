"""
app/ml/corpus/_coverage.py
---------------------------
Shared coverage guard for the hand-curated series.

Three sources in this package cannot be fetched programmatically:

    DOE pump prices     doe.gov.ph serves the first request then times out
    BSP OFW remittances www.bsp.gov.ph returns 403 to programmatic clients
    SWS hunger survey   press-release HTML, layout varies between releases

Their values are curated by hand, so each one has a last quarter it actually
covers. Before this guard existed those modules simply returned a shorter
frame when asked for a later quarter, and the caller could not tell a complete
series from a truncated one -- which is how the whole feature matrix came to
stop at 2025-Q4 while the production panel ran on to 2026-Q2.

The check is quarter-accurate, not year-accurate. A year-granular comparison
passes a request for "2026" against a series that stops at 2026-Q2, which is
the exact hole this is meant to close.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone


class CuratedCoverageError(RuntimeError):
    """Raised when a hand-curated series is asked for a quarter it does not have."""


def last_completed_quarter(today: date | None = None) -> tuple[int, int]:
    """The most recent quarter that has actually finished, as (year, quarter)."""
    today = today or datetime.now(timezone.utc).date()
    q = (today.month - 1) // 3 + 1
    return (today.year - 1, 4) if q == 1 else (today.year, q - 1)


def check_coverage(*, series: str, coverage_end: tuple[int, int], end_year: int,
                   reason: str, source_url: str, extend_hint: str,
                   strict: bool, logger: logging.Logger) -> None:
    """
    Raise (or warn loudly) when the requested window outruns curated coverage.

    The requested window is read as "through the last completed quarter of
    end_year", so asking for the current year checks only quarters that have
    actually closed.
    """
    now_y, now_q = last_completed_quarter()
    wanted = (end_year, 4) if end_year < now_y else (min(end_year, now_y), now_q)
    if wanted <= coverage_end:
        return

    msg = (
        f"{series} is curated only through {coverage_end[0]}-Q{coverage_end[1]}, "
        f"but the window reaches {wanted[0]}-Q{wanted[1]}. {reason} "
        f"Add the missing quarters from {source_url} and {extend_hint}, or "
        f"call with strict=False to accept a series that stops at "
        f"{coverage_end[0]}-Q{coverage_end[1]}."
    )
    if strict:
        raise CuratedCoverageError(msg)
    logger.warning("=" * 78)
    logger.warning("DEGRADED OUTPUT: %s", msg)
    logger.warning("=" * 78)
