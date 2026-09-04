"""
app/services/feedback.py
-------------------------
System Usability Scale submissions.

The survey was scored in the browser and pushed to localStorage, so each admin
only ever saw responses submitted from their own machine. Submissions now
persist server-side and the score is recomputed here -- a client-supplied total
is never stored, because the number goes into a thesis result.

SUS_QUESTIONS is versioned onto every row. If the wording changes, older
responses stay interpretable against the wording that produced them.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FeedbackRecord

logger = logging.getLogger(__name__)

SUS_VERSION = "sus-v1"

# Standard SUS wording. Odd-numbered items (1,3,5,7,9 as presented; indices
# 0,2,4,6,8 here) are positively worded, the rest negatively.
SUS_QUESTIONS = (
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
)

POSITIVE_ITEMS = (0, 2, 4, 6, 8)

AGE_BANDS = ("<=25", "26-35", "36-45", "46-60", ">60")
SEXES = ("Male", "Female")
CLIENT_TYPES = ("Citizen", "Business", "Government", "Others")


class FeedbackError(Exception):
    """400 — the submission is not scorable."""

    def __init__(self, message: str, code: str = "invalid_submission"):
        super().__init__(message)
        self.code = code


def score_sus(answers: dict[str, int]) -> float:
    """
    Standard SUS: positively worded items contribute (answer - 1), negatively
    worded items (5 - answer); the ten sum, times 2.5, gives 0-100.

    All ten answers are required. A partial response cannot be scored on this
    scale, and defaulting the gaps would quietly invent data.
    """
    normalised: dict[int, int] = {}
    for key, value in (answers or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise FeedbackError(f"'{key}' is not a question index.")
        if not 0 <= index <= 9:
            raise FeedbackError(f"Question index {index} is outside 0-9.")
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise FeedbackError(
                f"Answer to question {index} must be an integer 1-5, got {value!r}."
            )
        normalised[index] = value

    missing = [i for i in range(10) if i not in normalised]
    if missing:
        raise FeedbackError(
            "All ten SUS questions must be answered; missing "
            + ", ".join(str(i) for i in missing),
            code="incomplete_submission",
        )

    total = sum(
        (normalised[i] - 1) if i in POSITIVE_ITEMS else (5 - normalised[i])
        for i in range(10)
    )
    return round(total * 2.5, 1)


async def create(db: AsyncSession, payload: dict) -> FeedbackRecord:
    demographics = payload.get("demographics") or {}
    record = FeedbackRecord(
        score=score_sus(payload.get("answers")),
        answers={str(k): int(v) for k, v in (payload.get("answers") or {}).items()},
        sus_version=SUS_VERSION,
        full_name=demographics.get("fullName"),
        email=demographics.get("email"),
        agency=demographics.get("agency"),
        designation=demographics.get("designation"),
        age_band=demographics.get("age"),
        sex=demographics.get("sex"),
        client_type=demographics.get("clientType"),
        province=demographics.get("province"),
        municipality=demographics.get("municipality"),
        liked=payload.get("liked"),
        improvements=payload.get("improvements"),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def listing(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[FeedbackRecord], int, dict]:
    """A page of submissions plus aggregates over the whole set, not the page."""
    total = await db.scalar(select(func.count()).select_from(FeedbackRecord)) or 0

    rows = list((await db.scalars(
        select(FeedbackRecord)
        .order_by(FeedbackRecord.submitted_at.desc(), FeedbackRecord.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )).all())

    scores = list((await db.scalars(select(FeedbackRecord.score))).all())
    client_types = list((await db.scalars(select(FeedbackRecord.client_type))).all())

    summary = {
        "count": total,
        "meanScore": round(statistics.mean(scores), 1) if scores else None,
        "medianScore": round(statistics.median(scores), 1) if scores else None,
        "byClientType": dict(Counter(c for c in client_types if c)),
        "susVersion": SUS_VERSION,
    }
    return rows, total, summary
