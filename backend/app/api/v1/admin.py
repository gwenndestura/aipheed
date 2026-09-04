"""
app/api/v1/admin.py
--------------------
The review pipeline and its public face.

admin_router  (auth: admin)   mounted at /api/v1/admin
    GET    /review?quarter=&status=
    POST   /review/{id}/approve
    POST   /review/{id}/reject
    POST   /review/{id}/undo
    DELETE /rejections/{province_id}/{quarter}
    GET    /audit
    GET    /feedback

public_router (no auth)       mounted at /api/v1
    GET    /rejections?quarter=
    POST   /feedback

Rejections are public because the map needs them to grey provinces out.
Everything that *changes* one is admin-only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AdminUser, AuditRecord, FeedbackRecord, ReviewRecord
from app.dependencies import get_db
from app.schemas.admin import (
    AuditResponse,
    AuditRow,
    Demographics,
    FeedbackCreated,
    FeedbackListResponse,
    FeedbackRow,
    FeedbackSubmit,
    FeedbackSummary,
    RejectionRow,
    RejectionsResponse,
    RejectRequest,
    ReviewItem,
    ReviewListResponse,
)
from app.services import dashboard as svc
from app.services import feedback as fb
from app.services import review as rv
from app.services.ratelimit import feedback_guard
from app.services.security import require_admin

logger = logging.getLogger(__name__)

admin_router = APIRouter(dependencies=[Depends(require_admin)])
public_router = APIRouter()


# ---------------------------------------------------------------------------
# Serialisers
# ---------------------------------------------------------------------------

def _item(row: ReviewRecord) -> ReviewItem:
    return ReviewItem(
        id=row.id,
        provinceId=row.province_id,
        provinceCode=row.province_code,
        province=row.province_name,
        quarter=row.quarter,
        riskScore=row.risk_score,
        riskLevel=row.risk_level,
        status=row.status,
        rejectionReason=row.rejection_reason,
        rejectionNotes=row.rejection_notes,
        updatedAt=row.updated_at,
        updatedBy=row.updated_by,
        updatedByName=row.updated_by_name,
    )


def _rejection(row: ReviewRecord) -> RejectionRow:
    return RejectionRow(
        provinceId=row.province_id,
        province=row.province_name,
        quarter=row.quarter,
        reason=row.rejection_reason,
        notes=row.rejection_notes,
        rejectedBy=row.updated_by,
        rejectedByName=row.updated_by_name,
        timestamp=row.updated_at,
    )


def _feedback_row(row: FeedbackRecord) -> FeedbackRow:
    return FeedbackRow(
        id=row.id,
        date=row.submitted_at,
        score=row.score,
        susVersion=row.sus_version,
        answers=row.answers,
        demographics=Demographics(
            fullName=row.full_name,
            email=row.email,
            agency=row.agency,
            designation=row.designation,
            age=row.age_band,
            sex=row.sex,
            clientType=row.client_type,
            province=row.province,
            municipality=row.municipality,
        ),
        liked=row.liked,
        improvements=row.improvements,
    )


def _audit_row(row: AuditRecord) -> AuditRow:
    return AuditRow(
        id=row.id,
        actorEmail=row.actor_email,
        actorId=row.actor_id,
        action=row.action,
        subjectType=row.subject_type,
        subjectId=row.subject_id,
        provinceId=row.province_id,
        quarter=row.quarter,
        reason=row.reason,
        notes=row.notes,
        timestamp=row.timestamp,
    )


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

@admin_router.get("/review", response_model=ReviewListResponse)
async def list_review(
    quarter: str | None = Query(None, description="Defaults to the latest scorable quarter"),
    status: str | None = Query(None, description="Staged | Approved | Rejected"),
    db: AsyncSession = Depends(get_db),
) -> ReviewListResponse:
    """
    The review queue.

    Rows are seeded from real model output the first time a quarter is opened,
    and an existing decision is never reset by re-seeding. Omit `quarter` to
    see every quarter at once.
    """
    resolved = svc.resolve_quarter(quarter) if quarter else None
    rows = await rv.list_items(db, quarter=resolved, status=status)
    return ReviewListResponse(
        quarter=resolved,
        status=status,
        statuses=list(rv.STATUSES),
        rejectionReasons=list(rv.REJECTION_REASONS),
        data=[_item(r) for r in rows],
    )


@admin_router.post("/review/{item_id}/approve", response_model=ReviewItem)
async def approve(
    item_id: str,
    user: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> ReviewItem:
    """Publish this province-quarter."""
    return _item(await rv.approve(db, item_id, user))


@admin_router.post("/review/{item_id}/reject", response_model=ReviewItem)
async def reject(
    item_id: str,
    body: RejectRequest,
    user: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> ReviewItem:
    """
    Withhold this province-quarter from the public map.

    Takes effect immediately: /forecast stops resolving for the pair and
    /rejections lists it.
    """
    return _item(await rv.reject(db, item_id, user, body.reason, body.notes))


@admin_router.post("/review/{item_id}/undo", response_model=ReviewItem)
async def undo(
    item_id: str,
    user: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> ReviewItem:
    """Return an Approved or Rejected item to Staged."""
    return _item(await rv.undo(db, item_id, user))


@admin_router.delete("/rejections/{province_id}/{quarter}", response_model=ReviewItem)
async def restore(
    province_id: str,
    quarter: str,
    user: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> ReviewItem:
    """Lift a rejection so the forecast publishes again."""
    return _item(await rv.restore(db, province_id.lower(), quarter, user))


@admin_router.get("/audit", response_model=AuditResponse)
async def audit(
    page: int = Query(1, ge=1),
    pageSize: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> AuditResponse:
    """Who changed what, when, and why. Append-only."""
    rows, total = await rv.list_audit(db, page, pageSize)
    return AuditResponse(
        data=[_audit_row(r) for r in rows],
        page=page,
        pageSize=pageSize,
        total=total,
    )


@admin_router.get("/feedback", response_model=FeedbackListResponse)
async def list_feedback(
    page: int = Query(1, ge=1),
    pageSize: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> FeedbackListResponse:
    """
    Every SUS submission, with aggregates over the whole set.

    Previously each admin saw only what had been submitted from their own
    browser; this is the first view across all respondents.
    """
    rows, total, summary = await fb.listing(db, page, pageSize)
    return FeedbackListResponse(
        summary=FeedbackSummary(**summary),
        questions=list(fb.SUS_QUESTIONS),
        data=[_feedback_row(r) for r in rows],
        page=page,
        pageSize=pageSize,
        total=total,
    )


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

@public_router.get("/rejections", response_model=RejectionsResponse)
async def rejections(
    quarter: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
) -> RejectionsResponse:
    """
    Province-quarters an admin has withheld. No auth: the public map needs it.

    Distinct from a missing forecast -- these were generated and then held
    back, which is why they carry a reason and an actor.
    """
    rows = await rv.list_rejections(db, quarter)
    return RejectionsResponse(quarter=quarter, data=[_rejection(r) for r in rows])


@public_router.post("/feedback", response_model=FeedbackCreated, status_code=201,
                    dependencies=[Depends(feedback_guard)])
async def submit_feedback(
    body: FeedbackSubmit,
    db: AsyncSession = Depends(get_db),
) -> FeedbackCreated:
    """
    Record a SUS response. No auth -- the survey is open to any dashboard user.

    The score returned is the server's own calculation from `answers`.
    """
    record = await fb.create(db, body.model_dump())
    return FeedbackCreated(
        id=record.id,
        date=record.submitted_at,
        score=record.score,
        susVersion=record.sus_version,
    )
