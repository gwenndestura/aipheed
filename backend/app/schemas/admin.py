"""
app/schemas/admin.py
---------------------
Auth, review pipeline, audit and feedback contracts.

camelCase and slug-keyed like app/schemas/public.py, since the same frontend
consumes both.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: EmailStr
    # No max: a passphrase is a good password. Argon2 handles length fine.
    password: str = Field(min_length=1)


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    role: str

    model_config = {"from_attributes": True}


class LoginResponse(BaseModel):
    token: str
    expiresIn: int
    expiresAt: datetime
    user: UserOut


class LogoutResponse(BaseModel):
    revoked: bool
    message: str


# ---------------------------------------------------------------------------
# Review pipeline
# ---------------------------------------------------------------------------

class ReviewItem(BaseModel):
    id: str
    provinceId: str
    provinceCode: str
    province: str
    quarter: str
    riskScore: float
    riskLevel: str
    status: str                       # Staged | Approved | Rejected
    rejectionReason: str | None = None
    rejectionNotes: str | None = None
    updatedAt: datetime | None = None
    updatedBy: int | None = None
    updatedByName: str | None = None


class ReviewListResponse(BaseModel):
    quarter: str | None = None
    status: str | None = None
    statuses: list[str]
    rejectionReasons: list[str]
    data: list[ReviewItem]


class RejectRequest(BaseModel):
    # Constrained on purpose: a rejection removes a number the public would
    # otherwise see, so "why" has to outlive the person who clicked it.
    reason: str
    notes: str | None = None


class RejectionRow(BaseModel):
    provinceId: str
    province: str
    quarter: str
    reason: str | None = None
    notes: str | None = None
    rejectedBy: int | None = None
    rejectedByName: str | None = None
    timestamp: datetime | None = None


class RejectionsResponse(BaseModel):
    """
    Public. The map greys out these province-quarters.

    A rejection is not the same as a missing forecast: this list is generated
    output an admin withheld, whereas an absent quarter was never produced.
    """
    quarter: str | None = None
    data: list[RejectionRow]


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditRow(BaseModel):
    id: int
    actorEmail: str
    actorId: int | None = None
    action: str
    subjectType: str
    subjectId: str | None = None
    provinceId: str | None = None
    quarter: str | None = None
    reason: str | None = None
    notes: str | None = None
    timestamp: datetime | None = None


class AuditResponse(BaseModel):
    data: list[AuditRow]
    page: int
    pageSize: int
    total: int


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class Demographics(BaseModel):
    fullName: str | None = None
    email: str | None = None
    agency: str | None = None
    designation: str | None = None
    age: str | None = None
    sex: str | None = None
    clientType: str | None = None
    province: str | None = None
    municipality: str | None = None


class FeedbackSubmit(BaseModel):
    demographics: Demographics = Field(default_factory=Demographics)
    # qIndex "0".."9" -> 1..5. Validated and scored server-side; a
    # client-supplied total is never stored.
    answers: dict[str, int]
    liked: str | None = None
    improvements: str | None = None


class FeedbackCreated(BaseModel):
    id: int
    date: datetime
    score: float
    susVersion: str


class FeedbackRow(BaseModel):
    id: int
    date: datetime
    score: float
    susVersion: str
    answers: dict[str, int]
    demographics: Demographics
    liked: str | None = None
    improvements: str | None = None


class FeedbackSummary(BaseModel):
    count: int
    meanScore: float | None = None
    medianScore: float | None = None
    byClientType: dict[str, int]
    susVersion: str


class FeedbackListResponse(BaseModel):
    summary: FeedbackSummary
    questions: list[str]
    data: list[FeedbackRow]
    page: int
    pageSize: int
    total: int
