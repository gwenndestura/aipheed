"""
app/api/v1/auth.py
-------------------
POST /api/v1/auth/login
GET  /api/v1/auth/me
POST /api/v1/auth/logout

Replaces both client-side login paths. Neither talked to a server: one ignored
the password field entirely, the other compared against a credential compiled
into the JS bundle and printed in the UI. `/admin` was gated by a
sessionStorage flag anyone could set from devtools.

The frontend should call /auth/me on mount rather than trusting its own stored
copy of the session.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AdminUser, AuditRecord
from app.dependencies import get_db
from app.schemas.admin import LoginRequest, LoginResponse, LogoutResponse, UserOut
from app.services import security
from app.services.ratelimit import login_guard

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", response_model=LoginResponse,
             dependencies=[Depends(login_guard)])
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    """
    Exchange credentials for a short-lived bearer token.

    Every failure returns the same message. Saying "no such account" would let
    anyone enumerate which DA addresses are registered.
    """
    email = body.email.strip().lower()

    user = await db.scalar(select(AdminUser).where(AdminUser.email == email))
    password_ok = security.verify_password(body.password, user.password_hash if user else None)

    # Domain is checked at account creation; re-checked here so a row that
    # predates the rule cannot sign in.
    if not (user and password_ok and user.is_active and security.email_domain_allowed(email)):
        logger.warning("failed login for %s from %s", email, request.client.host if request.client else "?")
        raise security.AuthError("Email or password is incorrect.")

    if security.needs_rehash(user.password_hash):
        user.password_hash = security.hash_password(body.password)

    token, expires_in, expires_at = security.create_access_token(user)
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.add(AuditRecord(
        actor_id=user.id,
        actor_email=user.email,
        action="login",
        subject_type="session",
        subject_id=None,
    ))
    await db.commit()

    return LoginResponse(
        token=token,
        expiresIn=expires_in,
        expiresAt=expires_at,
        user=UserOut(id=user.id, email=user.email, name=user.full_name, role=user.role),
    )


@router.get("/me", response_model=UserOut)
async def me(user: AdminUser = Depends(security.current_user)) -> UserOut:
    """Who the bearer token belongs to. 401 if it is missing, expired or revoked."""
    return UserOut(id=user.id, email=user.email, name=user.full_name, role=user.role)


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    user: AdminUser = Depends(security.current_user),
    db: AsyncSession = Depends(get_db),
) -> LogoutResponse:
    """
    Revoke the presented token.

    A JWT stays valid until it expires, so this records the token id in the
    revocation table. Without it, logout would only clear the browser's copy
    while the token itself kept working for the rest of its hour.
    """
    await security.revoke(db, request.state.token_payload)
    return LogoutResponse(
        revoked=True,
        message="Signed out. This token will no longer be accepted.",
    )
