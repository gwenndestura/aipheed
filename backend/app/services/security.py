"""
app/services/security.py
-------------------------
Password hashing, token issuing, and the dependencies that gate admin routes.

What this replaces: two client-side checks that never talked to a server. One
accepted any password against an @calabarzon.da.gov.ph address; the other
compared against `admin2026` hardcoded in the bundle. Both wrote a
sessionStorage flag that anyone could set from devtools.

Design notes worth keeping:

* Argon2id for passwords. It is already a dependency and is the current
  recommendation over bcrypt.
* A wrong password and an unknown email return the same error and both pay the
  hashing cost, so response timing does not disclose which accounts exist.
* Logout genuinely revokes. A JWT is otherwise valid until it expires, so
  without the revocation table logout would only clear the browser's copy.
* Tokens are refused outright while SECRET_KEY is the shipped default -- with
  a known key anyone can mint an admin token, so failing loudly beats issuing
  a forgeable one.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import Depends, Request
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import AdminUser, RevokedToken
from app.dependencies import get_db

logger = logging.getLogger(__name__)

_hasher = PasswordHasher()

ISSUER = "aipheed-api"

# A dummy hash to verify against when the email is unknown, so the failure path
# costs the same as a real password check.
_DUMMY_HASH = _hasher.hash("no-such-account")


class AuthError(Exception):
    """401 — the caller is not authenticated."""

    def __init__(self, message: str, code: str = "invalid_credentials"):
        super().__init__(message)
        self.code = code


class ForbiddenError(Exception):
    """403 — authenticated, but not allowed to do this."""

    def __init__(self, message: str, code: str = "forbidden"):
        super().__init__(message)
        self.code = code


class ServerMisconfigured(Exception):
    """500 — refusing to operate insecurely."""


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, stored_hash: str | None) -> bool:
    """
    Constant-ish time check. An absent hash still runs a verification against
    the dummy so an unknown account is not faster than a wrong password.
    """
    try:
        _hasher.verify(stored_hash or _DUMMY_HASH, plain)
        return stored_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True when the hash was made with weaker parameters than current policy."""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:
        return True


def email_domain_allowed(email: str) -> bool:
    return email.strip().lower().endswith("@" + settings.ADMIN_EMAIL_DOMAIN.lower())


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def _assert_signable() -> None:
    if settings.secret_is_default:
        raise ServerMisconfigured(
            "SECRET_KEY is still the default value. Set a strong SECRET_KEY in "
            "the environment before issuing tokens -- with the shipped default "
            "anyone can mint a valid admin token."
        )


def create_access_token(user: AdminUser) -> tuple[str, int, datetime]:
    """Return (token, expires_in_seconds, absolute_expiry)."""
    _assert_signable()
    now = datetime.now(timezone.utc)
    ttl = settings.ACCESS_TOKEN_TTL_SECONDS
    expires_at = now + timedelta(seconds=ttl)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.full_name,
        "role": user.role,
        "iss": ISSUER,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, ttl, expires_at


def decode_token(token: str) -> dict:
    _assert_signable()
    try:
        return jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            issuer=ISSUER,
            options={"require": ["exp", "sub", "jti"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Session expired. Sign in again.", code="token_expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid session token.", code="invalid_token")


def bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        raise AuthError("Authorization header must be 'Bearer <token>'.",
                        code="missing_token")
    return value.strip()


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------

async def revoke(db: AsyncSession, payload: dict) -> None:
    """Record a token as revoked and drop rows that have expired anyway."""
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc).replace(tzinfo=None)
    exists = await db.scalar(select(RevokedToken).where(RevokedToken.jti == payload["jti"]))
    if exists is None:
        db.add(RevokedToken(jti=payload["jti"], expires_at=expires_at))
    await db.execute(
        delete(RevokedToken).where(RevokedToken.expires_at < datetime.now(timezone.utc).replace(tzinfo=None))
    )
    await db.commit()


async def is_revoked(db: AsyncSession, jti: str) -> bool:
    return await db.scalar(select(RevokedToken).where(RevokedToken.jti == jti)) is not None


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AdminUser:
    """
    Resolve the signed-in account, or raise 401.

    The token is only a claim; the account is re-read every request so a
    deactivated user stops working immediately instead of at token expiry.
    """
    payload = decode_token(bearer_token(request))

    if await is_revoked(db, payload["jti"]):
        raise AuthError("Session has been signed out.", code="token_revoked")

    user = await db.scalar(select(AdminUser).where(AdminUser.id == int(payload["sub"])))
    if user is None or not user.is_active:
        raise AuthError("Account is no longer active.", code="account_inactive")

    request.state.token_payload = payload
    return user


async def require_admin(user: AdminUser = Depends(current_user)) -> AdminUser:
    """Gate for /api/v1/admin/**. 401 without a token, 403 with the wrong role."""
    if user.role != "admin":
        raise ForbiddenError("This action requires an admin account.")
    return user
