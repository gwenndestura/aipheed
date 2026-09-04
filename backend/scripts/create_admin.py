"""
scripts/create_admin.py
------------------------
Create or update a DA CALABARZON admin account.

There is deliberately no seeded default account. The old console shipped
admin@calabarzon.da.gov.ph / admin2026 compiled into the JS bundle and printed
in the UI; anything comparable committed here would be the same mistake in a
new place.

Usage
-----
    python scripts/create_admin.py --email analyst@calabarzon.da.gov.ph \
                                   --name "J. Dela Cruz"

The password is read from a hidden prompt, or from the AIPHEED_ADMIN_PASSWORD
environment variable for non-interactive setup. It is never taken from a
command-line flag, where it would land in shell history and the process list.

    --deactivate    disable an existing account without deleting its audit trail
    --list          show existing accounts
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db.database import AsyncSessionLocal  # noqa: E402
from app.db.models import AdminUser  # noqa: E402
from app.services import security  # noqa: E402

MIN_PASSWORD_LENGTH = 12


def read_password() -> str:
    """Hidden prompt with confirmation, or the env var when non-interactive."""
    from_env = os.environ.get("AIPHEED_ADMIN_PASSWORD")
    if from_env:
        return from_env

    first = getpass.getpass("Password: ")
    second = getpass.getpass("Confirm password: ")
    if first != second:
        sys.exit("Passwords do not match.")
    return first


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters. "
            "A passphrase of a few words is easier to remember and stronger "
            "than a short complex string."
        )


async def list_accounts() -> None:
    async with AsyncSessionLocal() as db:
        rows = list((await db.scalars(select(AdminUser).order_by(AdminUser.email))).all())
    if not rows:
        print("No admin accounts exist yet.")
        return
    print(f"{'EMAIL':<40} {'NAME':<24} {'ROLE':<8} ACTIVE  LAST LOGIN")
    for u in rows:
        last = u.last_login_at.strftime("%Y-%m-%d %H:%M") if u.last_login_at else "never"
        print(f"{u.email:<40} {u.full_name:<24} {u.role:<8} "
              f"{'yes' if u.is_active else 'no':<7} {last}")


async def deactivate(email: str) -> None:
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(AdminUser).where(AdminUser.email == email))
        if user is None:
            sys.exit(f"No account for {email}.")
        user.is_active = False
        await db.commit()
    # The row is kept so the audit trail still resolves who did what.
    print(f"Deactivated {email}. Their audit history is retained.")


async def upsert(email: str, name: str, password: str) -> None:
    async with AsyncSessionLocal() as db:
        user = await db.scalar(select(AdminUser).where(AdminUser.email == email))
        if user is None:
            db.add(AdminUser(
                email=email,
                full_name=name,
                password_hash=security.hash_password(password),
                role="admin",
                is_active=True,
            ))
            action = "Created"
        else:
            user.full_name = name or user.full_name
            user.password_hash = security.hash_password(password)
            user.is_active = True
            action = "Updated"
        await db.commit()
    print(f"{action} admin account {email}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument("--email")
    parser.add_argument("--name", default="")
    parser.add_argument("--list", action="store_true", help="show existing accounts")
    parser.add_argument("--deactivate", action="store_true", help="disable the account")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_accounts())
        return

    if not args.email:
        sys.exit("--email is required (or pass --list).")

    email = args.email.strip().lower()
    if not security.email_domain_allowed(email):
        sys.exit(
            f"{email} is outside the permitted domain "
            f"@{settings.ADMIN_EMAIL_DOMAIN}. Change ADMIN_EMAIL_DOMAIN if that "
            "is wrong."
        )

    if args.deactivate:
        asyncio.run(deactivate(email))
        return

    if settings.secret_is_default:
        # Accounts would work, but every token minted against them would be
        # forgeable, so fix the key before creating the first one.
        sys.exit(
            "SECRET_KEY is still 'changeme'. Set a strong random SECRET_KEY in "
            "backend/.env first -- otherwise anyone can forge an admin token "
            "and the account you are about to create is meaningless.\n\n"
            "  python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    if not args.name:
        sys.exit("--name is required when creating or updating an account.")

    password = read_password()
    check_password(password)
    asyncio.run(upsert(email, args.name.strip(), password))


if __name__ == "__main__":
    main()
