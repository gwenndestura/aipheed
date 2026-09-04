from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": the .env carries fetcher API keys that individual scripts
    # read directly. Forbidding them made `alembic` and anything importing
    # settings fail on a file that was otherwise valid.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "sqlite+aiosqlite:///./aipheed.db"
    DB_ECHO: bool = False
    SECRET_KEY: str = "changeme"
    MODEL_PATH: str = "models"
    PSA_MAIN_URL: str = "https://psa.gov.ph"
    PSA_RSSO_URL: str = "https://rsso04a.psa.gov.ph"
    DATA_RAW_PATH: str = "data/raw"

    # ── Auth ──────────────────────────────────────────────────────────────
    # Only addresses in this domain may hold an account. The frontend checked
    # this client-side, where anyone could edit it; it is enforced here now.
    ADMIN_EMAIL_DOMAIN: str = "calabarzon.da.gov.ph"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_SECONDS: int = 3600

    # ── Rate limiting ─────────────────────────────────────────────────────
    # Counters are per-process; behind multiple workers the effective limit
    # multiplies by the worker count. See app/services/ratelimit.py.
    RATE_LIMIT_ENABLED: bool = True
    # Only set this when a reverse proxy is definitely in front. If the app is
    # directly reachable, trusting X-Forwarded-For lets a caller forge a fresh
    # identity per request and the limits become decorative.
    RATE_LIMIT_TRUST_FORWARDED: bool = False
    RATE_LIMIT_LOGIN_PER_15MIN: int = 10
    RATE_LIMIT_REPORT_PER_HOUR: int = 60
    RATE_LIMIT_FEEDBACK_PER_HOUR: int = 20

    # ── CORS ──────────────────────────────────────────────────────────────
    # The previous regex was https://.*\.vercel\.app -- ANY Vercel deployment,
    # combined with allow_credentials=True. Anyone could ship a site to
    # attacker.vercel.app and make authenticated cross-origin calls from a
    # signed-in admin's browser. Scoped to this project's own previews.
    CORS_ORIGINS: str = (
        "http://localhost:5173,http://localhost:3000,"
        "https://aipheed-frontend.vercel.app"
    )
    CORS_PREVIEW_REGEX: str = r"https://aipheed-frontend-[a-z0-9-]+\.vercel\.app"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def secret_is_default(self) -> bool:
        """True when SECRET_KEY was never set — every token would be forgeable."""
        return self.SECRET_KEY in ("", "changeme")


settings = Settings()
