from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.router import router
from app.config import settings
from app.services.dashboard import ForecastWithheld, QuarterNotAvailable, SubjectNotFound
from app.services.feedback import FeedbackError
from app.services.review import ReviewError, ReviewNotFound
from app.services.ratelimit import RateLimited
from app.services.security import AuthError, ForbiddenError, ServerMisconfigured


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up aiPHeed backend...")
    # Scheduler and model loading will be added in Week 14
    yield
    # Shutdown
    print("Shutting down aiPHeed backend...")


app = FastAPI(
    title="aiPHeed API",
    description="Food Insecurity Forecasting System for DSWD CALABARZON",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: local dev servers + the deployed Vercel frontend. The prod origin is
# required or the browser blocks every dashboard request to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    # Scoped to THIS project's preview deploys. It was https://.*\.vercel\.app
    # -- every Vercel deployment on the internet -- alongside
    # allow_credentials=True, so any site shipped to *.vercel.app could make
    # authenticated cross-origin calls from a signed-in admin's browser.
    allow_origin_regex=settings.CORS_PREVIEW_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    """The error envelope the frontend contract expects."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": {}}},
    )


@app.exception_handler(QuarterNotAvailable)
async def quarter_not_available(request: Request, exc: QuarterNotAvailable):
    """
    404, not an empty forecast.

    The map greys a province out on this code. Returning 0.00 instead would
    render as "no risk" when the truth is "nothing published for that quarter".
    """
    return _error(404, "forecast_not_found", str(exc))


@app.exception_handler(SubjectNotFound)
async def subject_not_found(request: Request, exc: SubjectNotFound):
    return _error(404, "subject_not_found", str(exc))


@app.exception_handler(ForecastWithheld)
async def forecast_withheld(request: Request, exc: ForecastWithheld):
    """
    Also 404, but a different code from forecast_not_found.

    "Never generated" and "generated, then held back by a reviewer" are
    different facts and the map says different things about them. The reason
    itself is not returned here -- GET /api/v1/rejections carries it.
    """
    return _error(404, "forecast_withheld", str(exc))


@app.exception_handler(RateLimited)
async def rate_limited(request: Request, exc: RateLimited):
    """
    429 with Retry-After, so a well-behaved client waits rather than retrying
    into the same wall.
    """
    response = _error(429, "rate_limited", str(exc))
    response.headers["Retry-After"] = str(exc.retry_after)
    return response


@app.exception_handler(AuthError)
async def auth_error(request: Request, exc: AuthError):
    """
    401 with WWW-Authenticate, so a client can tell "sign in" apart from
    "signed in but not allowed".
    """
    response = _error(401, exc.code, str(exc))
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


@app.exception_handler(ForbiddenError)
async def forbidden_error(request: Request, exc: ForbiddenError):
    return _error(403, exc.code, str(exc))


@app.exception_handler(ServerMisconfigured)
async def server_misconfigured(request: Request, exc: ServerMisconfigured):
    """
    Refusing to run insecurely is a server fault, not the caller's. The message
    is safe to surface: it names the setting, never its value.
    """
    return _error(500, "server_misconfigured", str(exc))


@app.exception_handler(ReviewNotFound)
async def review_not_found(request: Request, exc: ReviewNotFound):
    return _error(404, "review_not_found", str(exc))


@app.exception_handler(ReviewError)
async def review_error(request: Request, exc: ReviewError):
    return _error(400, exc.code, str(exc))


@app.exception_handler(FeedbackError)
async def feedback_error(request: Request, exc: FeedbackError):
    return _error(400, exc.code, str(exc))


@app.get("/")
def root():
    return {"message": "Welcome to aiPHeed backend!"}


app.include_router(router)